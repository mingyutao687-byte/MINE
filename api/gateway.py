"""MINE 主调度网关 — 基于 FastAPI 的 HTTP 服务 (端口 7000)。

提供以下核心 API:
  - POST /ask_for_quota: Worker 向调度器请求执行配额
  - POST /worker_go_idle: Worker 通知调度器自己进入空闲
  - POST /v1/completions: 接收推理请求（类 OpenAI API）
  - POST /set_config: 动态修改运行时配置
  - POST /get_config: 获取当前配置
  - POST /start_monitor: 开始监控
  - POST /end_monitor: 结束监控并返回日志

架构:
  Gateway (FastAPI) → PoolManager → Pool → Node → Worker → HTTP → vLLM Worker
"""

import asyncio
import time
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
import aiohttp
import uvicorn

from Mine.config import settings as config
from Mine.config.settings import scheduler_config
from Mine.models.request_tracker import ReqTracker
from Mine.models.enums import WorkerHangingReleaseType
from Mine.core.pool import PoolManager

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s:%(lineno)d %(message)s',
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# FastAPI 应用
app = FastAPI()

# 全局 PoolManager (在 lifespan 中初始化)
pool_manager: PoolManager


# ====================================================================
# 生命周期
# ====================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理。

    启动时:
      - 创建 aiohttp 会话（连接池 1024, force_close）
      - 创建 PoolManager 实例
      - 等待所有 Worker 启动完成

    关闭时:
      - aiohttp 会话自动清理
    """
    global pool_manager
    logger.warning("Starting up...")
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=1024, force_close=True),
        timeout=aiohttp.ClientTimeout(),
    ) as session:
        pool_manager = PoolManager(config.pools_config, session)
        await pool_manager.check_start_complete()
        logger.warning('Start-up complete')
        yield
    logger.warning("Shutting down...")


app.router.lifespan_context = lifespan


# ====================================================================
# POST /ask_for_quota — Worker 配额请求
# ====================================================================

@app.post("/ask_for_quota")
async def ask_for_quota(request: Request):
    """Worker 向调度器请求执行配额。

    流程:
      1. 标记 Worker 为挂起状态 (being_hanged=True)
      2. 处理 sllm+share 的计费逻辑
      3. 检查是否有待执行的 Action
      4. 若 Worker 可调度且有权限 → 触发 Node.schedule()
      5. 阻塞等待调度事件:
         - being_scheduled: 被选中，获得配额
         - finished_action: Action 完成，可能触发重新调度
      6. 返回推荐的配额大小
    """
    body = await request.json()
    worker_info = body['worker_info']
    was_being_scheduled = body['was_being_scheduled']

    worker = pool_manager.get_worker(worker_info)
    node = pool_manager.get_node_from_worker(worker)
    logger.debug(
        f'{worker.node_type}-{worker.node_id:02d}-{worker.worker_id:02d} '
        f'is asking for quota'
    )

    assert not worker.being_hanged
    worker.being_hanged = True
    assert worker.being_scheduled == was_being_scheduled
    worker.being_scheduled = False

    # 清空残留事件
    while not worker.hanging_events.empty():
        worker.hanging_events.get_nowait()

    # sllm+share 计费逻辑
    if scheduler_config.system == 'serverlessllm' and scheduler_config.sllm_enable_sharing:
        if not was_being_scheduled:
            worker.sllm_start_billing()
        else:
            worker.sllm_handle_one_iteration_complete()

    worker.check_whether_performing_dispatched_actions()

    # 触发调度
    if was_being_scheduled or (
        worker.can_be_scheduled() and node.acquire_schedule_permission()
    ):
        asyncio.create_task(node.schedule())

    # 等待调度事件
    while True:
        event_type = await worker.hanging_events.get()
        if event_type == WorkerHangingReleaseType.being_scheduled:
            logger.debug(
                f'{worker.node_type}-{worker.node_id:02d}-{worker.worker_id:02d} '
                f'hanging is released due to being_scheduled'
            )
            break
        elif event_type == WorkerHangingReleaseType.finished_action:
            logger.debug(
                f'{worker.node_type}-{worker.node_id:02d}-{worker.worker_id:02d} '
                f'hanging is released due to action finish'
            )
            if worker.can_be_scheduled() and node.acquire_schedule_permission():
                asyncio.create_task(node.schedule())
        else:
            raise Exception("Unknown hanging release type")

    worker.being_hanged = False

    if scheduler_config.system == 'serverlessllm' and scheduler_config.sllm_enable_sharing:
        worker.sllm_handle_one_iteration_start()

    return JSONResponse({'quota': worker.get_recommend_quota_size()})


# ====================================================================
# POST /worker_go_idle — Worker 空闲通知
# ====================================================================

@app.post("/worker_go_idle")
async def worker_go_idle(request: Request):
    """Worker 通知调度器自己进入空闲状态。

    触发重新调度（将调度权限移交给其他 Worker）。
    """
    body = await request.json()
    worker_info = body['worker_info']
    worker = pool_manager.get_worker(worker_info)
    node = pool_manager.get_node_from_worker(worker)
    logger.debug(
        f'{worker.node_type}-{worker.node_id:02d}-{worker.worker_id:02d} '
        f'is going idle'
    )

    assert not worker.being_hanged
    assert worker.being_scheduled
    worker.being_scheduled = False
    worker.check_whether_performing_dispatched_actions()

    asyncio.create_task(node.schedule())
    return JSONResponse({'result': True})


# ====================================================================
# POST /v1/completions — 推理请求 (类 OpenAI API)
# ====================================================================

@app.post("/v1/completions")
async def create_completion(request: Request):
    """接收推理请求并调度执行。

    Returns:
        {'result': True, 'e2e_metrics': {...}} 成功
        {'result': False} DDL 违反，请求被丢弃
    """
    body = await request.json()
    request_info = body['request_info']
    request_info['start_time'] = time.time()

    request_tracker = ReqTracker(request_info)
    logger.info(f'receive request: {request_tracker.get_info()}')

    while not request_tracker.check_finish():
        # 尝试调度
        while True:
            success = pool_manager.schedule_incoming_request(request_tracker)
            if success:
                break
            # 调度失败，等待后重试
            await asyncio.sleep(0.25)
            if request_tracker.ddl_violate():
                logger.info(f'failed request: {request_tracker.get_info()}')
                pool_manager.delete_request(request_tracker)
                return JSONResponse({'result': False})

        # 等待请求完成或被驱逐
        logger.info(f'schedule request: {request_tracker.get_info()}')
        await request_tracker.terminate_event.wait()
        request_tracker.terminate_event.clear()

        # 请求完成
        e2e_metrics = request_tracker.get_e2e_metrics()
    pool_manager.delete_request(request_tracker)
    logger.info(f'complete request: {request_tracker.get_info()}')
    return JSONResponse({'result': True, 'e2e_metrics': e2e_metrics})


# ====================================================================
# POST /set_config — 动态配置
# ====================================================================

# 有效的配置键白名单
_VALID_CONFIG_KEYS = {
    'system', 'keep_alive_time', 'pool_priority', 'enable_defragmentation',
    'enable_preempt', 'enable_sharing', 'sllm_enable_sharing',
    'enable_detailed_logging', 'minimal_tokens_per_instance',
    'kv_scale_watermark', 'enable_cpu',
}

_CONFIG_ASSERTIONS = {
    'system': lambda v: v in ['serverlessllm', 'sota'],
    'keep_alive_time': lambda v: v >= 0,
    'pool_priority': lambda v: v in ['cpu', 'gpu'],
    'enable_defragmentation': lambda v: v in [True, False],
    'enable_preempt': lambda v: v in [True, False],
    'enable_sharing': lambda v: v in [True, False],
    'sllm_enable_sharing': lambda v: v in [True, False],
    'enable_detailed_logging': lambda v: v in [True, False],
    'minimal_tokens_per_instance': lambda v: v >= 0,
    'kv_scale_watermark': lambda v: 0 <= v <= 1,
    'enable_cpu': lambda v: v in [True, False],
}


@app.post('/set_config')
async def set_config(request: Request):
    """动态修改调度器运行时配置。

    先重置部分配置到默认值，再根据请求体逐个应用。

    修改后自动更新所有 CPU 节点的分布式调度器配置。
    """
    # 重置默认值
    scheduler_config.sllm_enable_sharing = False
    scheduler_config.enable_cpu = True

    body = await request.json()
    for config_key in body.keys():
        if config_key not in _VALID_CONFIG_KEYS:
            raise Exception(f'unknown config key: {config_key}')

        new_value = body[config_key]
        assert_fn = _CONFIG_ASSERTIONS.get(config_key)
        if assert_fn:
            assert assert_fn(new_value), f"Invalid value for {config_key}: {new_value}"

        if config_key == 'system':
            scheduler_config.system = new_value
        elif config_key == 'keep_alive_time':
            scheduler_config.keep_alive_time = new_value
        elif config_key == 'pool_priority':
            scheduler_config.pool_priority = new_value
        elif config_key == 'enable_defragmentation':
            scheduler_config.enable_defragmentation = new_value
        elif config_key == 'enable_preempt':
            scheduler_config.enable_preempt = new_value
        elif config_key == 'enable_sharing':
            scheduler_config.enable_sharing = new_value
        elif config_key == 'sllm_enable_sharing':
            scheduler_config.sllm_enable_sharing = new_value
        elif config_key == 'enable_detailed_logging':
            scheduler_config.enable_detailed_logging = new_value
        elif config_key == 'minimal_tokens_per_instance':
            scheduler_config.minimal_tokens_per_instance = new_value
        elif config_key == 'kv_scale_watermark':
            scheduler_config.kv_scale_watermark = new_value
        elif config_key == 'enable_cpu':
            scheduler_config.enable_cpu = new_value

    # 同步到分布式调度器
    for cpu_node in pool_manager.cpu_pool.nodes.values():
        cpu_node.update_dist_scheduler()

    return JSONResponse({'result': True})


# ====================================================================
# POST /get_config — 获取配置
# ====================================================================

@app.post('/get_config')
async def get_config(request: Request):
    """返回当前完整配置（供测试工具和调试使用）。"""
    return JSONResponse({
        'pools_config': scheduler_config.pools_config,
        'keep_alive_time': scheduler_config.keep_alive_time,
        'system': scheduler_config.system,
        'decode_preempt_metric': scheduler_config.decode_preempt_metric,
        'memory_preempt_metric': scheduler_config.memory_preempt_metric,
        'pool_priority': scheduler_config.pool_priority,
        'enable_defragmentation': scheduler_config.enable_defragmentation,
        'enable_preempt': scheduler_config.enable_preempt,
        'enable_sharing': scheduler_config.enable_sharing,
        'enable_detailed_logging': scheduler_config.enable_detailed_logging,
        'minimal_tokens_per_instance': scheduler_config.minimal_tokens_per_instance,
        'kv_scale_watermark': scheduler_config.kv_scale_watermark,
        'ddl_based_schedule': scheduler_config.ddl_based_schedule,
    })


# ====================================================================
# POST /start_monitor /end_monitor — 监控
# ====================================================================

@app.post('/start_monitor')
async def start_monitor(request: Request):
    """开始周期性监控日志记录。"""
    pool_manager.start_monitor_async()
    return Response(status_code=200)


@app.post('/end_monitor')
async def end_monitor(request: Request):
    """结束监控，返回收集的日志。"""
    pool_manager.end_monitor()
    return JSONResponse(pool_manager.logs)


# ====================================================================
# 入口
# ====================================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7000, log_level='warning')
