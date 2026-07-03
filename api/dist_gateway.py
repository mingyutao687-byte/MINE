"""MINE 分布式调度网关 — 每 CPU 节点本地运行的调度器。

与主 Gateway 的区别:
  - 仅管理单个 CPU 节点的 Worker
  - 不做内存检查（CPU 内存预分配）
  - 不做抢占（CPU 不可抢占）
  - 使用轻量 MiniNode/MiniWorker 替代完整的 Node/Worker
  - 通过 WebSocket 接收来自 Node 的 DDL 和 batch 实时更新

提供 API:
  - POST /ask_for_quota: Worker 配额请求
  - POST /worker_go_idle: Worker 空闲通知
  - POST /init: 初始化分布式调度器
  - POST /update_system_config: 更新配置
  - WebSocket /ws/update_ddl: 实时 DDL 更新通道

"""

import asyncio
import time
import json
import argparse
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, Request
from fastapi.websockets import WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
import uvicorn

from Mine.models.enums import WorkerHangingReleaseType
from Mine.core.node import MiniNode

# CLI 参数 (仅在直接运行时解析)
_port = 7001  # 默认端口，被 __main__ 覆盖

# 日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s:%(lineno)d %(message)s',
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

app = FastAPI()

# 全局 MiniNode 实例
node = MiniNode()


# ====================================================================
# POST /ask_for_quota — Worker 配额请求
# ====================================================================

@app.post("/ask_for_quota")
async def ask_for_quota(request: Request):
    """Worker 向分布式调度器请求执行配额。

    与主 Gateway 的区别:
      - 使用 MiniNode/MiniWorker 而非完整 Node/Worker
      - 不支持 Action 系统（CPU Worker 无 Action）
      - 不支持 sllm+share 计费
      - stupid_schedule 模式直接返回 quota=1（绕过调度）
    """
    body = await request.json()
    worker_info = body['worker_info']
    was_being_scheduled = body['was_being_scheduled']
    worker_id = worker_info['worker_id']
    worker = node.workers[worker_id]

    logger.debug(f'{worker.worker_id:02d} is asking for quota')

    # sllm+share 模式: 不做调度，直接给配额
    if node.enable_stupid_schedule:
        logger.debug(
            f'{worker.worker_id:02d} '
            f'hanging is released due to being_scheduled'
        )
        return JSONResponse({'quota': 1})

    assert not worker.being_hanged
    worker.being_hanged = True
    assert worker.being_scheduled == was_being_scheduled
    worker.being_scheduled = False

    # 触发调度
    if was_being_scheduled or node.acquire_schedule_permission():
        asyncio.create_task(node.schedule())

    # 等待调度事件
    while True:
        event_type = await worker.hanging_events.get()
        if event_type == WorkerHangingReleaseType.being_scheduled:
            logger.debug(
                f'{worker.worker_id:02d} '
                f'hanging is released due to being_scheduled'
            )
            break
        else:
            raise Exception("Unexpected hanging release type in dist_gateway")

    worker.being_hanged = False
    return JSONResponse({'quota': 1})


# ====================================================================
# POST /worker_go_idle — Worker 空闲通知
# ====================================================================

@app.post("/worker_go_idle")
async def worker_go_idle(request: Request):
    """Worker 通知分布式调度器自己进入空闲状态。"""
    body = await request.json()
    worker_info = body['worker_info']
    worker_id = worker_info['worker_id']
    worker = node.workers[worker_id]
    logger.debug(f'{worker.worker_id:02d} is going idle')

    if node.enable_stupid_schedule:
        return JSONResponse({'result': True})

    assert not worker.being_hanged
    assert worker.being_scheduled
    worker.being_scheduled = False

    asyncio.create_task(node.schedule())
    return JSONResponse({'result': True})


# ====================================================================
# WebSocket /ws/update_ddl — DDL 实时更新
# ====================================================================

@app.websocket('/ws/update_ddl')
async def update_ddl(websocket: WebSocket):
    """接收来自 Node.periodic_update_ddl_to_dist_scheduler() 的 DDL 更新。

    更新频率: 每 100ms
    数据格式:
      {
        'info_version': int,        # 单调递增的版本号
        'workers_ddl': {id: ddl},   # Worker ID → 相对 DDL (秒)
        'workers_batch_num': {id: n} # Worker ID → running_requests 数量
      }
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            info_version = data['info_version']
            if info_version > node.info_version:
                node.info_version = info_version
                workers_ddl: dict = data['workers_ddl']
                workers_batch_num: dict = data['workers_batch_num']
                logger.info(
                    f'receive new ddl info: {workers_ddl}, '
                    f'batch_num info: {workers_batch_num}'
                )
                for worker_id, ddl in workers_ddl.items():
                    node.workers[int(worker_id)].ddl = ddl
                for worker_id, batch_num in workers_batch_num.items():
                    node.workers[int(worker_id)].batch_num = batch_num
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error in WebSocket communication: {e}")


# ====================================================================
# POST /init — 初始化
# ====================================================================

@app.post('/init')
async def init(request: Request):
    """初始化分布式调度器。

    由 Node.__init__() 在启动时调用。
    """
    body = await request.json()
    worker_num: int = body['worker_num']
    ddl_based_schedule_config: dict = body['ddl_based_schedule_config']
    is_sllm_share: bool = body['is_sllm_share']

    logger.info(f'initialize with {worker_num} workers')
    logger.info(f'update ddl_based_schedule_config with {ddl_based_schedule_config}')
    logger.info(f'is_sllm_share? {is_sllm_share}')

    if is_sllm_share:
        node.enable_stupid_schedule = True
    else:
        node.enable_stupid_schedule = False

    node.initialize_workers(worker_num)
    for k, v in ddl_based_schedule_config.items():
        assert k in node.ddl_based_schedule_config
        node.ddl_based_schedule_config[k] = v


# ====================================================================
# POST /update_system_config — 更新系统配置
# ====================================================================

@app.post('/update_system_config')
async def update_system_config(request: Request):
    """更新分布式调度器的系统配置。

    由 Node.update_dist_scheduler() 在 /set_config 时调用。
    """
    body = await request.json()
    is_sllm_share: bool = body['is_sllm_share']
    logger.warning(f'update: is_sllm_share? {is_sllm_share}')
    if is_sllm_share:
        node.enable_stupid_schedule = True
    else:
        node.enable_stupid_schedule = False


# ====================================================================
# 生命周期
# ====================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up...")
    logger.info('Start-up complete')
    yield
    logger.info("Shutting down...")


app.router.lifespan_context = lifespan


# ====================================================================
# 入口
# ====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, required=True)
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level='warning')
