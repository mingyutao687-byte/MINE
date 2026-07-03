"""MINE vLLM API 端点 — 添加到 vLLM OpenAI API Server 的自定义路由。

这些端点使 MINE 调度器能够控制 vLLM Worker 的:
  - 生命周期: register_worker, clear_worker
  - 模型管理: load_model, offload_model
  - KV Cache 管理: kv_scale, kv_info, kv_send, kv_receive
  - 请求驱逐: evict_requests
  - 流量控制: set_traffic_light

集成方式:
  在 vllm/entrypoints/openai/api_server.py 的 build_app() 中:
    1. 导入: from Mine.engine.api.mine_endpoints import create_mine_router
    2. 注册: app.include_router(create_mine_router(engine, kv_manager))

"""

import asyncio
import gc
import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


def create_mine_router(engine, kv_manager) -> APIRouter:
    """创建包含所有 MINE 自定义端点的 FastAPI Router。

    Args:
        engine: AsyncLLMEngine 实例（需包含 MINE 扩展方法）
        kv_manager: KVManager 全局单例

    Returns:
        配置好的 APIRouter，可被 vLLM 主应用挂载
    """
    router = APIRouter(prefix="/mine", tags=["MINE"])

    # ================================================================
    # Worker 生命周期
    # ================================================================

    @router.post("/register_worker")
    async def register_worker(raw_request: Request):
        """注册 Worker 信息到引擎。

        Worker 启动后调用，告知引擎其调度器地址和分布式配置。

        Body:
            worker_info: {pool_type, node_id, worker_id, gateway_ip,
                          using_dist_scheduler, scheduler_port, ...}
            dist_info: {master_addr, master_port, rank, world_size,
                        socket_addr, socket_port}
        """
        json_data = await raw_request.json()
        worker_info = json_data['worker_info']
        # dist_info = json_data['dist_info']  # 可选：用于 KV 迁移

        engine.register_worker(worker_info)
        # 以下两行可选：启用 KV 迁移功能
        # kv_manager.register_worker(worker_info)
        # kv_manager.init_gloo(dist_info)

        return JSONResponse({'result': True})

    @router.post("/clear_worker")
    async def clear_worker(raw_request: Request):
        """清除 Worker 状态。

        在 Worker 重启或关闭前调用，清理引擎内部状态。
        """
        engine.clear_worker()
        gc.collect()
        return JSONResponse({'result': True})

    # ================================================================
    # 模型管理
    # ================================================================

    @router.post("/load_model")
    async def load_model(raw_request: Request):
        """加载模型到 GPU 显存。

        GPU Worker 初始不持有模型（NO_MODEL_LOADING_AT_START=True），
        由调度器在需要时通过此端点触发加载。
        """
        engine.load_model()
        return JSONResponse({'result': True})

    @router.post("/offload_model")
    async def offload_model(raw_request: Request):
        """卸载模型，释放 GPU 显存。

        可选地在卸载后重新分配 KV cache。

        Body:
            new_num_blocks: -1 表示不重新分配，≥0 表示卸载后分配指定 blocks
        """
        json_data = await raw_request.json()
        new_num_blocks = json_data['new_num_blocks']

        engine.offload_model()
        if new_num_blocks >= 0:
            engine.scale_kv_cache_chxu(new_num_blocks)
        else:
            assert new_num_blocks == -1

        return JSONResponse({'result': True})

    # ================================================================
    # KV Cache 管理
    # ================================================================

    @router.post("/kv_info")
    async def kv_info(raw_request: Request):
        """获取当前 KV cache 状态信息。"""
        infos = engine.get_kv_info_chxu()
        return JSONResponse(infos)

    @router.post("/kv_scale")
    async def kv_scale(raw_request: Request):
        """动态调整 KV cache 的 block 数量。

        扩容: 分配更多 GPU 显存给 KV cache
        缩容: 释放不再需要的 KV cache 空间

        Body:
            new_num_blocks: 目标 block 数量

        Returns:
            {result, old_num_blocks, new_num_blocks}
        """
        json_data = await raw_request.json()
        new_num_blocks = json_data['new_num_blocks']
        old_num_blocks, new_num_blocks = engine.scale_kv_cache_chxu(new_num_blocks)
        return JSONResponse({
            'result': True,
            'old_num_blocks': old_num_blocks,
            'new_num_blocks': new_num_blocks,
        })

    @router.post("/kv_send")
    async def kv_send(raw_request: Request):
        """将指定请求的 KV cache 发送到目标 Worker。

        用于请求迁移（抢占、PD分离）。

        Body:
            request_id_list: 要迁移的请求 ID 列表
            transfer_config: {src_rank, dst_rank, dst_socket_addr, dst_socket_port}
        """
        logger.info('kv_send start')
        json_data = await raw_request.json()
        kv_manager.send_kv(
            request_id_list=json_data['request_id_list'],
            transfer_config=json_data['transfer_config'],
        )
        logger.info('kv_send end')
        return JSONResponse({'result': True})

    @router.post("/kv_receive")
    async def kv_receive(raw_request: Request):
        """接收来自其他 Worker 的 KV cache。

        异步处理，立即返回。
        """
        json_data = await raw_request.json()
        asyncio.create_task(kv_manager.receive_kv(json_data))
        return JSONResponse({'result': True})

    # ================================================================
    # 请求驱逐
    # ================================================================

    @router.post("/evict_requests")
    async def evict_requests(raw_request: Request):
        """驱逐指定的请求（可选地保存 KV cache）。

        用于 Worker 抢占: 源 Worker 保存 KV cache 后驱逐请求，
        目标 Worker 恢复 KV cache 后继续推理。

        Body:
            request_id_list: 要驱逐的请求 ID 列表
            save_kv: 是否保存 KV cache（用于后续恢复）
        """
        json_data = await raw_request.json()
        request_id_list = json_data['request_id_list']
        save_kv = json_data['save_kv']

        await engine.perform_kv_offload_chxu(request_id_list, save_kv)
        return JSONResponse({'result': True})

    # ================================================================
    # 流量控制
    # ================================================================

    @router.post("/set_traffic_light")
    async def set_traffic_light(raw_request: Request):
        """设置引擎的流量灯状态。

        - green: 正常处理请求
        - yellow: 等待中（迁移进行中）
        - red: 暂停处理（正在保存/迁移 KV cache）

        Body:
            color: 'red' | 'yellow' | 'green'
        """
        json_data = await raw_request.json()
        color = json_data['color']
        engine.set_traffic_light(color)
        return JSONResponse({'result': True})

    @router.get("/ping")
    async def ping():
        """健康检查端点。"""
        return JSONResponse({'status': 'ok'})

    return router
