"""Mine — SLINFER GPU+CPU 异构 LLM 推理调度系统（完整重构版）。

针对 4× NVIDIA A10 (24GB) + 256GB 主存 + Intel CPU (无 AMX) 环境优化。

基于原始 SLINFER 三组件重构:
  - SLINFER_core/scheduler/  →  Mine/scheduler/  (调度器)
  - vLLM_modify/              →  Mine/engine/     (推理引擎扩展)
  - ServerlessLLM_modify/     →  Mine/store/      (模型存储集成)

移除的组件:
  - OpenVINO (需要 Intel AMX)
  - TPU, XPU, Neuron (需要专用硬件)
  - AMD ROCm 特定代码 (仅保留 NVIDIA CUDA + 通用 CPU)

包结构:
  scheduler/         — 调度器 (Gateway, Pool, Node, Worker, ReqTracker)
  engine/            — vLLM 推理引擎扩展 (KVManager, KV Transfer, SLINFER API)
  store/             — ServerlessLLM 模型存储集成
  config/            — A10 GPU 资源池配置
  config_template/   — Pool/Model 配置模板
  tools/             — 测试与工具
"""

__version__ = "2.0.0"

# ============================================================================
# Scheduler 导出
# ============================================================================
from Mine.config.settings import scheduler_config
from Mine.models.request_tracker import ReqTracker
from Mine.models.enums import WorkerHangingReleaseType
from Mine.models.actions import (
    WorkerActionBase,
    WorkerKVScaleAction,
    WorkerEvictRequestsAction,
    WorkerLoadAction,
    WorkerOffloadAction,
    WorkerGiveOutMemory,
    WorkerSleepAction,
)
from Mine.core.worker import Worker
from Mine.core.node import Node, MiniNode, MiniWorker
from Mine.core.pool import Pool, PoolManager, ModelManager
from Mine.core.power_estimator import estimate_duration, get_serverlessllm_concurrency
from Mine.api.gateway import app as gateway_app
from Mine.api.dist_gateway import app as dist_gateway_app

# ============================================================================
# Engine 导出
# ============================================================================
from Mine.engine.core.kv_manager import KVManager, KVInfo, kv_manager

# ============================================================================
# Config 导出
# ============================================================================
from Mine.config.a10_pools import (
    A10_MODELS,
    POOLS_3B_4GPU_0CPU,
    POOLS_7B_4GPU_0CPU,
    POOLS_3B_1GPU_1CPU_DEBUG,
)

__all__ = [
    # Scheduler
    "scheduler_config",
    "ReqTracker",
    "WorkerHangingReleaseType",
    "WorkerActionBase",
    "WorkerKVScaleAction",
    "WorkerEvictRequestsAction",
    "WorkerLoadAction",
    "WorkerOffloadAction",
    "WorkerGiveOutMemory",
    "WorkerSleepAction",
    "Worker",
    "Node",
    "MiniNode",
    "MiniWorker",
    "Pool",
    "PoolManager",
    "ModelManager",
    "estimate_duration",
    "get_serverlessllm_concurrency",
    "gateway_app",
    "dist_gateway_app",
    # Engine
    "KVManager",
    "KVInfo",
    "kv_manager",
    # Config
    "A10_MODELS",
    "POOLS_3B_4GPU_0CPU",
    "POOLS_7B_4GPU_0CPU",
    "POOLS_3B_1GPU_1CPU_DEBUG",
]
