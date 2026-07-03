"""SLINFER Engine — vLLM 推理引擎的 SLINFER 扩展。

提供:
  - KVManager: 全局 KV cache 管理器
  - kv_transfer: Gloo+TCP 底层 KV 传输
  - slinfer_endpoints: 添加到 vLLM API Server 的自定义路由
"""

from Mine.engine.core.kv_manager import KVManager, KVInfo, kv_manager
from Mine.engine.core.kv_transfer import (
    kv_sender,
    kv_receiver,
    kv_transfer_manager,
)

__all__ = [
    "KVManager",
    "KVInfo",
    "kv_manager",
    "kv_sender",
    "kv_receiver",
    "kv_transfer_manager",
]
