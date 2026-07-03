"""HTTP API 模块 — 调度网关的 FastAPI 应用。

- gateway: 主调度网关 (端口 7000)，协调所有 GPU/CPU Worker
- dist_gateway: 分布式调度网关 (每 CPU 节点本地)，DDL-based 本地调度
"""

from Mine.api.gateway import app as gateway_app
from Mine.api.dist_gateway import app as dist_gateway_app

__all__ = [
    "gateway_app",
    "dist_gateway_app",
]
