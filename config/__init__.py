"""MINE 配置模块。

提供 A10 GPU 优化的资源池配置和模型元信息。
"""
from Mine.config.a10_pools import (
    A10_MODELS,
    POOLS_3B_4GPU_0CPU,
    POOLS_7B_4GPU_0CPU,
    POOLS_3B_1GPU_1CPU_DEBUG,
)

__all__ = [
    "A10_MODELS",
    "POOLS_3B_4GPU_0CPU",
    "POOLS_7B_4GPU_0CPU",
    "POOLS_3B_1GPU_1CPU_DEBUG",
]
