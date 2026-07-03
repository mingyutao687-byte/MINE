"""数据模型定义 — 调度系统中使用的结构化数据类。

使用 dataclass 替代原始 dict，提供类型安全和 IDE 自动补全。
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WorkerConfig:
    """单个 Worker 的静态配置信息。

    从 pools_info_template 中的 worker_info dict 解析得到。
    """
    model_type: str                     # 模型类型标识，如 'llama-2-7b'
    model_memory_gb: float              # 模型在内存中占用大小 (GB)
    per_token_kv_memory_kb: float       # 每个 token 的 KV cache 内存 (KB)
    block_size_cpu: int                 # CPU 端 KV block 大小 (token 数)
    block_size_gpu: int                 # GPU 端 KV block 大小 (token 数)
    cpu_kv_gb: float                    # CPU 端 KV cache 总容量 (GB)


@dataclass
class NodeConfig:
    """单个 Node（物理机器）的静态配置信息。

    从 pools_info_template 中的 node_info dict 解析得到。
    """
    node_label: str                     # 节点标签: 'cpu' | 'gpu' | 'aliyun-16c' 等
    node_memory_capacity_gb: float      # 节点总可用内存 (GB)
    node_ip: str                        # 节点 IP 地址
    gateway_ip: str                     # 网关 IP 地址
    dist_scheduler: bool                # 是否使用分布式调度器
    base_port: int                      # Worker HTTP 端口起始值
    workers: dict[int, WorkerConfig]    # worker_id → WorkerConfig 映射


@dataclass
class DistConfig:
    """分布式训练/通信配置信息。

    vLLM 使用 Gloo 后端做分布式 KV cache 传输。
    """
    master_addr: str                    # 主节点地址
    master_port: int                    # 主节点端口 (30000)
    rank: int                           # 当前进程的全局 rank
    world_size: int                     # 全局进程数
    socket_addr: str                    # KV 传输 socket 地址
    socket_port: int                    # KV 传输 socket 端口 (worker_port + 10000)


@dataclass
class WorkerRuntimeInfo:
    """Worker 运行时信息 — 发送给 vLLM Worker 的注册信息。"""
    pool_type: str                      # 'cpu' | 'gpu'
    node_id: int                        # 所属 Node ID
    worker_id: int                      # Worker ID
    gateway_ip: str                     # 网关 IP
    using_dist_scheduler: bool          # 是否使用分布式调度
    scheduler_port: int                 # 调度器端口
    worker_ip: str                      # Worker IP
    worker_port: int                    # Worker HTTP 端口
