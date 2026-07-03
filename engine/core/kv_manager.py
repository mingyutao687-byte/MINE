"""KV Cache 管理器 — 全局单例，管理跨 Worker 的 KV cache 传输。

这是 SLINFER 对 vLLM 的核心扩展之一。KVManager 维护一个请求级 KV cache 字典，
支持:
  - save: Worker 保存请求的 KV cache 到管理器
  - send: 将 KV cache 通过 Gloo 后端发送到另一个 Worker
  - receive: 从其他 Worker 接收 KV cache 并存储

用于实现 CPU→GPU 和 GPU→GPU 的请求迁移（抢占/PD分离/负载均衡）。

传输架构:
  Python 主进程 (KVManager)
    ├── receive_thread: 从 receive_queue 读取收到的 KV tensor
    ├── logger_thread: 从 logger_queue 读取子进程日志
    └── transfer_process (mp.Process)
        └── kv_transfer_manager (Gloo init + sender/receiver 线程)
            ├── kv_sender 线程: TCP 交换元数据 → Gloo dist.send
            └── kv_receiver 线程: TCP 接收元数据 → Gloo dist.recv

原文件: vLLM_modify/vllm/core/kv_manager.py
"""

import asyncio
import socket
import threading
from typing import Optional

import aiohttp
import torch
import torch.multiprocessing as mp

from .kv_transfer import kv_transfer_manager

import logging
logger = logging.getLogger(__name__)


# ============================================================================
# KVInfo — 单个请求的 KV cache 数据封装
# ============================================================================

class KVInfo:
    """封装一个请求的 KV cache 数据。

    Attributes:
        data_config: 元数据字典，包含:
            - request_id: 请求 ID
            - length: 序列总长度
            - tensor_shape: KV tensor 的形状 [layers, 2(KV), blocks, ...]
            - device: 设备类型 ('cpu' | 'gpu')
            - src_rank, dst_rank: Gloo 通信 rank
            - dst_socket_addr, dst_socket_port: TCP 元数据通道
        tensor: 实际的 KV cache tensor (shared memory)
    """

    def __init__(self, data_config: dict, tensor: torch.Tensor):
        self.data_config = data_config
        self.tensor = tensor


# ============================================================================
# KVManager — 全局 KV Cache 管理器
# ============================================================================

class KVManager:
    """全局 KV Cache 管理器（单例模式）。

    负责:
      1. 存储已保存的 KV cache (requests dict)
      2. 管理 Gloo 传输子进程的生命周期
      3. 协调跨 Worker 的 KV cache 发送和接收
      4. 通知 Gateway 迁移完成

    使用方式:
      from Mine.engine.core.kv_manager import kv_manager  # 全局单例
    """

    def __init__(self):
        # 请求级 KV cache 存储: request_id → KVInfo
        self.requests: dict[str, KVInfo] = {}

        # 网络 socket（未使用，保留用于未来扩展）
        self.send_sock: Optional[socket.socket] = None
        self.receive_sock: Optional[socket.socket] = None

        # Gloo 传输子进程
        self.transfer_process: Optional[mp.Process] = None
        self.receive_thread: Optional[threading.Thread] = None
        self.logger_thread: Optional[threading.Thread] = None

        # 进程间通信队列
        self.send_queue: Optional[mp.Queue] = None
        self.receive_queue: Optional[mp.Queue] = None
        self.logger_queue: Optional[mp.Queue] = None

        # Worker 元信息（用于 HTTP 回调）
        self.worker_info: Optional[dict] = None

    # ------------------------------------------------------------------
    # Worker 注册
    # ------------------------------------------------------------------

    def register_worker(self, worker_info: dict):
        """注册 Worker 信息，用于迁移完成后的 HTTP 回调。

        Args:
            worker_info: 包含 gateway_ip 等字段的字典
        """
        self.worker_info = worker_info

    # ------------------------------------------------------------------
    # Gloo 初始化
    # ------------------------------------------------------------------

    def init_gloo(self, gloo_config: dict):
        """初始化 Gloo 分布式传输后端。

        启动一个独立的子进程运行 kv_transfer_manager，
        该子进程内创建 sender 和 receiver 两个线程。

        Args:
            gloo_config: 包含 rank, world_size, master_addr, master_port 等字段
        """
        if self.transfer_process is not None:
            self.transfer_process.kill()

        ctx = mp.get_context('spawn')
        self.send_queue = ctx.Queue()
        self.receive_queue = ctx.Queue()
        self.logger_queue = ctx.Queue()

        # 启动接收线程（主进程侧）
        self.receive_thread = threading.Thread(target=self._receive_kv_loop)
        self.receive_thread.start()

        # 启动日志线程
        self.logger_thread = threading.Thread(target=self._print_logger_loop)
        self.logger_thread.start()

        # 启动传输子进程
        self.transfer_process = ctx.Process(
            target=kv_transfer_manager,
            args=(gloo_config, self.send_queue, self.receive_queue, self.logger_queue),
            daemon=True,
        )
        self.transfer_process.start()

    # ------------------------------------------------------------------
    # HTTP 回调
    # ------------------------------------------------------------------

    async def _post_migration_complete(self, request_id_list: list[str]):
        """通知 Gateway KV 迁移已完成。

        HTTP POST → http://{gateway_ip}:7000/migration_complete
        """
        url = f'http://{self.worker_info["gateway_ip"]}:7000/migration_complete'
        headers = {"Content-Type": "application/json"}
        data = {'request_id_list': request_id_list}
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as response:
                res = await response.json()
        assert res['result'] is True

    # ------------------------------------------------------------------
    # 接收循环（主进程线程）
    # ------------------------------------------------------------------

    def _receive_kv_loop(self):
        """持续从 receive_queue 读取接收到的 KV cache。

        每收到一个完整的 (data_config, tensor) 对:
          1. 存入 self.requests
          2. 异步通知 Gateway 迁移完成
        """
        while True:
            data_config = self.receive_queue.get()
            logger.warning('main process receive data_config')
            tensor = self.receive_queue.get()
            logger.warning('main process receive tensor')

            request_id = data_config['request_id']
            self.requests[request_id] = KVInfo(data_config, tensor)
            logger.warning(f'kv_tensor receive success, info: {data_config}')

            asyncio.run(self._post_migration_complete(request_id_list=[request_id]))

    def _print_logger_loop(self):
        """持续从 logger_queue 读取子进程日志并输出。"""
        while True:
            output = self.logger_queue.get()
            logger.warning('From kv_manager_subprocess: ' + output)

    # ------------------------------------------------------------------
    # KV Cache CRUD
    # ------------------------------------------------------------------

    def have_request_kv_cache(self, request_id: str) -> bool:
        """检查是否持有指定请求的 KV cache。"""
        return request_id in self.requests

    def add_request(self, data_config: dict, tensor: torch.Tensor):
        """手动添加一个请求的 KV cache。

        Args:
            data_config: 元数据字典
            tensor: KV cache tensor (必须是 shared memory)
        """
        request_id = data_config['request_id']
        assert request_id not in self.requests
        self.requests[request_id] = KVInfo(data_config, tensor)

    def pop_request(self, request_id: str) -> KVInfo:
        """取出并删除指定请求的 KV cache。

        Raises:
            AssertionError: 如果请求不在字典中
        """
        assert request_id in self.requests
        return self.requests.pop(request_id)

    # ------------------------------------------------------------------
    # KV Cache 发送
    # ------------------------------------------------------------------

    def send_kv(self, request_id_list: list[str], transfer_config: dict):
        """将指定请求的 KV cache 发送到目标 Worker。

        对于每个请求:
          1. 从 self.requests 中 pop 出 KVInfo
          2. 将 (data_config + transfer_config, tensor) 放入 send_queue
          3. 子进程中的 kv_sender 线程负责实际的 Gloo 传输

        Args:
            request_id_list: 要发送的请求 ID 列表
            transfer_config: 包含 src_rank, dst_rank, dst_socket_addr, dst_socket_port
        """
        for request_id in request_id_list:
            assert request_id in self.requests
            kv_info = self.requests.pop(request_id)
            data_config = kv_info.data_config
            tensor = kv_info.tensor
            assert tensor.is_shared()

            # 合并 transfer_config 到 data_config
            data_config.update(transfer_config)

            # 发送到子进程
            self.send_queue.put(data_config)
            self.send_queue.put(tensor)


# ============================================================================
# 全局单例
# ============================================================================

kv_manager = KVManager()
