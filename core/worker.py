"""Worker 工作进程抽象 — 调度系统中对单个 vLLM 推理实例的代理。

Worker 是 MINE 调度系统的核心抽象单元。每个 Worker 对应一个运行在
CPU 或 GPU 节点上的 vLLM 推理进程（OpenAI-compatible API server）。

核心职责:
  1. 生命周期管理: start_worker → allocate → deallocate
  2. 请求调度: activate_request, fire_request_async, delete_request_tracker
  3. 资源管理: 模型加载/卸载 (register_load_action/register_offload_action)
  4. KV Cache 管理: 扩缩容 (register_kv_scale_action)
  5. Action 系统: 所有异步操作通过 Action 队列统一管理
  6. sllm+share: ServerlessLLM 多租户共享的计费和时间配额

Action 系统设计:
  所有改变 Worker 状态的操作（加载模型、卸载模型、KV 扩缩容、驱逐请求）
  都通过 Action 系统进行：
    1. register_*_action() → 创建 Action 实例，入队到 Node 的 action queue
    2. Node.try_commit_and_dispatch_action() → 内存预算检查
    3. Worker.dispatch_*_action() → 加入 dispatched_action_list
    4. Worker.digest_dispatched_actions() → 依次执行

  版本控制: 每个 Action 有唯一的 action_version，用于检测和丢弃过时的 Action。

"""

import asyncio
import subprocess
import time
import math
import json
import os
import logging
from abc import ABC
from typing import Optional

import aiohttp

from Mine.config import settings as config
from Mine.config.settings import scheduler_config
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
from Mine.models.request_tracker import ReqTracker
from Mine.core.power_estimator import estimate_duration

logger = logging.getLogger(__name__)

# ============================================================================
# Worker 类
# ============================================================================


class Worker:
    """单个 vLLM 推理实例的代理。

    维护 Worker 的完整状态，包括模型持有状态、KV cache 容量、
    运行中的请求列表、以及待执行的 Action 队列。

    CPU Worker 与 GPU Worker 的行为差异:
      - CPU Worker: 启动时即持有模型 (hold_model=True)，KV cache 预分配
      - GPU Worker: 启动后需动态加载模型，KV cache 按需扩缩容
    """

    def __init__(
        self,
        node_type: str,
        node_label: str,
        node_id: int,
        worker_id: int,
        worker_info: dict,
        node_ip: str,
        base_port: int,
        rank: int,
        using_dist_scheduler: bool,
        worker_actions_queue: asyncio.Queue,  # Queue[WorkerActionBase]
        session: aiohttp.ClientSession,
    ):
        """初始化 Worker。

        Args:
            node_type: 'cpu' | 'gpu'
            node_label: 节点标签 (用于性能估算)
            node_id: 所属 Node 的 ID
            worker_id: 在该 Node 内的唯一 ID
            worker_info: 从 pools_info_template 解析的 Worker 配置
            node_ip: Worker 所在机器的 IP
            base_port: 端口基址 (实际端口 = base_port + worker_id)
            rank: 分布式全局 rank
            using_dist_scheduler: 是否使用独立的分布式调度器
            worker_actions_queue: 共享的 Action 注册队列 (指向 Node 的 queue)
            session: aiohttp 会话 (复用 TCP 连接)
        """
        # --- 基本标识 ---
        self.node_type: str = node_type
        self.node_label: str = node_label
        self.node_id: int = node_id
        self.worker_id: int = worker_id
        self.rank: int = rank
        self.using_dist_scheduler: bool = using_dist_scheduler
        self.session: aiohttp.ClientSession = session

        # --- 模型配置 ---
        self.model_type: str = worker_info['model_type']
        self.model_memory_KB: float = worker_info['model_memory_GB'] * 1024 * 1024

        # --- KV Cache 配置 ---
        self.kv_block_size: int = worker_info['block_size'][self.node_type]
        self.per_kv_block_memory_KB: float = (
            worker_info['per_token_kv_memory_KB'] * self.kv_block_size
        )

        # CPU Worker 有预分配的 KV cache；GPU Worker 从 0 开始按需扩缩容
        if self.node_type == 'cpu':
            cpu_kv_gb = worker_info['cpu_kv_gb']
            self.num_blocks_remote = math.floor(
                cpu_kv_gb * 1024 * 1024 / self.per_kv_block_memory_KB
            )
            logger.warning(
                f'{self.node_type}-{self.node_id}-{self.worker_id} '
                f'num_block: {self.num_blocks_remote}'
            )
        elif self.node_type == 'gpu':
            self.num_blocks_remote = 0
        else:
            raise ValueError(f"Unknown node_type: {self.node_type}")

        # num_blocks_remote: 实际已生效的 block 数
        self.num_blocks_remote_version: int = 0
        # num_blocks_local: 调度器期望的最终 block 数 (可能尚未生效)
        self.num_blocks_local: int = self.num_blocks_remote
        self.num_blocks_local_version: int = 0

        # --- Action 系统 ---
        self.action_version: int = 0                      # 单调递增的版本号
        self.dispatched_action_list: list[WorkerActionBase] = []  # 已分发待执行的 Action
        self.is_performing_action: bool = False           # 是否正在执行 Action
        self.actions_register_queue: asyncio.Queue = worker_actions_queue

        # --- 分配状态 ---
        self.allocated: bool = False          # 是否已分配模型
        self.will_be_removed: bool = False    # 即将被移除（不再接受新请求）
        self.being_scheduled: bool = False    # 是否已被调度器选中
        self.being_hanged: bool = False       # 是否正在挂起等待调度
        self.idle_start_time: float = 0.0     # 进入空闲的时间戳

        # --- 网络配置 ---
        self.node_ip: str = node_ip
        self.port: int = base_port + worker_id

        # --- 请求管理 ---
        self.running_requests: dict[int, ReqTracker] = {}
        self.model_id: int = -1                     # 当前加载的模型 ID
        self.hanging_events: asyncio.Queue = asyncio.Queue()  # 挂起等待事件队列

        # --- 性能追踪 ---
        self.time_slice: float = 0.0                # 当前计算时间片 (decode 能力占比)
        self.TPOT: float = scheduler_config.TPOT    # TPOT SLO (秒)

        # --- 模型持有状态 ---
        # CPU Worker 初始就持有模型，GPU Worker 初始不持有
        self.hold_model_remote: bool = (self.node_type == 'cpu')
        self.hold_model_remote_version: int = 0
        self.hold_model_local: bool = (self.node_type == 'cpu')
        self.hold_model_local_version: int = 0
        self.is_loading_model: bool = False
        self.loading_event: asyncio.Event = asyncio.Event()
        self.is_offloading_model: bool = False
        self.offloading_event: asyncio.Event = asyncio.Event()

        # --- 模型路径 ---
        self.model_path: str = config.models_path[self.model_type][self.node_label]

        # --- 启动与统计 ---
        self.start_complete: asyncio.Event = asyncio.Event()
        self.served_request_cnt: int = 0

        # KV 扩缩容日志
        self.past_lifecycle_kv_scale_log_list: list = []
        self.cur_kv_scale_log_list: list = []
        self.allocate_time: float = 0.0

        # --- sllm+share 专用变量 ---
        self.bill_start_time: float = 0.0
        self.exec_total_duration: float = 0.0
        self.exec_start_time: float = 0.0

    # ==================================================================
    # sllm+share 计费
    # ==================================================================

    def sllm_start_billing(self):
        """sllm+share: 开始一个新的计费周期。"""
        self.bill_start_time = time.time()
        self.exec_total_duration = 0.0

    def sllm_handle_one_iteration_complete(self):
        """sllm+share: 完成一轮推理迭代。

        根据已消耗时间计算需要的休眠时间，确保不超出公平份额。
        """
        cur_time = time.time()
        self.exec_total_duration += cur_time - self.exec_start_time
        nxt_wakeup_time = (self.bill_start_time
                           + self.exec_total_duration * scheduler_config.sllm_max_shares)
        if nxt_wakeup_time > cur_time:
            self.register_sleep_action(nxt_wakeup_time)

    def sllm_handle_one_iteration_start(self):
        """sllm+share: 开始执行新一轮推理。"""
        self.exec_start_time = time.time()

    # ==================================================================
    # 状态查询
    # ==================================================================

    def under_prefill(self) -> bool:
        """检查是否有请求处于 prefill 阶段。"""
        for req in self.running_requests.values():
            if req.under_prefill:
                return True
        return False

    def get_recommend_quota_size(self) -> int:
        """返回推荐的执行配额大小。

        CPU Worker 或 prefill 阶段: 1（每次只处理一个请求的 prefill）
        GPU Decode Worker: 按模型 1~4（支持更大的 batch）

        配额决定 Worker 在一次配额周期内可执行的推理步数。
        """
        if self.under_prefill() or self.node_type == 'cpu':
            return 1

        # GPU decode 阶段的配额按模型特性调整
        quota_map = {
            'llama-3.2-3b': 4,
            'llama-2-7b': 2,
            'llama-3.1-8b': 2,
            'llama-2-13b': 1,
            'codestral-22b': 1,
        }
        for prefix, quota in quota_map.items():
            if self.model_type.startswith(prefix):
                return quota
        raise Exception(f"Unknown model type for quota: {self.model_type}")

    def can_be_scheduled(self) -> bool:
        """Worker 是否可以被调度器选中。

        条件:
          1. 正在挂起等待 (being_hanged)
          2. 没有正在执行的 Action
          3. 模型已加载且本地已确认 (hold_model_remote & hold_model_local)
          4. KV cache 可用 (num_blocks > 0)
        """
        return (self.being_hanged
                and (not self.is_performing_action)
                and self.hold_model_remote
                and self.hold_model_local
                and self.num_blocks_remote > 0
                and self.num_blocks_local > 0)

    def empty(self) -> bool:
        """检查 Worker 是否完全空闲（无模型、无请求、无 KV cache）。

        GPU Worker 还需要确认模型已卸载且 KV cache 已清零。
        """
        return (self.model_id == -1
                and (not self.allocated)
                and len(self.running_requests) == 0
                and (self.node_type == 'cpu'
                     or ((not self.hold_model_remote)
                         and (not self.hold_model_local)
                         and (not self.is_loading_model)
                         and (not self.is_offloading_model)
                         and self.num_blocks_local == 0
                         and self.num_blocks_remote == 0)))

    def can_allocate_with_model(self, target_model_type: str) -> bool:
        """检查是否可以分配指定的模型类型。"""
        return self.model_type == target_model_type and (not self.allocated)

    def exist_loading_event(self) -> bool:
        """检查是否存在未完成的模型加载事件。"""
        if (not self.hold_model_remote) or (not self.hold_model_local):
            return True
        for action in self.dispatched_action_list:
            if isinstance(action, WorkerLoadAction):
                return True
        return False

    # ==================================================================
    # 内存管理
    # ==================================================================

    def get_memory_footprint(self) -> float:
        """计算当前内存占用 (KB) = 模型内存 + KV cache 内存。"""
        memory_footprint_KB = self.per_kv_block_memory_KB * self.num_blocks_local
        if self.hold_model_local:
            memory_footprint_KB += self.model_memory_KB
        return memory_footprint_KB

    def get_monitor_memory_detail(self) -> tuple:
        """返回监控用内存详情。

        Returns:
            (model_memory_KB, used_kv_memory_KB, scheduled_kv_memory_KB)
        """
        model_memory = 0.0
        if self.hold_model_local:
            model_memory = self.model_memory_KB

        use_kv_memory = 0.0
        for req in self.running_requests.values():
            use_kv_memory += (
                math.ceil(req.total_length() / self.kv_block_size)
                * self.per_kv_block_memory_KB
            )

        schedule_kv_memory = self.per_kv_block_memory_KB * self.num_blocks_local
        return model_memory, use_kv_memory, schedule_kv_memory

    def get_kv_usage(self) -> int:
        """计算当前 KV cache 实际使用的 block 数。"""
        num = 0
        for req in self.running_requests.values():
            num += math.ceil((req.total_length() + 512) / self.kv_block_size)
        return num

    def get_kv_minimal_required_blocks(self) -> int:
        """计算满足当前请求所需的最小 KV block 数。

        不低于 minimal_tokens_per_instance 保证的最小容量。
        """
        minimal_required_blocks = self.get_kv_usage()
        if minimal_required_blocks > 0:
            minimal_required_blocks = max(
                minimal_required_blocks,
                math.ceil(scheduler_config.minimal_tokens_per_instance / self.kv_block_size)
            )
        return minimal_required_blocks

    def get_kv_recommended_blocks(self) -> int:
        """计算推荐 KV block 数（含扩容水位）。"""
        recommended_blocks = math.ceil(
            self.get_kv_usage() * (1 + scheduler_config.kv_scale_watermark)
        )
        if recommended_blocks > 0:
            recommended_blocks = max(
                recommended_blocks,
                math.ceil(scheduler_config.minimal_tokens_per_instance / self.kv_block_size)
            )
        return recommended_blocks

    def check_whether_need_kv_scale_in(self):
        """检查是否需要 KV cache 缩容。

        当推荐 block 数显著低于当前分配时触发缩容。
        serverlessllm 模式下不缩容（避免频繁操作）。
        CPU Worker 不缩容（KV 预分配）。
        """
        if scheduler_config.system == 'serverlessllm':
            return
        if self.node_type == 'cpu':
            return
        recommend_blocks = self.get_kv_recommended_blocks()
        if recommend_blocks < self.num_blocks_local / (1 + scheduler_config.kv_scale_watermark):
            self.register_kv_scale_action(new_num_blocks=recommend_blocks)

    # ==================================================================
    # 请求管理
    # ==================================================================

    def activate_request(self, request_tracker: ReqTracker):
        """正式激活一个请求到本 Worker。"""
        request_id = request_tracker.request_id
        assert request_id not in self.running_requests
        self.served_request_cnt += 1
        self.running_requests[request_id] = request_tracker
        assert request_tracker.detach_from_worker
        request_tracker.detach_from_worker = False
        request_tracker.set_location(self.node_type, self.node_id, self.worker_id)

    def shadow_add_request(self, request_tracker: ReqTracker):
        """影子添加 — 仅在本地模拟添加，用于调度可行性检查。"""
        request_id = request_tracker.request_id
        assert request_id not in self.running_requests
        self.running_requests[request_id] = request_tracker
        assert request_tracker.detach_from_worker
        request_tracker.detach_from_worker = False

    def shadow_del_request(self, request_tracker: ReqTracker):
        """影子删除 — 撤销影子添加。"""
        request_id = request_tracker.request_id
        assert request_id in self.running_requests
        self.running_requests.pop(request_id)
        assert not request_tracker.detach_from_worker
        request_tracker.detach_from_worker = True

    def delete_request_tracker(self, request_tracker: ReqTracker):
        """删除一个请求的追踪记录。

        删除后检查是否需要 KV cache 缩容。
        若所有请求完成，记录空闲开始时间。
        """
        assert not request_tracker.detach_from_worker
        request_tracker.detach_from_worker = True
        self.running_requests.pop(request_tracker.request_id)
        self.check_whether_need_kv_scale_in()
        if len(self.running_requests) == 0:
            self.idle_start_time = time.time()

    def allocate_with_model(self, model_id: int):
        """标记 Worker 已分配指定模型。"""
        assert not self.allocated
        assert self.model_id == -1
        self.allocated = True
        self.cur_kv_scale_log_list = []
        self.allocate_time = time.time()
        self.model_id = model_id

    async def deallocate(self):
        """释放 Worker — 等待所有请求完成后清理 GPU 资源。"""
        assert self.allocated
        assert self.model_id != -1

        # 等待所有运行中请求完成
        while len(self.running_requests) > 0:
            await asyncio.sleep(0.1)

        lifetime = round(time.time() - self.allocate_time, 3)
        if self.node_type == 'gpu':
            # 等待 GPU 资源完全释放
            while self.hold_model_remote or self.num_blocks_remote > 0:
                await asyncio.sleep(0.1)
            self.past_lifecycle_kv_scale_log_list.append(
                (lifetime, self.cur_kv_scale_log_list)
            )
            self.cur_kv_scale_log_list = []

        self.allocated = False
        self.model_id = -1

    # ==================================================================
    # 计算时间片
    # ==================================================================

    def get_total_token_num_and_batch_size(self) -> tuple:
        """获取总 token 数和当前 batch 大小。"""
        total_token_num = 0
        bs = 0
        for req in self.running_requests.values():
            if not req.detach_from_worker:
                bs += 1
                total_token_num += req.total_length()
        return total_token_num, bs

    def update_time_slice(self) -> float:
        """更新并返回当前 decode 时间片。

        时间片 = 预期 decode 耗时 / TPOT_SLO

        model_id >= 256: prefill 专用实例，time_slice = 0
        batch_size == 0: 无请求，time_slice = 0
        """
        if self.model_id >= 256:
            self.time_slice = 0.0
            return self.time_slice

        total_token_num, batch_size = self.get_total_token_num_and_batch_size()
        if batch_size == 0:
            self.time_slice = 0.0
        else:
            avg_context_len = total_token_num / batch_size
            self.time_slice = (
                estimate_duration(self.model_type, self.node_type, self.node_label,
                                  avg_context_len, batch_size, 'decode')
                / self.TPOT
            )
        return self.time_slice

    # ==================================================================
    # DDL 计算
    # ==================================================================

    def get_prefill_ddl_and_duration(self) -> tuple:
        """计算 prefill 的 DDL 和预计耗时。

        Returns:
            (next_token_ddl, prefill_duration_seconds)
        """
        prefill_duration = 0.0
        for req in self.running_requests.values():
            if req.under_prefill and (not req.detach_from_worker):
                prefill_duration += estimate_duration(
                    self.model_type, self.node_type, self.node_label,
                    req.total_length(), -1, 'prefill'
                )
        return self.get_next_token_ddl(), prefill_duration

    def get_next_token_ddl(self) -> float:
        """获取下一个 token 的最早 DDL。

        Returns:
            最小 DDL（秒，Unix 时间戳）；无请求时返回 1e18
        """
        ddl = 1e18
        for req in self.running_requests.values():
            if not req.detach_from_worker:
                ddl = min(ddl, req.expect_next_token_time())
        return ddl

    # ==================================================================
    # 信息查询
    # ==================================================================

    def get_info(self) -> dict:
        """返回 Worker 的标识信息。"""
        return {
            'pool': self.node_type,
            'node': self.node_id,
            'worker': self.worker_id,
            'model': self.model_id,
        }

    def delay_requests_due_to_model_loading(self, delay_time: float):
        """为所有运行中请求添加模型加载延迟。

        延迟会被计入请求的 tolerate_time，放宽其 SLO。
        """
        for req in self.running_requests.values():
            req.tolerate_time += delay_time
            if req.output_length == 0:
                req.TTFT_cold_start = True

    # ==================================================================
    # 请求发送
    # ==================================================================

    def fire_request_async(self, request_tracker: ReqTracker):
        """异步发送推理请求到 vLLM Worker。

        使用 fire_version 确保只有一个活跃的 fire_task。
        """
        request_tracker.fire_version += 1
        assert request_tracker.fire_task is None
        request_tracker.fire_task = asyncio.create_task(
            self.fire_request_sync(request_tracker, request_tracker.fire_version)
        )

    async def fire_request_sync(self, request_tracker: ReqTracker, fire_ver: int):
        """同步发送推理请求到 vLLM Worker 并流式处理响应。

        通过 HTTP POST /v1/completions (stream=True) 发送请求，
        使用 Server-Sent Events 逐个接收 token。

        被取消时 (CancelledError): 由 perform_evict 或 check_finish 触发。
        """
        try:
            assert not request_tracker.check_finish()
            assert self.model_id == request_tracker.model_id

            # 设置默认 prompt（如果未提供）
            if request_tracker.text is None:
                request_tracker.text = 'There are many countries in the world'

            prompt = request_tracker.text
            expect_length = request_tracker.expect_output_length - request_tracker.output_length
            url = f'http://{self.node_ip}:{self.port}/v1/completions'
            remote_request_id = f'{request_tracker.request_id}-{fire_ver}'

            data = {
                "model": self.model_path,
                "prompt": prompt,
                "min_tokens": expect_length,
                "max_tokens": expect_length,
                'request_id': remote_request_id,
                "stream": True,
            }

            logger.info(
                f'fire request {remote_request_id} to '
                f'{request_tracker.node_type}-{request_tracker.node_id:02d}-'
                f'{request_tracker.worker_id:02d}, '
                f'cur_out: {request_tracker.output_length}, '
                f'exp_out: {request_tracker.expect_output_length}'
            )

            async with self.session.post(url, json=data) as response:
                async for line in response.content:
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line == '\n':
                            continue
                        if decoded_line[:12] == 'data: [DONE]':
                            continue
                        now_json = json.loads(decoded_line[6:])
                        request_tracker.receive_new_token(
                            now_json['choices'][0]['text']
                        )
                        request_tracker.check_finish()

        except asyncio.CancelledError:
            logger.info(
                f'{request_tracker.request_id} fire_ver {fire_ver} canceled'
            )

    # ==================================================================
    # Action 系统 — 注册
    # ==================================================================

    def register_sleep_action(self, wakeup_time: float):
        """注册睡眠 Action（sllm+share 专用）。

        直接加入 dispatched_action_list，无需中心协调。
        """
        self.action_version += 1
        self.dispatched_action_list.append(
            WorkerSleepAction(self.worker_id, self.action_version, wakeup_time)
        )

    def register_evict_requests_action(self):
        """注册请求驱逐 Action。

        将所有运行中请求标记为 detach_from_worker。
        """
        self.action_version += 1
        new_action = WorkerEvictRequestsAction(self.worker_id, self.action_version)
        for req in self.running_requests.values():
            assert not req.detach_from_worker
            req.detach_from_worker = True

        # 分布式调度模式: 直接执行，无需中心协调
        if self.using_dist_scheduler:
            assert self.node_type == 'cpu'
            asyncio.create_task(self.perform_evict_requests_action(new_action))
            return

        self.actions_register_queue.put_nowait(new_action)

    def register_load_action(self):
        """注册模型加载 Action（仅 GPU Worker）。"""
        assert self.node_type == 'gpu'
        assert not self.hold_model_local

        self.action_version += 1
        self.hold_model_local = True
        self.hold_model_local_version = self.action_version
        self.actions_register_queue.put_nowait(
            WorkerLoadAction(self.worker_id, self.action_version)
        )

    def register_offload_action(self):
        """注册模型卸载 Action（仅 GPU Worker）。"""
        assert self.node_type == 'gpu'
        assert self.hold_model_local is True

        self.action_version += 1
        self.hold_model_local = False
        self.hold_model_local_version = self.action_version
        self.actions_register_queue.put_nowait(
            WorkerOffloadAction(self.worker_id, self.action_version)
        )

    def register_kv_scale_action(self, new_num_blocks: int):
        """注册 KV cache 扩缩容 Action（仅 GPU Worker）。"""
        assert self.node_type == 'gpu'
        self.action_version += 1
        self.num_blocks_local = new_num_blocks
        self.num_blocks_local_version = self.action_version
        self.actions_register_queue.put_nowait(
            WorkerKVScaleAction(self.worker_id, self.action_version, new_num_blocks)
        )

    # ==================================================================
    # Action 系统 — 分发
    # ==================================================================

    def dispatch_evict_requests_action(self, action: WorkerEvictRequestsAction):
        """分发请求驱逐 Action。"""
        self.dispatched_action_list.append(action)
        self.check_whether_performing_dispatched_actions()

    def dispatch_load_action(self, action: WorkerLoadAction):
        """分发模型加载 Action。"""
        assert self.hold_model_local is True
        self.is_loading_model = True
        self.loading_event.clear()
        self.dispatched_action_list.append(action)
        self.check_whether_performing_dispatched_actions()

    def dispatch_offload_action(self, action: WorkerOffloadAction):
        """分发模型卸载 Action。"""
        self.is_offloading_model = True
        self.offloading_event.clear()
        self.dispatched_action_list.append(action)
        self.check_whether_performing_dispatched_actions()

    def dispatch_kv_scale_action(self, action: WorkerKVScaleAction):
        """分发 KV cache 扩缩容 Action。"""
        self.dispatched_action_list.append(action)
        self.check_whether_performing_dispatched_actions()

    # ==================================================================
    # Action 系统 — 执行与协调
    # ==================================================================

    async def digest_dispatched_actions(self):
        """依次执行所有已分发的 Action。

        执行顺序: FIFO (dispatched_action_list[0] 先执行)。
        执行完成后通知调度器 (finished_action 事件)。
        """
        assert self.is_performing_action
        while len(self.dispatched_action_list) > 0:
            action = self.dispatched_action_list[0]
            logger.debug(
                f'{self.node_type}-{self.node_id:02d}-{self.worker_id:02d} '
                f'perform {action.__class__.__name__}'
            )

            if isinstance(action, WorkerLoadAction):
                await self._perform_load_action_impl()
            elif isinstance(action, WorkerOffloadAction):
                await self._perform_offload_action_impl(action)
            elif isinstance(action, WorkerKVScaleAction):
                await self._perform_kv_scale_action_impl(action)
            elif isinstance(action, WorkerEvictRequestsAction):
                await self._perform_evict_requests_action_impl(action)
            elif isinstance(action, WorkerSleepAction):
                await self._perform_sleep_action_impl(action)
            else:
                raise Exception(f"Unknown action type: {type(action)}")

            # 完成后才从列表中移除
            self.dispatched_action_list.pop(0)

        self.is_performing_action = False
        self.hanging_events.put_nowait(WorkerHangingReleaseType.finished_action)

    def check_whether_performing_dispatched_actions(self):
        """检查并开始执行已分发的 Action。

        仅当 Worker 未被调度且有待执行 Action 时启动执行。
        """
        if self.being_scheduled:
            return
        if self.is_performing_action or len(self.dispatched_action_list) == 0:
            return

        self.is_performing_action = True
        asyncio.create_task(self.digest_dispatched_actions())

    # ==================================================================
    # Action 系统 — 具体实现
    # ==================================================================

    async def _perform_sleep_action_impl(self, action: WorkerSleepAction):
        """执行睡眠 Action。"""
        await asyncio.sleep(max(0.0, action.wakeup_time - time.time()))

    async def _perform_evict_requests_action_impl(self, action: WorkerEvictRequestsAction):
        """执行请求驱逐。

        1. 向 vLLM Worker 发送驱逐请求（不保存 KV cache）
        2. 本地清理请求状态
        """
        evict_requests = list(self.running_requests.values())
        await self._evict_requests_remote(evict_requests, save_kv=False)

        for request in evict_requests:
            if not request.check_finish():
                request.perform_evict(save_kv=False)
        self.running_requests.clear()

    async def _perform_load_action_impl(self):
        """执行模型加载。

        仅 GPU Worker 需要加载。加载完成后延迟补偿已加入的请求。
        """
        if self.node_type == 'gpu':
            load_st = time.perf_counter()
            await self._trigger_remote_model_loading()
            load_ed = time.perf_counter()
            self.delay_requests_due_to_model_loading(load_ed - load_st)

        assert self.hold_model_remote is True
        self.is_loading_model = False
        self.loading_event.set()

    async def _perform_offload_action_impl(self, action: WorkerOffloadAction):
        """执行模型卸载。

        卸载后清理 remote 状态。若版本已更新则不重复操作。
        """
        await self._trigger_remote_model_offloading(new_num_blocks=-1)
        self.is_offloading_model = False
        self.offloading_event.set()

        # 检查是否有更新的 load action 已提交
        if self.hold_model_remote_version > action.action_id:
            pass  # 有更新的 action 覆盖了本次卸载
        else:
            assert self.hold_model_remote_version < action.action_id
            self.hold_model_remote = False
            self.hold_model_remote_version = action.action_id
            self.actions_register_queue.put_nowait(
                WorkerGiveOutMemory(self.worker_id)
            )

    async def _perform_kv_scale_action_impl(self, action: WorkerKVScaleAction):
        """执行 KV cache 扩缩容。

        扩容: remote 值已提前提交（early commit）
        缩容: 执行后更新 remote 值
        """
        await self._scale_remote_kv(action.new_num_blocks)

        if self.num_blocks_remote_version > action.action_id:
            # 有更新的 action 覆盖
            pass
        elif self.num_blocks_remote_version == action.action_id:
            # 扩容: remote 已提前提交
            assert self.num_blocks_remote == action.new_num_blocks
        else:
            # 缩容: 执行后更新，通知释放内存
            assert self.num_blocks_remote >= action.new_num_blocks
            if self.num_blocks_local_version == action.action_id:
                self.num_blocks_remote = action.new_num_blocks
                self.num_blocks_remote_version = action.action_id
                self.actions_register_queue.put_nowait(
                    WorkerGiveOutMemory(self.worker_id)
                )

    # ==================================================================
    # 远程通信 — vLLM Worker API 调用
    # ==================================================================

    async def _trigger_remote_model_loading(self):
        """调用 vLLM Worker: POST /load_model"""
        assert self.node_type == 'gpu'
        url = f'http://{self.node_ip}:{self.port}/load_model'
        async with self.session.post(url) as response:
            res = await response.json()
        assert res['result'] is True

    async def _trigger_remote_model_offloading(self, new_num_blocks: int):
        """调用 vLLM Worker: POST /offload_model"""
        assert self.node_type == 'gpu'
        url = f'http://{self.node_ip}:{self.port}/offload_model'
        async with self.session.post(url, json={'new_num_blocks': new_num_blocks}) as response:
            res = await response.json()
        assert res['result'] is True

    async def _scale_remote_kv(self, new_num_blocks: int):
        """调用 vLLM Worker: POST /kv_scale"""
        logger.debug(
            f'{self.node_type}-{self.node_id:02d}-{self.worker_id:02d} '
            f'scale num_blocks to {new_num_blocks}'
        )
        st = time.perf_counter()
        try:
            async with self.session.post(
                    f'http://{self.node_ip}:{self.port}/kv_scale',
                    json={'new_num_blocks': new_num_blocks}) as response:
                res = await response.json()
        except Exception:
            return
        ed = time.perf_counter()
        assert res['result'] is True
        self.cur_kv_scale_log_list.append((
            res['old_num_blocks'] * self.per_kv_block_memory_KB,
            res['new_num_blocks'] * self.per_kv_block_memory_KB,
            round(ed - st, 3),
        ))

    async def _evict_requests_remote(self, requests_list: list[ReqTracker], save_kv: bool):
        """调用 vLLM Worker: POST /evict_requests"""
        async with self.session.post(
                f'http://{self.node_ip}:{self.port}/evict_requests',
                json={
                    'request_id_list': [
                        f'{req.request_id}-{req.fire_version}-0'
                        for req in requests_list
                    ],
                    'save_kv': save_kv,
                }) as response:
            res = await response.json()
        assert res['result'] is True

    async def _clear_worker_remote(self):
        """调用 vLLM Worker: POST /clear_worker"""
        async with self.session.post(
                f'http://{self.node_ip}:{self.port}/clear_worker') as response:
            res = await response.json()
        assert res['result'] is True

    async def _register_worker_info_remote(self, worker_info: dict, dist_info: dict):
        """调用 vLLM Worker: POST /register_worker"""
        async with self.session.post(
                f'http://{self.node_ip}:{self.port}/register_worker',
                json={'worker_info': worker_info, 'dist_info': dist_info}) as response:
            res = await response.json()
        assert res['result'] is True

    async def perform_kv_send(self, dst_worker: "Worker", request_list: list[ReqTracker]):
        """调用 vLLM Worker: POST /kv_send（将 KV cache 发送到目标 Worker）"""
        request_id_list = [request.request_id for request in request_list]
        transfer_config = {
            'src_rank': self.rank,
            'dst_rank': dst_worker.rank,
            'dst_socket_addr': dst_worker.node_ip,
            'dst_socket_port': dst_worker.port + 10000,
        }
        async with self.session.post(
                f'http://{self.node_ip}:{self.port}/kv_send',
                json={
                    'transfer_config': transfer_config,
                    'request_id_list': request_id_list,
                }) as response:
            res = await response.json()
        assert res['result'] is True

    # ==================================================================
    # Worker 启动与初始化
    # ==================================================================

    async def start_worker(self, worker_info: dict, dist_info: dict,
                           start_lock: asyncio.Lock):
        """Worker 启动流程。

        步骤:
          1. 清理 vLLM Worker 状态
          2. GPU: 加载模型 + 分配初始 KV cache (256 blocks)
          3. 发送预热请求确认 Worker 可用
          4. 注册 Worker 信息到 vLLM
          5. GPU: 卸载模型 + 清零 KV cache（恢复到就绪状态）
          6. 标记启动完成
        """
        async with start_lock:
            await self._clear_worker_remote()

            if self.node_type == 'gpu':
                # GPU Worker 需要预热
                self.register_load_action()
                self.register_kv_scale_action(256)
                # 等待模型加载完成
                while (not self.hold_model_remote) or self.is_performing_action:
                    await asyncio.sleep(0.1)
                # 等待 KV scale 完成
                while self.num_blocks_remote != 256 or self.is_performing_action:
                    await asyncio.sleep(0.1)

            assert self.hold_model_local and self.hold_model_remote
            await self._fire_a_test_request()
            await asyncio.sleep(0.1)
            await self._register_worker_info_remote(worker_info, dist_info)

            if self.node_type == 'gpu':
                # GPU Worker 恢复到初始状态
                self.register_offload_action()
                self.register_kv_scale_action(0)
                while self.hold_model_remote or self.num_blocks_remote > 0:
                    await asyncio.sleep(0.1)

            self.start_complete.set()

    async def _fire_a_test_request(self):
        """发送一个最小化的预热请求确认 Worker 可用。

        使用 4 个 token 的 prompt，请求 16 个 token 输出。
        """
        url = f'http://{self.node_ip}:{self.port}/v1/completions'
        data = {
            "model": self.model_path,
            "prompt": [1, 2, 3, 4],
            "min_tokens": 16,
            "max_tokens": 16,
            "stream": True,
        }
        async with self.session.post(url, json=data) as response:
            await response.read()
            if response.status != 200:
                logger.warning(
                    f'Test request returned {response.status}, '
                    f'worker may need model pre-loaded'
                )
