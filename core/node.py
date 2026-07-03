"""Node 节点管理 — 物理机器的抽象与调度逻辑。

Node 表示集群中的一台物理机器（GPU 服务器或 CPU 服务器），
管理该机器上运行的所有 Worker 实例。

两类 Node:
  1. Node: 由全局 Gateway 管理的节点，执行内存检查、DDL 验证、抢占判断
  2. MiniNode/MiniWorker: 分布式调度器中的轻量代理（仅用于 CPU 节点的本地 DDL 调度）

核心调度流程 (Node.schedule):
  1. 收集所有 can_be_scheduled() 的 Worker
  2. DDL-based 选择: 优先选 DDL 最紧急的 Worker
  3. Batch-aware 回退: 若所有 Worker DDL 都充裕，选 batch 最大的 Worker
  4. 标记 being_scheduled，发送释放事件

"""

import math
import time
import json
import logging
import asyncio
from typing import Optional

import aiohttp
import requests
import websockets

from Mine.config import settings as config
from Mine.config.settings import scheduler_config
from Mine.models.enums import WorkerHangingReleaseType
from Mine.models.actions import (
    WorkerActionBase,
    WorkerLoadAction,
    WorkerOffloadAction,
    WorkerKVScaleAction,
    WorkerEvictRequestsAction,
    WorkerGiveOutMemory,
)
from Mine.models.request_tracker import ReqTracker
from Mine.core.worker import Worker
from Mine.core.power_estimator import estimate_duration, get_serverlessllm_concurrency

logger = logging.getLogger(__name__)

# ============================================================================
# MiniWorker — 分布式调度器的轻量 Worker 代理
# ============================================================================


class MiniWorker:
    """分布式调度器的轻量 Worker 代理（仅跟踪 DDL 和 batch）。"""

    def __init__(self, worker_id: int):
        self.worker_id: int = worker_id
        self.being_hanged: bool = False
        self.being_scheduled: bool = False
        self.ddl: float = 0.0
        self.batch_num: int = 0
        self.hanging_events: asyncio.Queue = asyncio.Queue()


# ============================================================================
# MiniNode — 分布式调度器的轻量 Node 代理
# ============================================================================


class MiniNode:
    """分布式调度器中的轻量 Node 代理。

    每个 CPU 节点本地运行一个 MiniNode 实例（通过 dist_gateway.py），
    负责该节点上所有 CPU Worker 的本地 DDL-based 调度。

    与全局 Node 的区别:
      - 不做内存检查（CPU 内存预分配）
      - 不做抢占（CPU 不可抢占）
      - 仅基于 DDL 和 batch 做调度决策
    """

    def __init__(self):
        self.workers: dict[int, MiniWorker] = {}
        self.have_schedule_permission: bool = True
        self.info_version: int = 0
        self.enable_stupid_schedule: bool = False
        self.ddl_based_schedule_config: dict = {
            'enable_batch_aware': False,
            'safe_ddl_threshold': 5,
        }

    def initialize_workers(self, worker_num: int):
        """初始化 Worker 列表。"""
        self.workers.clear()
        for worker_id in range(worker_num):
            self.workers[worker_id] = MiniWorker(worker_id)
        self.have_schedule_permission = True
        self.info_version = 0

    async def schedule(self):
        """执行一次调度 — 选择 DDL 最紧急或 batch 最大的 Worker。"""
        assert not self.have_schedule_permission
        ddl_min = 1e18
        target_worker = None
        can_be_scheduled_workers = []

        for worker in self.workers.values():
            if worker.being_hanged:
                assert not worker.being_scheduled
                can_be_scheduled_workers.append(worker)
                if worker.ddl < ddl_min:
                    ddl_min = worker.ddl
                    target_worker = worker

        if target_worker is None:
            self.have_schedule_permission = True
            return

        # DDL 充裕时切换到 batch-aware 调度
        if (self.ddl_based_schedule_config['enable_batch_aware']
                and ddl_min > self.ddl_based_schedule_config['safe_ddl_threshold']):
            batch_num_min = 1000
            for worker in can_be_scheduled_workers:
                if worker.batch_num < batch_num_min:
                    batch_num_min = worker.batch_num
                    target_worker = worker

        self.have_schedule_permission = False
        target_worker.being_scheduled = True
        target_worker.hanging_events.put_nowait(
            WorkerHangingReleaseType.being_scheduled
        )

    def acquire_schedule_permission(self) -> bool:
        """尝试获取调度许可（互斥）。"""
        if self.have_schedule_permission:
            self.have_schedule_permission = False
            return True
        else:
            return False


# ============================================================================
# Node — 完整节点管理
# ============================================================================


class Node:
    """集群中的一个物理节点，管理该节点上所有 Worker 的生命周期和调度。

    通过 Action Monitor 持续监听 Worker 的 Action 注册请求，
    根据节点内存预算决定是否立即分发或暂存。

    调度时执行多维检查:
      1. 内存预算 (check_worker_need_kv_scale_up)
      2. Decode 时间片 (check_decode_time_slice)
      3. Prefill DDL (check_prefill_DDLs_with_target_worker)
    """

    def __init__(
        self,
        node_type: str,
        node_id: int,
        node_info: dict,
        base_rank: int,
        world_size: int,
        session: aiohttp.ClientSession,
        gateway_logs: dict,
    ):
        """初始化 Node。

        Args:
            node_type: 'cpu' | 'gpu'
            node_id: 全局唯一 Node ID
            node_info: 从 pools_info_template 解析的节点配置
            base_rank: 该节点第一个 Worker 的全局 rank
            world_size: 集群总 worker 数
            session: 共享的 aiohttp 会话
            gateway_logs: Gateway 的日志字典 (用于详细日志)
        """
        self.node_type: str = node_type
        self.node_id: int = node_id
        self.node_label: str = node_info['node_label']
        self.workers_info: dict = node_info['workers']
        self.worker_num: int = len(self.workers_info)

        # 节点内存容量 (GB → KB)，减去每 Worker 1.5GB 的系统开销
        self.node_memory_capacity_GB: float = (
            node_info['node_memory_capacity_GB'] - self.worker_num * 1.5
        )
        self.node_memory_capacity_KB: float = self.node_memory_capacity_GB * 1024 * 1024

        self.node_ip: str = node_info['node_ip']
        self.gateway_ip: str = node_info['gateway_ip']
        self.base_port: int = node_info['base_port']
        self.session: aiohttp.ClientSession = session
        self.gateway_logs: dict = gateway_logs

        # Worker 管理
        self.workers: dict[int, Worker] = {}
        self.worker_actions_register_queue: asyncio.Queue = asyncio.Queue()
        cur_rank = base_rank
        self.rank_cnt: int = 0

        # 分布式调度器
        self.using_dist_scheduler: bool = node_info['dist_scheduler']
        self.scheduler_port: int = 7000
        if self.using_dist_scheduler:
            assert self.node_type == 'cpu'
            self.scheduler_port = self.base_port - 1
            self._init_dist_scheduler()
            asyncio.create_task(self._periodic_update_ddl_to_dist_scheduler())

        # 启动 Action Monitor
        asyncio.create_task(self._worker_action_monitor())

        # 创建并启动所有 Worker
        start_lock = asyncio.Lock()
        for worker_id, worker_info in self.workers_info.items():
            self.workers[worker_id] = Worker(
                self.node_type, self.node_label, self.node_id, worker_id,
                worker_info, self.node_ip, self.base_port, cur_rank,
                self.using_dist_scheduler,
                self.worker_actions_register_queue, session,
            )
            asyncio.create_task(self.workers[worker_id].start_worker(
                {
                    'pool_type': self.node_type,
                    'node_id': self.node_id,
                    'worker_id': worker_id,
                    'gateway_ip': self.gateway_ip,
                    'using_dist_scheduler': self.using_dist_scheduler,
                    'scheduler_port': self.scheduler_port,
                    'worker_ip': self.workers[worker_id].node_ip,
                    'worker_port': self.workers[worker_id].port,
                },
                {
                    'master_addr': self.gateway_ip,
                    'master_port': 30000,
                    'rank': cur_rank,
                    'world_size': world_size,
                    'socket_addr': self.workers[worker_id].node_ip,
                    'socket_port': self.workers[worker_id].port + 10000,
                },
                start_lock,
            ))
            cur_rank += 1
            self.rank_cnt += 1

        # 调度状态
        self.have_schedule_permission: bool = True
        self.last_schedule_time: float = -1.0

    # ==================================================================
    # 分布式调度器通信
    # ==================================================================

    def _init_dist_scheduler(self):
        """初始化远程分布式调度器 — HTTP POST /init"""
        is_sllm_share = (
            scheduler_config.system == 'serverlessllm'
            and scheduler_config.sllm_enable_sharing
        )
        r = requests.post(
            f'http://{self.node_ip}:{self.scheduler_port}/init',
            json={
                'worker_num': self.worker_num,
                'ddl_based_schedule_config': scheduler_config.ddl_based_schedule,
                'is_sllm_share': is_sllm_share,
            },
        )
        assert r.status_code == 200

    def update_dist_scheduler(self):
        """更新分布式调度器配置。"""
        is_sllm_share = (
            scheduler_config.system == 'serverlessllm'
            and scheduler_config.sllm_enable_sharing
        )
        r = requests.post(
            f'http://{self.node_ip}:{self.scheduler_port}/update_system_config',
            json={'is_sllm_share': is_sllm_share},
        )
        assert r.status_code == 200

    async def _periodic_update_ddl_to_dist_scheduler(self):
        """周期性 (每 100ms) 通过 WebSocket 发送 Worker 的 DDL 和 batch 信息。

        分布式调度器 (MiniNode) 接收这些信息后独立做调度决策。
        """
        info_version = 0
        uri = f'ws://{self.node_ip}:{self.scheduler_port}/ws/update_ddl'
        async with websockets.connect(uri) as websocket:
            while True:
                info_version += 1
                workers_ddl = {}
                workers_batch_num = {}
                now_time = time.time()

                for worker_id, worker in self.workers.items():
                    workers_batch_num[worker_id] = len(worker.running_requests)
                    if len(worker.running_requests) > 0:
                        workers_ddl[worker_id] = round(
                            worker.get_next_token_ddl() - now_time, 3
                        )
                    else:
                        workers_ddl[worker_id] = 0.0

                message = json.dumps({
                    'info_version': info_version,
                    'workers_ddl': workers_ddl,
                    'workers_batch_num': workers_batch_num,
                })
                await websocket.send(message)
                await asyncio.sleep(0.1)

    # ==================================================================
    # 状态查询
    # ==================================================================

    def is_allocated(self) -> bool:
        """检查是否有任何 Worker 已分配模型。"""
        for worker in self.workers.values():
            if worker.allocated:
                return True
        return False

    def get_density(self) -> int:
        """返回已分配模型的 Worker 数量。"""
        res = 0
        for worker in self.workers.values():
            if worker.allocated:
                res += 1
        return res

    async def check_start_complete(self):
        """等待所有 Worker 启动完成。"""
        for worker in self.workers.values():
            await worker.start_complete.wait()

    def acquire_schedule_permission(self) -> bool:
        """尝试获取调度许可（互斥锁）。"""
        if self.have_schedule_permission:
            self.have_schedule_permission = False
            return True
        else:
            return False

    # ==================================================================
    # 调度
    # ==================================================================

    async def schedule(self):
        """执行一次 Node 级别的调度。

        调度策略:
          1. 收集所有可调度的 Worker (can_be_scheduled)
          2. 默认: 选择 DDL 最紧急的 Worker
          3. Batch-aware 回退: 若所有 DDL 都充裕，选 batch 最大的 Worker
          4. 标记选中 Worker 的 being_scheduled，发送释放事件
        """
        logger.debug(f'node schedule start: {self.node_type}-{self.node_id:02d}')
        assert not self.have_schedule_permission

        now_time = time.time()

        # 收集可调度的 Worker
        can_be_scheduled_workers = []
        for worker in self.workers.values():
            if worker.can_be_scheduled():
                assert not worker.being_scheduled
                can_be_scheduled_workers.append(worker)
                ddl_cur = worker.get_next_token_ddl() - now_time
                logger.debug(
                    f'{self.node_type}-{self.node_id:02d}-worker{worker.worker_id:02d} '
                    f'ddl: {ddl_cur:.2f} s'
                )

        if scheduler_config.enable_detailed_logging:
            self.gateway_logs['node_schedule_times'].append(time.time() - now_time)

        # 选择目标 Worker
        target_worker = self._select_target_worker(can_be_scheduled_workers)

        if target_worker is None:
            self.have_schedule_permission = True
            logger.debug(
                f'node schedule end: {self.node_type}-{self.node_id:02d}: None'
            )
            return

        self.have_schedule_permission = False
        self.last_schedule_time = now_time
        logger.debug(
            f'node schedule end: {self.node_type}-{self.node_id:02d}: '
            f'schedule worker-{target_worker.worker_id:02d}'
        )
        target_worker.being_scheduled = True
        await target_worker.hanging_events.put(
            WorkerHangingReleaseType.being_scheduled
        )

    def _select_target_worker(self, can_be_scheduled_workers: list[Worker]) -> Optional[Worker]:
        """从可调度 Worker 列表中选择目标。

        优先 DDL 最紧急的；若 DDL 都充裕且启用 batch_aware，
        则选 batch 最大的 Worker。
        """
        if not can_be_scheduled_workers:
            return None

        now_time = time.time()
        ddl_min = 1e18
        target_worker = None

        for worker in can_be_scheduled_workers:
            ddl_cur = worker.get_next_token_ddl() - now_time
            if ddl_cur < ddl_min:
                ddl_min = ddl_cur
                target_worker = worker

        # Batch-aware 回退
        if (scheduler_config.ddl_based_schedule['enable_batch_aware']
                and ddl_min > scheduler_config.ddl_based_schedule['safe_ddl_threshold']):
            batch_num_max = 0
            for worker in can_be_scheduled_workers:
                if len(worker.running_requests) > batch_num_max:
                    batch_num_max = len(worker.running_requests)
                    target_worker = worker

        return target_worker

    # ==================================================================
    # 内存管理
    # ==================================================================

    def get_cur_node_memory_footprint(self) -> float:
        """计算节点当前内存占用 (KB)。"""
        memory_KB = 0.0
        for worker in self.workers.values():
            if worker.hold_model_remote:
                memory_KB += worker.model_memory_KB
            memory_KB += worker.num_blocks_remote * worker.per_kv_block_memory_KB
        return memory_KB

    def cal_node_memory_budget_with_target_worker(
        self, target_worker: Optional[Worker], new_num_blocks: Optional[int]
    ) -> float:
        """计算包含目标 Worker 未来状态的内存预算 (KB)。

        用于验证内存操作是否会导致 OOM。

        Args:
            target_worker: 目标 Worker（None 表示不修改）
            new_num_blocks: 目标 Worker 的新 block 数（None 表示不变）
        """
        memory_budget_KB = 0.0
        for worker in self.workers.values():
            if worker.hold_model_local or worker == target_worker:
                memory_budget_KB += worker.model_memory_KB
            if worker != target_worker:
                memory_budget_KB += worker.num_blocks_local * worker.per_kv_block_memory_KB
            else:
                memory_budget_KB += (
                    max(worker.num_blocks_local, new_num_blocks or 0)
                    * worker.per_kv_block_memory_KB
                )
        return memory_budget_KB

    def get_decode_time_slice(self) -> float:
        """计算节点总 decode 时间片（所有 Worker 的 time_slice 之和）。

        time_slice > 1 表示节点 decode 负载超限。
        """
        occupied_time_slice = 0.0
        for worker in self.workers.values():
            occupied_time_slice += worker.update_time_slice()
        return occupied_time_slice

    def check_decode_time_slice(self) -> bool:
        """检查 decode 时间片是否在容量内（≤ 1.0）。"""
        return self.get_decode_time_slice() <= 1.0

    # ==================================================================
    # KV Cache 管理
    # ==================================================================

    def check_worker_need_kv_scale_up(self, target_worker: Worker) -> tuple:
        """检查 Worker 是否需要 KV cache 扩容。

        Returns:
            (success: bool, new_num_blocks: int)
            - success=True, new_num_blocks=-1: 当前容量已足够
            - success=True, new_num_blocks>=0: 需要扩容到指定值
            - success=False: 无法扩容（内存不足）
        """
        minimal_required_blocks = target_worker.get_kv_minimal_required_blocks()

        # 当前容量已足够
        if (minimal_required_blocks <= target_worker.num_blocks_local
                and (target_worker.node_type == 'cpu' or target_worker.hold_model_local)):
            return True, -1

        # 需要扩容
        if target_worker.node_type == 'cpu':
            # CPU 不可扩容
            return False, -1

        # GPU 扩容
        if scheduler_config.system == 'serverlessllm' and scheduler_config.sllm_enable_sharing:
            # sllm+share: 固定分配，不管内存预算
            return True, math.floor(
                (self.node_memory_capacity_KB / scheduler_config.sllm_max_shares
                 - target_worker.model_memory_KB) / target_worker.per_kv_block_memory_KB
            )

        minimal_node_budget_KB = self.cal_node_memory_budget_with_target_worker(
            target_worker, minimal_required_blocks
        )
        if minimal_node_budget_KB > self.node_memory_capacity_KB:
            return False, -1

        if scheduler_config.system == 'serverlessllm':
            recommend_blocks = 1000000  # 全部分配
        else:
            recommend_blocks = target_worker.get_kv_recommended_blocks()

        return True, min(
            recommend_blocks,
            minimal_required_blocks + math.floor(
                (self.node_memory_capacity_KB - minimal_node_budget_KB)
                / target_worker.per_kv_block_memory_KB
            ),
        )

    # ==================================================================
    # Prefill DDL 验证
    # ==================================================================

    def check_future_workload(self, workers_ddl_time_pair: list) -> bool:
        """检查未来工作负载是否满足 DDL。

        Args:
            workers_ddl_time_pair: [(ddl, duration, is_target), ...]
        """
        workers_ddl_time_pair.sort()
        cur_time = time.time()
        for ddl_time_pair in workers_ddl_time_pair:
            cur_time += ddl_time_pair[1]
            if cur_time > ddl_time_pair[0]:
                return False
        return True

    def check_prefill_DDLs_with_target_worker(self, target_worker: Worker) -> bool:
        """影子验证：模拟加入 target_worker 后，所有请求的 prefill DDL 是否仍满足。

        这是调度决策中最精细的检查 — 确保将新请求加入某个 Worker 后，
        不会导致该 Worker 或其他 Worker 的 prefill 请求超时。

        Returns:
            True: 所有 prefill DDL 可以满足
            False: 加入后会导致某个 prefill DDL 违反
        """
        workers_ddl_time_pair = []
        for worker in self.workers.values():
            # 忽略 GPU Worker 正在加载模型的情况
            if self.node_type == 'cpu' or (not worker.exist_loading_event()):
                prefill_ddl, prefill_duration = worker.get_prefill_ddl_and_duration()
                if prefill_duration > 0:
                    is_target = (1 if worker == target_worker else 0)
                    workers_ddl_time_pair.append(
                        (prefill_ddl, prefill_duration, is_target)
                    )
        workers_ddl_time_pair.sort()

        # 计算起点时间
        if self.have_schedule_permission or self.using_dist_scheduler:
            cur_time = time.time()
        else:
            cur_time = self.last_schedule_time

        enter_influence_zone = False
        for ddl_time_pair in workers_ddl_time_pair:
            if ddl_time_pair[2] == 1:
                enter_influence_zone = True
            cur_time += ddl_time_pair[1]
            if enter_influence_zone and cur_time > ddl_time_pair[0]:
                return False
        return True

    # ==================================================================
    # 容量检查
    # ==================================================================

    def can_add_request_to_worker(
        self, target_worker: Worker, request_tracker: ReqTracker, ignores=None
    ) -> bool:
        """检查是否可以将请求添加到指定 Worker。

        三种检查模式:
          1. serverlessllm: 仅检查并发限制
          2. sota (完整): 内存 → decode → prefill DDL
          3. sota (部分): 通过 ignores 跳过某些检查（用于抢占判断）

        Args:
            ignores: 要跳过的检查列表，如 ['memory'] 或 ['decode']
        """
        if ignores is None:
            ignores = []

        # ServerlessLLM 模式
        if scheduler_config.system == 'serverlessllm':
            assert len(ignores) == 0
            sllm_max_shares = 1
            if scheduler_config.sllm_enable_sharing:
                sllm_max_shares = scheduler_config.sllm_max_shares
            return len(target_worker.running_requests) <= get_serverlessllm_concurrency(
                target_worker.model_type, target_worker.node_type,
                target_worker.node_label, sllm_max_shares
            )

        # SOTA 模式
        logger.debug(
            f'Checking can add request {request_tracker.request_id} '
            f'to worker {target_worker.get_info()}...'
        )

        # 内存检查
        if 'memory' not in ignores:
            logger.debug('Checking worker memory usage...')
            success, new_num_blocks = self.check_worker_need_kv_scale_up(target_worker)
            if not success:
                logger.debug("Memory check failed!")
                return False
            logger.debug('Memory check pass.')

        # Decode 检查
        if 'decode' not in ignores:
            logger.debug('Checking decode time-slice...')
            if not self.check_decode_time_slice():
                logger.debug('Decode check failed!')
                return False
            logger.debug('Decode check pass.')

        # Prefill DDL 检查
        logger.debug('Checking prefill DDLs...')
        start_time = time.time()
        shadow_validation_pass = self.check_prefill_DDLs_with_target_worker(target_worker)
        if scheduler_config.enable_detailed_logging:
            self.gateway_logs['shadow_validation_times'].append(
                time.time() - start_time
            )
        if shadow_validation_pass:
            logger.debug('Prefill check pass. Request can add to worker!')
            return True
        else:
            logger.debug('Prefill check failed!')
            return False

    # ==================================================================
    # Worker 分配
    # ==================================================================

    def try_allocate_worker(self, request_tracker: ReqTracker) -> Optional[Worker]:
        """尝试分配一个新 Worker 来服务请求。

        分配限制:
          - serverlessllm: 受 sllm_max_shares 限制，且 CPU 节点只能有一个 13B 模型
          - sota (enable_sharing=False): 每个节点最多一个已分配 worker
        """
        if scheduler_config.system == 'serverlessllm' or (not scheduler_config.enable_sharing):
            allocate_num = 0
            hold_13b_model = False
            for worker in self.workers.values():
                if worker.allocated:
                    allocate_num += 1
                    if worker.model_type == 'llama-2-13b':
                        hold_13b_model = True

            if scheduler_config.system == 'serverlessllm' and scheduler_config.sllm_enable_sharing:
                if allocate_num >= scheduler_config.sllm_max_shares:
                    return None
                if self.node_type == 'cpu':
                    if hold_13b_model or (
                        allocate_num >= 1 and request_tracker.model_type == 'llama-2-13b'
                    ):
                        return None
            else:
                if allocate_num > 0:
                    return None

        for target_worker in self.workers.values():
            if target_worker.can_allocate_with_model(request_tracker.model_type):
                assert target_worker.empty()
                target_worker.shadow_add_request(request_tracker)
                success = self.can_add_request_to_worker(target_worker, request_tracker)
                target_worker.shadow_del_request(request_tracker)
                if success:
                    return target_worker
                else:
                    return None
        return None

    # ==================================================================
    # 抢占
    # ==================================================================

    def find_decode_preemption_with_target_worker(
        self, target_worker: Worker
    ) -> Optional[Worker]:
        """为 decode 绑定寻找可抢占的 GPU Worker。

        CPU 节点的 decode 抢占策略:
          按 'compute' (time_slice) 或 'batch' (running_requests 数量) 排序，
          找到可以被 target_worker 替代的 worker。

        Returns:
            可被抢占的 Worker，或 None（无法抢占）
        """
        assert self.node_type == 'cpu'
        assert scheduler_config.decode_preempt_metric in ['batch', 'compute']

        decode_time_slice_gap = self.get_decode_time_slice() - 1.0
        target_time_slice = target_worker.update_time_slice()

        # target_worker 已经超限
        if target_time_slice > 1:
            return None

        preemption_candidates = []

        if scheduler_config.decode_preempt_metric == 'compute':
            for worker in self.workers.values():
                if worker == target_worker:
                    continue
                this_time_slice = worker.update_time_slice()
                if 0 < this_time_slice <= target_time_slice:
                    preemption_candidates.append((worker, this_time_slice))
            preemption_candidates.sort(key=lambda x: x[1])
        elif scheduler_config.decode_preempt_metric == 'batch':
            for worker in self.workers.values():
                if worker == target_worker:
                    continue
                if 0 < len(worker.running_requests) <= len(target_worker.running_requests):
                    preemption_candidates.append(
                        (worker, len(worker.running_requests))
                    )
            preemption_candidates.sort(key=lambda x: x[1])

        for entry in preemption_candidates:
            preempted_worker = entry[0]
            if preempted_worker.update_time_slice() >= decode_time_slice_gap:
                return preempted_worker
        return None

    def find_memory_preemption_with_target_worker(
        self, target_worker: Worker
    ) -> Optional[Worker]:
        """为内存需求寻找可抢占的 GPU Worker。

        GPU 节点的内存抢占策略:
          找到持有模型、batch 不超过 target 且释放后能满足内存缺口的 worker。

        Returns:
            可被抢占的 Worker，或 None
        """
        assert self.node_type == 'gpu'
        assert scheduler_config.memory_preempt_metric in ['batch', 'memory']

        minimal_required_blocks = target_worker.get_kv_minimal_required_blocks()
        memory_gap_KB = (
            self.cal_node_memory_budget_with_target_worker(
                target_worker, minimal_required_blocks
            ) - self.node_memory_capacity_KB
        )
        target_batch_degree = len(target_worker.running_requests)

        preemption_candidates = []
        for worker in self.workers.values():
            if worker == target_worker:
                continue
            if worker.hold_model_local and len(worker.running_requests) <= target_batch_degree:
                preemption_candidates.append((worker, len(worker.running_requests)))

        preemption_candidates.sort(key=lambda x: x[1])
        for entry in preemption_candidates:
            preempted_worker = entry[0]
            if preempted_worker.get_memory_footprint() >= memory_gap_KB:
                return preempted_worker
        return None

    # ==================================================================
    # Action 管理
    # ==================================================================

    def try_to_commit_worker_load_action(
        self, target_worker: Worker, action_id: int
    ) -> bool:
        """尝试提交模型加载 Action（内存检查）。"""
        assert target_worker.hold_model_remote is False
        cur_memory_KB = self.get_cur_node_memory_footprint()
        if cur_memory_KB + target_worker.model_memory_KB <= self.node_memory_capacity_KB:
            target_worker.hold_model_remote = True
            target_worker.hold_model_remote_version = action_id
            return True
        return False

    def try_to_commit_worker_kv_scale_action(
        self, target_worker: Worker, action_id: int, new_num_blocks: int
    ) -> bool:
        """尝试提交 KV 扩缩容 Action（内存检查）。

        scale-down 始终可以提交；scale-up 需要内存预算检查。
        """
        if new_num_blocks <= target_worker.num_blocks_remote:
            return True
        cur_memory_KB = self.get_cur_node_memory_footprint()
        delta_KB = (
            (new_num_blocks - target_worker.num_blocks_remote)
            * target_worker.per_kv_block_memory_KB
        )
        if cur_memory_KB + delta_KB <= self.node_memory_capacity_KB:
            target_worker.num_blocks_remote = new_num_blocks
            target_worker.num_blocks_remote_version = action_id
            return True
        return False

    def try_commit_and_dispatch_action(self, action: WorkerActionBase) -> bool:
        """尝试提交并分发一个 Action。

        对不同类型的 Action 执行相应的内存检查：
          - LoadAction: try_to_commit_worker_load_action
          - OffloadAction: 直接分发
          - KVScaleAction: try_to_commit_worker_kv_scale_action
          - EvictRequestsAction: 直接分发

        Returns:
            True: 成功分发
            False: 内存不足，暂存到 pending_actions
        """
        target_worker = self.workers[action.worker_id]

        if isinstance(action, WorkerLoadAction):
            if action.action_id != target_worker.hold_model_local_version:
                return True  # 过时的 action，丢弃
            success = self.try_to_commit_worker_load_action(
                target_worker, action.action_id
            )
            if success:
                target_worker.dispatch_load_action(action)
                return True
            return False

        elif isinstance(action, WorkerOffloadAction):
            if action.action_id != target_worker.hold_model_local_version:
                return True
            target_worker.dispatch_offload_action(action)
            return True

        elif isinstance(action, WorkerKVScaleAction):
            if action.action_id != target_worker.num_blocks_local_version:
                return True
            success = self.try_to_commit_worker_kv_scale_action(
                target_worker, action.action_id, action.new_num_blocks
            )
            if success:
                target_worker.dispatch_kv_scale_action(action)
                return True
            return False

        elif isinstance(action, WorkerEvictRequestsAction):
            target_worker.dispatch_evict_requests_action(action)
            return True

        else:
            raise Exception(f"Unknown action type: {type(action)}")

    async def _worker_action_monitor(self):
        """持续监听并处理 Worker Action 注册。

        WorkerGiveOutMemory: 触发 pending_actions 的重试提交
        其他 Action: 尝试立即提交/分发，失败则加入 pending_actions
        """
        pending_actions: list[WorkerActionBase] = []
        while True:
            worker_action = await self.worker_actions_register_queue.get()
            logger.debug(
                f'{self.node_type}-{self.node_id:02d}-{worker_action.worker_id:02d} '
                f'register {worker_action.__class__.__name__}'
            )

            if isinstance(worker_action, WorkerGiveOutMemory):
                # 有内存被释放，重试 pending_actions
                logger.debug(
                    f'{self.node_type}-{self.node_id:02d} '
                    f'pending_actions: {len(pending_actions)}'
                )
                while len(pending_actions) > 0:
                    success = self.try_commit_and_dispatch_action(pending_actions[0])
                    if success:
                        logger.debug(
                            f'{self.node_type}-{self.node_id:02d}-'
                            f'{pending_actions[0].worker_id:02d} '
                            f'commit {pending_actions[0].__class__.__name__}.'
                        )
                        pending_actions.pop(0)
                    else:
                        break
            else:
                success = self.try_commit_and_dispatch_action(worker_action)
                if not success:
                    logger.debug(
                        f'{self.node_type}-{self.node_id:02d}-'
                        f'{worker_action.worker_id:02d} '
                        f'move {worker_action.__class__.__name__} to pending list.'
                    )
                    pending_actions.append(worker_action)
                else:
                    logger.debug(
                        f'{self.node_type}-{self.node_id:02d}-'
                        f'{worker_action.worker_id:02d} '
                        f'commit {worker_action.__class__.__name__}.'
                    )
