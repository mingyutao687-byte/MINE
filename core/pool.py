"""资源池管理 — 集群中所有节点的全局视图。

Pool: 同类型节点集合 (CPU Pool 或 GPU Pool)
PoolManager: 全局资源管理器，协调 CPU Pool 和 GPU Pool 的调度

核心调度流程 (PoolManager.schedule_incoming_request):
  1. 尝试将请求加入已有 Worker (can_add_request_to_worker)
  2. 尝试通过抢占已有 Worker 来容纳请求 (can_add_request_to_worker_via_preemption)
  3. 分配全新 Worker (allocate_worker_for_request)
  4. 激活请求 + KV 扩容 + 发送请求

"""

import asyncio
import time
import logging
from typing import Optional

from Mine.config import settings as config
from Mine.config.settings import scheduler_config
from Mine.models.request_tracker import ReqTracker
from Mine.core.node import Node
from Mine.core.worker import Worker

logger = logging.getLogger(__name__)

# ============================================================================
# Pool — 同类型节点集合
# ============================================================================


class Pool:
    """同类型节点 (CPU 或 GPU) 的集合。

    提供:
      - 节点遍历和查询
      - 按 compute/memory 排序选择节点
      - Worker 分配和日志收集
    """

    def __init__(
        self,
        pool_type: str,
        nodes_config: dict[int, dict],
        base_rank: int,
        world_size: int,
        session,
        gateway_logs: dict,
    ):
        """初始化 Pool。

        Args:
            pool_type: 'cpu' | 'gpu'
            nodes_config: node_id → node_info 映射
            base_rank: 该 Pool 的起始全局 rank
            world_size: 集群总 worker 数
            session: 共享 aiohttp 会话
            gateway_logs: Gateway 日志字典
        """
        assert pool_type in ['cpu', 'gpu']
        self.pool_type: str = pool_type
        self.nodes: dict[int, Node] = {}
        cur_rank = base_rank
        self.rank_cnt: int = 0

        for node_id, node_info in nodes_config.items():
            new_node = Node(
                node_type=pool_type,
                node_id=node_id,
                node_info=node_info,
                base_rank=cur_rank,
                world_size=world_size,
                session=session,
                gateway_logs=gateway_logs,
            )
            self.nodes[node_id] = new_node
            cur_rank += new_node.rank_cnt
            self.rank_cnt += new_node.rank_cnt

    async def check_start_complete(self):
        """等待所有 Node 的 Worker 启动完成。"""
        for node in self.nodes.values():
            await node.check_start_complete()

    def count_allocated_nodes(self) -> int:
        """统计已分配模型的节点数。"""
        return sum(1 for node in self.nodes.values() if node.is_allocated())

    def collect_kv_scale_logs(self) -> list:
        """收集所有 Worker 的 KV 扩缩容日志。"""
        res = []
        for node in self.nodes.values():
            for worker in node.workers.values():
                res.extend(worker.past_lifecycle_kv_scale_log_list)
                worker.past_lifecycle_kv_scale_log_list = []
        return res

    def count_nodes_density(self) -> list:
        """统计每个节点的 Worker 密度。"""
        return [node.get_density() for node in self.nodes.values()]

    def count_node_batch(self) -> list:
        """统计每个节点每个 Worker 的 batch 大小。"""
        res = []
        for node in self.nodes.values():
            cur_node = [
                len(worker.running_requests)
                for worker in node.workers.values()
            ]
            res.append(cur_node)
        return res

    def count_node_memory(self) -> list:
        """统计每个节点每个 Worker 的内存详情。"""
        res = []
        for node in self.nodes.values():
            cur_node = [
                worker.get_monitor_memory_detail()
                for worker in node.workers.values()
            ]
            res.append(cur_node)
        return res

    def get_node_list_in_x_descent(self, metric: str) -> list[Node]:
        """按指定指标降序排列节点。

        Args:
            metric: 'compute' (decode 时间片) 或 'memory' (内存预算)
        """
        assert metric in ['compute', 'memory']
        node_list = []
        for node in self.nodes.values():
            if metric == 'compute':
                node_list.append((node, node.get_decode_time_slice()))
            elif metric == 'memory':
                node_list.append((
                    node,
                    node.cal_node_memory_budget_with_target_worker(None, None),
                ))
        node_list.sort(key=lambda x: x[1], reverse=True)
        return [entry[0] for entry in node_list]

    def try_allocate_worker(self, request_tracker: ReqTracker) -> Optional[Worker]:
        """尝试在 Pool 中分配一个空闲 Worker。

        排序策略:
          - CPU pool: 按 compute (decode 时间片) 排序
          - GPU pool: 按 memory 排序
          - enable_defragmentation: 降序（优先集中到已负载节点）
          - !enable_defragmentation: 升序（负载均衡）

        未分配的节点优先级更高（插入到列表末尾）。
        """
        if self.pool_type == 'cpu' and (not scheduler_config.enable_cpu):
            return None

        sort_metric = 'compute' if self.pool_type == 'cpu' else 'memory'
        node_list = self.get_node_list_in_x_descent(sort_metric)

        if not scheduler_config.enable_defragmentation:
            node_list.reverse()  # 负载均衡: 优先选择负载低的节点

        # 未分配的节点优先级更高
        unallocated_nodes = []
        for node in node_list[:]:  # 迭代副本
            if not node.is_allocated():
                node_list.remove(node)
                unallocated_nodes.append(node)
        node_list.extend(unallocated_nodes)

        for node in node_list:
            worker = node.try_allocate_worker(request_tracker)
            if worker is not None:
                return worker
        return None

    def get_worker(self, node_id: int, worker_id: int) -> Worker:
        """根据 ID 获取 Worker 对象。"""
        return self.nodes[node_id].workers[worker_id]


# ============================================================================
# ModelManager — 简单的模型-位置映射
# ============================================================================


class ModelManager:
    """模型 Manager — 记录模型被加载到哪个 Worker。

    当前为占位实现，仅存储 pool_type, node_id, worker_id。
    """

    def __init__(self, pool_type: str, node_id: int, worker_id: int):
        self.pool_type = pool_type
        self.node_id = node_id
        self.worker_id = worker_id


# ============================================================================
# PoolManager — 全局资源管理器
# ============================================================================


class PoolManager:
    """全局资源管理器 — 调度系统的顶层接口。

    协调 GPU Pool 和 CPU Pool，管理模型到 Worker 的映射，
    负责请求的完整调度流程。

    关键数据结构:
      - models_worker_list: model_id → [Worker, ...] (已分配该模型的 Worker 列表)
      - models_worker_round_robin_id: model_id → int (sllm round-robin 指针)
      - requests_tracker: request_id → ReqTracker (活跃请求字典)
    """

    def __init__(self, pools_config: dict, session):
        """初始化 PoolManager。

        Args:
            pools_config: {'gpu': {node_id: {...}}, 'cpu': {node_id: {...}}}
            session: 共享 aiohttp 会话
        """
        # 计算全局 world_size
        world_size = 0
        for pool_type in ['cpu', 'gpu']:
            for node_info in pools_config[pool_type].values():
                world_size += len(node_info['workers'])
        self.world_size: int = world_size

        # 日志
        self.logs: dict = {
            'node_schedule_times': [],
            'shadow_validation_times': [],
        }

        # 创建 Pool
        base_rank = 0
        self.gpu_pool = Pool(
            'gpu', pools_config['gpu'],
            base_rank=base_rank, world_size=world_size,
            session=session, gateway_logs=self.logs,
        )
        base_rank += self.gpu_pool.rank_cnt
        self.cpu_pool = Pool(
            'cpu', pools_config['cpu'],
            base_rank=base_rank, world_size=world_size,
            session=session, gateway_logs=self.logs,
        )

        # 模型 → Worker 映射
        self.models_worker_list: dict[int, list[Worker]] = {
            i: [] for i in range(512)
        }
        # ServerlessLLM round-robin 指针
        self.models_worker_round_robin_id: dict[int, int] = {
            i: 0 for i in range(512)
        }

        # 活跃请求
        self.requests_tracker: dict[str, ReqTracker] = {}

        self.under_logging: bool = False

    # ==================================================================
    # 监控
    # ==================================================================

    def start_monitor_async(self):
        """开始周期性监控日志记录。"""
        self.under_logging = True
        asyncio.create_task(self._log_node_usage())

    def end_monitor(self):
        """结束监控，收集最后的 KV scale 日志。"""
        self.under_logging = False
        self.logs['workers_kv_scale'] = self.gpu_pool.collect_kv_scale_logs()

    async def _log_node_usage(self):
        """周期性 (每秒) 记录节点使用情况。"""
        monitor_interval = 1
        self.logs['node_usage'] = {'cpu': [], 'gpu': []}
        self.logs['node_density'] = {'cpu': [], 'gpu': []}
        self.logs['batch'] = {'cpu': [], 'gpu': []}
        self.logs['memory'] = {'cpu': [], 'gpu': []}
        self.logs['node_schedule_times'] = []
        self.logs['shadow_validation_times'] = []

        wake_up_time = time.perf_counter()
        while self.under_logging:
            self.logs['node_usage']['cpu'].append(
                self.cpu_pool.count_allocated_nodes()
            )
            self.logs['node_usage']['gpu'].append(
                self.gpu_pool.count_allocated_nodes()
            )
            self.logs['node_density']['cpu'].append(
                self.cpu_pool.count_nodes_density()
            )
            self.logs['node_density']['gpu'].append(
                self.gpu_pool.count_nodes_density()
            )
            if scheduler_config.enable_detailed_logging:
                self.logs['batch']['cpu'].append(self.cpu_pool.count_node_batch())
                self.logs['batch']['gpu'].append(self.gpu_pool.count_node_batch())
                self.logs['memory']['cpu'].append(self.cpu_pool.count_node_memory())
                self.logs['memory']['gpu'].append(self.gpu_pool.count_node_memory())

            wake_up_time += monitor_interval
            await asyncio.sleep(wake_up_time - time.perf_counter())

    # ==================================================================
    # 初始化
    # ==================================================================

    async def check_start_complete(self):
        """等待所有 Pool 初始化完成。"""
        for pool in [self.cpu_pool, self.gpu_pool]:
            await pool.check_start_complete()

    # ==================================================================
    # Pool 查询
    # ==================================================================

    def _get_pool_from_pooltype(self, pool_type: str) -> Pool:
        """根据 pool_type 获取 Pool 实例。"""
        if pool_type == 'cpu':
            return self.cpu_pool
        elif pool_type == 'gpu':
            return self.gpu_pool
        else:
            raise ValueError(f"Unknown pool_type: {pool_type}")

    def get_node_from_worker(self, worker: Worker) -> Node:
        """从 Worker 反查其所属 Node。"""
        pool = self._get_pool_from_pooltype(worker.node_type)
        return pool.nodes[worker.node_id]

    def get_worker(self, worker_info: dict) -> Worker:
        """根据 worker_info (pool_type, node_id, worker_id) 获取 Worker。"""
        target_pool = self._get_pool_from_pooltype(worker_info['pool_type'])
        return target_pool.get_worker(worker_info['node_id'], worker_info['worker_id'])

    # ==================================================================
    # Worker 分配
    # ==================================================================

    def allocate_worker_for_request(self, request_tracker: ReqTracker) -> Optional[Worker]:
        """为请求分配一个全新的 Worker。

        根据 pool_priority 决定优先尝试 CPU 还是 GPU pool。
        """
        if scheduler_config.pool_priority == 'cpu':
            pools = [self.cpu_pool, self.gpu_pool]
        elif scheduler_config.pool_priority == 'gpu':
            pools = [self.gpu_pool, self.cpu_pool]
        else:
            raise ValueError(f"Invalid pool_priority: {scheduler_config.pool_priority}")

        for pool in pools:
            logger.debug(
                f'try allocate a worker for model-{request_tracker.model_id} '
                f'in {pool.pool_type} pool'
            )
            worker = pool.try_allocate_worker(request_tracker)
            if worker is not None:
                logger.debug(
                    f'allocation in {pool.pool_type} success! '
                    f'info: {worker.get_info()}'
                )
                return worker
            else:
                logger.debug('allocation failed!')

        logger.debug(
            f'Allocation in all pools failed! '
            f'This request will be dropped! {request_tracker.request_id}'
        )
        return None

    # ==================================================================
    # 请求激活
    # ==================================================================

    def activate_request(self, target_worker: Worker, request_tracker: ReqTracker):
        """将请求激活到指定 Worker。

        sllm 模式下同时更新 round-robin 指针。
        """
        if scheduler_config.system == 'serverlessllm':
            # 更新 round-robin
            now_id = -1
            for idx, worker in enumerate(
                self.models_worker_list[request_tracker.model_id]
            ):
                if worker == target_worker:
                    now_id = idx
                    break
            assert now_id != -1
            self.models_worker_round_robin_id[request_tracker.model_id] = now_id + 1

        request_id = request_tracker.request_id
        self.requests_tracker[request_id] = request_tracker
        target_worker.activate_request(request_tracker)

    # ==================================================================
    # 容量检查 — 加入已有 Worker
    # ==================================================================

    def can_add_request_to_worker(self, request_tracker: ReqTracker) -> Optional[Worker]:
        """尝试将请求加入一个已有的 Worker（无需分配新 Worker）。

        sllm 模式: round-robin 遍历
        sota 模式: 按 batch 排序遍历

        Returns:
            可以容纳该请求的 Worker，或 None
        """
        if scheduler_config.system == 'serverlessllm':
            model_id = request_tracker.model_id
            now_choice = self.models_worker_round_robin_id[model_id]
            target_worker_list = self.models_worker_list[model_id]
            for offset in range(len(target_worker_list)):
                target_worker = target_worker_list[
                    (now_choice + offset) % len(target_worker_list)
                ]
                target_node = self.get_node_from_worker(target_worker)
                target_worker.shadow_add_request(request_tracker)
                success = target_node.can_add_request_to_worker(
                    target_worker, request_tracker
                )
                target_worker.shadow_del_request(request_tracker)
                if success:
                    return target_worker
            return None

        # SOTA 模式
        target_worker_list = []
        for worker in self.models_worker_list[request_tracker.model_id]:
            target_worker_list.append((worker, len(worker.running_requests)))
        target_worker_list.sort(key=lambda x: x[1], reverse=True)

        if not scheduler_config.enable_defragmentation:
            target_worker_list.sort(key=lambda x: x[1], reverse=False)

        for entry in target_worker_list:
            target_worker = entry[0]
            target_node = self.get_node_from_worker(target_worker)
            target_worker.shadow_add_request(request_tracker)
            success = target_node.can_add_request_to_worker(
                target_worker, request_tracker
            )
            target_worker.shadow_del_request(request_tracker)
            if success:
                return target_worker
        return None

    # ==================================================================
    # 容量检查 — 抢占
    # ==================================================================

    def can_add_request_to_worker_via_preemption(
        self, request_tracker: ReqTracker
    ) -> tuple:
        """尝试通过抢占已有 Worker 来容纳请求。

        GPU: 忽略内存检查 → 找内存抢占目标
        CPU: 忽略 decode 检查 → 找 decode 抢占目标

        Returns:
            (target_worker, preempted_worker) 或 (None, None)
        """
        if not scheduler_config.enable_defragmentation or not scheduler_config.enable_preempt:
            return None, None
        if scheduler_config.system == 'serverlessllm':
            return None, None

        target_worker_list = []
        for worker in self.models_worker_list[request_tracker.model_id]:
            target_worker_list.append((worker, len(worker.running_requests)))
        target_worker_list.sort(key=lambda x: x[1], reverse=True)

        for entry in target_worker_list:
            target_worker = entry[0]
            target_node = self.get_node_from_worker(target_worker)

            target_worker.shadow_add_request(request_tracker)

            if target_node.node_type == 'gpu':
                is_memory_bound = target_node.can_add_request_to_worker(
                    target_worker, request_tracker, ignores=['memory']
                )
                if is_memory_bound:
                    preempted_worker = (
                        target_node.find_memory_preemption_with_target_worker(
                            target_worker
                        )
                    )
                    target_worker.shadow_del_request(request_tracker)
                    if preempted_worker is not None:
                        return target_worker, preempted_worker
            elif target_node.node_type == 'cpu':
                is_decode_bound = target_node.can_add_request_to_worker(
                    target_worker, request_tracker, ignores=['decode']
                )
                if is_decode_bound:
                    preempted_worker = (
                        target_node.find_decode_preemption_with_target_worker(
                            target_worker
                        )
                    )
                    target_worker.shadow_del_request(request_tracker)
                    if preempted_worker is not None:
                        return target_worker, preempted_worker

            target_worker.shadow_del_request(request_tracker)

        return None, None

    # ==================================================================
    # ★ 核心调度入口
    # ==================================================================

    def schedule_incoming_request(self, request_tracker: ReqTracker) -> bool:
        """★★★ 核心调度入口 — 将传入请求调度到合适的 Worker。

        三阶段调度策略:
          1. can_add_request_to_worker() — 尝试加入已有 Worker
          2. can_add_request_to_worker_via_preemption() — 尝试通过抢占加入
          3. allocate_worker_for_request() — 分配全新 Worker

        调度成功后:
          4. 激活请求 (activate_request)
          5. GPU KV 扩容 (check_worker_need_kv_scale_up)
          6. 发送推理请求 (fire_request_async)

        Returns:
            True: 调度成功
            False: 调度失败（请求将被丢弃或稍后重试）
        """
        model_id = request_tracker.model_id
        assert model_id in self.models_worker_list

        target_worker: Optional[Worker] = None

        # 阶段 1: 尝试加入已有 Worker
        target_worker = self.can_add_request_to_worker(request_tracker)

        # 阶段 2: 尝试抢占
        if target_worker is None:
            target_worker, preempted_worker = (
                self.can_add_request_to_worker_via_preemption(request_tracker)
            )
            if target_worker is not None and preempted_worker is not None:
                self._preempt_worker_async(preempted_worker)

        # 阶段 3: 分配新 Worker
        if target_worker is None:
            target_worker = self.allocate_worker_for_request(request_tracker)
            if target_worker is None:
                return False

            assert target_worker.model_id == -1
            assert target_worker.empty()
            target_worker.allocate_with_model(model_id)
            self.models_worker_list[model_id].append(target_worker)

            if target_worker.node_type == 'gpu':
                target_worker.register_load_action()

        # 阶段 4: 激活 + KV scale + 发送
        assert target_worker is not None
        self.activate_request(target_worker, request_tracker)

        target_node = self.get_node_from_worker(target_worker)
        success, new_num_blocks = target_node.check_worker_need_kv_scale_up(target_worker)
        assert success
        if new_num_blocks != -1:
            target_worker.register_kv_scale_action(new_num_blocks)

        target_worker.fire_request_async(request_tracker)
        return True

    # ==================================================================
    # 抢占与清理
    # ==================================================================

    def _preempt_worker_async(self, preempted_worker: Worker):
        """异步抢占一个 Worker。

        步骤:
          1. 从 models_worker_list 中移除
          2. GPU: 卸载模型 (register_offload_action)
          3. 驱逐请求 (register_evict_requests_action)
          4. GPU: 清零 KV cache (register_kv_scale_action(0))
          5. 解除分配 (deallocate)
        """
        logger.info(
            f'{preempted_worker.node_type}-{preempted_worker.node_id:02d}-'
            f'{preempted_worker.worker_id:02d} is preempted!'
        )
        self.models_worker_list[preempted_worker.model_id].remove(preempted_worker)

        if preempted_worker.node_type == 'gpu':
            preempted_worker.register_offload_action()
        preempted_worker.register_evict_requests_action()
        if preempted_worker.node_type == 'gpu':
            preempted_worker.register_kv_scale_action(new_num_blocks=0)

        asyncio.create_task(preempted_worker.deallocate())

    async def _clean_a_worker_after_keep_alive(
        self, worker: Worker, keep_alive_time: float,
        cur_served_request_cnt: int, cur_action_version: int,
    ):
        """keep-alive 超时后清理空闲 Worker。"""
        await asyncio.sleep(keep_alive_time)
        # 检查是否有新的请求或 action 到达
        if (worker.served_request_cnt != cur_served_request_cnt
                or worker.action_version != cur_action_version):
            return

        logger.debug(
            f'clean {worker.node_type}-{worker.node_id:02d}-'
            f'{worker.worker_id:02d} due to keep-alive timeout'
        )
        assert worker.allocated
        assert len(worker.running_requests) == 0

        self.models_worker_list[worker.model_id].remove(worker)
        if worker.node_type == 'gpu':
            worker.register_offload_action()
            worker.register_kv_scale_action(new_num_blocks=0)
        await worker.deallocate()

    def delete_request(self, request_tracker: ReqTracker):
        """删除请求追踪记录。

        若 Worker 变为空闲（无剩余请求），启动 keep-alive 倒计时：
          - GPU Worker: 等待 keep_alive_time 秒后清理
          - CPU Worker: 立即清理 (keep_alive_time = 0)
        """
        self.requests_tracker.pop(request_tracker.request_id, None)

        if request_tracker.detach_from_worker:
            return

        pool = self._get_pool_from_pooltype(request_tracker.node_type)
        worker = pool.nodes[request_tracker.node_id].workers[request_tracker.worker_id]
        worker.delete_request_tracker(request_tracker)

        if len(worker.running_requests) == 0:
            keep_alive_time = scheduler_config.keep_alive_time
            if worker.node_type == 'cpu':
                keep_alive_time = 0.0
            worker.action_version += 1
            asyncio.create_task(
                self._clean_a_worker_after_keep_alive(
                    worker=worker,
                    keep_alive_time=keep_alive_time,
                    cur_served_request_cnt=worker.served_request_cnt,
                    cur_action_version=worker.action_version,
                )
            )
