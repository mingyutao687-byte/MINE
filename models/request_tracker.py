"""请求追踪器 — 单个推理请求的完整生命周期管理。

ReqTracker 跟踪一个推理请求从创建到完成的全过程：
  1. 创建: 从 HTTP request body 解析，设置初始参数
  2. 调度: Gateway 将其分配给某个 Worker
  3. 执行: Worker 通过 fire_request_sync 将请求发送给 vLLM 后端
     - Prefill 阶段: 处理整个 prompt，生成第一个 token
     - Decode 阶段: 逐个生成后续 token
  4. 终止: 正常完成 (output_length == expect_output_length) 或被驱逐 (evict)
  5. 指标收集: get_e2e_metrics() 返回 TTFT、TPOT 等端到端指标

原文件: scheduler/request_info.py
"""

import asyncio
import time
import logging
from typing import Optional

# 兼容原代码中的 config 导入
from Mine.config import settings as config
from Mine.config.settings import scheduler_config, DDL_SAFE_THRESHOLD

# 预生成的输入 token 表: [1,2,...,4096] * 8，供测试用
_INPUT_TABLE = list(range(1, 4097)) * 8

logger = logging.getLogger(__name__)


class ReqTracker:
    """单个推理请求的完整生命周期追踪器。

    属性分为三组:
      - 请求参数: 创建时设置，不可变
      - 执行状态: 随推理进度动态更新
      - 生命周期控制: 管理异步任务和事件

    线程安全: 所有异步操作通过 asyncio.Event 和 asyncio.Task 协调。
    """

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def __init__(self, request_info: dict):
        """从 HTTP 请求体创建 ReqTracker。

        Args:
            request_info: 包含以下字段的字典:
                - request_id: 请求唯一标识
                - model_id: 模型标识 (PD 模式: <256 为 prefill, >=256 为 decode)
                - model_type: 模型类型字符串
                - input_length: prompt 的 token 数
                - expect_output_length: 期望生成的 token 数
                - start_time: 请求创建时间戳
        """
        # --- 请求参数 (不可变) ---
        self.start_time: float = request_info['start_time']
        self.request_id: str = request_info['request_id']
        self.model_id: int = request_info['model_id']
        self.model_type: str = request_info['model_type']
        self.input_length: int = request_info['input_length']
        self.expect_output_length: int = request_info['expect_output_length']

        # --- 执行状态 ---
        self.output_length: int = 0
        self.text: list = _INPUT_TABLE[:self.input_length]  # token ID 序列
        self.under_prefill: bool = True
        self.first_token_time: float = -1.0
        self.finish_time: float = -1.0

        # --- SLO 参数 ---
        self.TPOT_SLO: float = scheduler_config.TPOT
        self.TTFT_SLO: float = min(
            max(scheduler_config.TTFT_baseline, self.input_length / 512 * 0.95),
            scheduler_config.TTFT_max_threshold
        )
        self.TTFT_cold_start: bool = False  # 是否经历了冷启动（模型加载等待）

        # 额外容忍时间: 模型加载延迟、KV 迁移等待等会累加到此处
        self.tolerate_time: float = 0.0

        # --- Worker 位置 ---
        self.node_type: str = ''
        self.node_id: int = -1
        self.worker_id: int = -1

        # --- 生命周期控制 ---
        self.fire_version: int = 0                   # 发送版本号，用于取消过期的 fire_task
        self.fire_task: Optional[asyncio.Task] = None  # 当前正在执行的异步发送任务
        self.finish: bool = False                    # 是否已完成全部 token 生成
        self.detach_from_worker: bool = True         # 是否与 Worker 脱钩
        self.terminate_event: asyncio.Event = asyncio.Event()  # 终止信号（完成或驱逐）

        # --- 迁移状态 ---
        self.status: str = 'exec'                    # 'exec' | 'migrate'
        self.migrate_event: Optional[asyncio.Event] = None

        # --- 处理的 Worker 历史 (用于端到端指标) ---
        self.handled_workers: list = []

    # ------------------------------------------------------------------
    # 信息查询
    # ------------------------------------------------------------------

    def get_info(self) -> dict:
        """返回请求的摘要信息，用于日志和调试。"""
        return {
            'request_id': self.request_id,
            'fire_version': self.fire_version,
            'model_type': self.model_type,
            'model_id': self.model_id,
            'pool_type': self.node_type,
            'node_id': self.node_id,
            'worker_id': self.worker_id,
            'in': self.input_length,
            'cur_out': self.output_length,
            'exp_out': self.expect_output_length,
        }

    def total_length(self) -> int:
        """当前总 token 数 = input_length + 已生成的 output_length。"""
        return self.input_length + self.output_length

    def expect_next_token_time(self) -> float:
        """计算下一个 token 的期望到达时间 (Deadline)。

        如果请求已完成，返回无穷大（不会超时）。

        DDL = start_time + tolerate_time + TTFT_SLO + TPOT_SLO * output_length

        其中 tolerate_time 包含:
          - 模型加载延迟
          - KV 迁移等待时间
          - 调度等待补偿
        """
        if self.finish:
            return 1e18
        return (self.start_time + self.tolerate_time + self.TTFT_SLO
                + self.TPOT_SLO * self.output_length)

    def ddl_violate(self) -> bool:
        """检查当前是否已超过 DDL（即已违反 SLO）。"""
        return self.expect_next_token_time() < time.time()

    # ------------------------------------------------------------------
    # 位置与迁移
    # ------------------------------------------------------------------

    def set_location(self, node_type: str, node_id: int, worker_id: int):
        """记录处理此请求的当前 Worker 位置。

        每次请求被分配到新 Worker 时调用，历史位置会被记录到 handled_workers。
        """
        self.node_type = node_type
        self.node_id = node_id
        self.worker_id = worker_id
        if node_type != '':
            self.handled_workers.append(
                (self.output_length, f'{node_type}-{node_id}-{worker_id}')
            )

    def start_migration(self):
        """标记请求进入迁移状态。"""
        self.status = 'migrate'

    def end_migration(self):
        """标记请求迁移完成，恢复执行状态。"""
        self.status = 'exec'

    async def wait_for_migration_complete(self):
        """等待迁移完成。"""
        await self.migrate_event.wait()
        self.migrate_event = None

    # ------------------------------------------------------------------
    # Token 接收
    # ------------------------------------------------------------------

    def receive_new_token(self, token_list: list):
        """接收一个新生成的 token。

        由 Worker.fire_request_sync 在收到 vLLM 流式响应时调用。
        首个 token 到达时记录 TTFT；首个 token 后标记 prefill 阶段结束。

        Args:
            token_list: vLLM 返回的 token 列表（通常长度为 1）
        """
        assert len(token_list) == 1
        self.text.append(token_list[0])
        self.output_length += 1

        if self.output_length == 1:
            self.first_token_time = time.time()

        if self.under_prefill:
            self.under_prefill = False

    # ------------------------------------------------------------------
    # 驱逐 (Eviction)
    # ------------------------------------------------------------------

    def perform_evict(self, save_kv: bool):
        """驱逐当前请求。

        当 Worker 被抢占或需要清理时调用。
        取消当前的 fire_task，标记与 Worker 脱钩。

        Args:
            save_kv: 是否保存 KV cache（用于后续迁移到其他 Worker）。
                     False 时标记 under_prefill=True，重新执行时从 prefill 开始。
        """
        assert self.fire_task is not None
        if not save_kv:
            self.under_prefill = True
        self.fire_task.cancel()
        self.fire_task = None
        self.set_location('', -1, -1)
        self.terminate_event.set()

    # ------------------------------------------------------------------
    # 完成检查
    # ------------------------------------------------------------------

    def check_finish(self) -> bool:
        """检查请求是否已完成全部 token 生成。

        Returns:
            True 如果 output_length >= expect_output_length。
            完成时自动记录 finish_time 并设置 terminate_event。
        """
        if self.finish:
            return True
        assert self.output_length <= self.expect_output_length
        if self.output_length == self.expect_output_length:
            self.finish = True
            self.finish_time = time.time()
            self.terminate_event.set()
            return True
        return False

    # ------------------------------------------------------------------
    # 端到端指标
    # ------------------------------------------------------------------

    def get_e2e_metrics(self) -> dict:
        """计算并返回端到端性能指标。

        Returns:
            {
                'TTFT': Time To First Token (秒),
                'TPOT': Time Per Output Token (秒),
                'tolerate_time': 额外容忍时间 (秒),
                'cold_start': 是否经历冷启动,
                'handled_workers': [(output_length, 'cpu-0-1'), ...] 处理历史
            }
        """
        ttft = self.first_token_time - self.start_time
        if self.output_length > 1:
            tpot = (self.finish_time - self.first_token_time) / (self.output_length - 1)
        else:
            tpot = 0.0
        return {
            'TTFT': round(ttft, 3),
            'TPOT': round(tpot, 3),
            'tolerate_time': round(self.tolerate_time, 3),
            'cold_start': self.TTFT_cold_start,
            'handled_workers': self.handled_workers,
        }
