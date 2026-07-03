"""Worker Action 类定义 — 所有可发送给 Worker 的异步操作。

从原 worker.py 中提取，每个 Action 代表一个需要 Worker 执行的异步任务。
Action 通过 Node.worker_action_monitor 进行排队和内存预算检查后分发执行。

Action 生命周期:
  1. Worker 调用 register_*_action() → 创建 Action 实例，入队到 Node.worker_actions_register_queue
  2. Node.worker_action_monitor() 收到 Action → try_commit_and_dispatch_action()
     - 内存检查通过 → dispatch → Worker.dispatched_action_list
     - 内存检查失败 → 加入 pending_actions 列表（等待 WorkerGiveOutMemory 触发重试）
  3. Worker.digest_dispatched_actions() 依次执行每个已分发的 Action
  4. 执行完成后标记 is_performing_action = False，触发下一轮调度
"""

from abc import ABC
from typing import Optional


class WorkerActionBase(ABC):
    """所有 Worker Action 的抽象基类。"""
    pass


class WorkerKVScaleAction(WorkerActionBase):
    """KV Cache 扩缩容操作。

    用于 GPU Worker 动态调整 KV cache 的 block 数量。
    扩容：当新请求加入时，需要更多 KV cache 空间
    缩容：当请求完成/驱逐后，释放不再需要的 KV cache 空间
    """

    def __init__(self, worker_id: int, action_id: int, new_num_blocks: int):
        self.worker_id = worker_id
        self.action_id = action_id          # 版本号，用于检测 stale action
        self.new_num_blocks = new_num_blocks  # 目标 KV block 数量

    def __repr__(self) -> str:
        return (f"WorkerKVScaleAction(worker={self.worker_id}, "
                f"action_id={self.action_id}, new_blocks={self.new_num_blocks})")


class WorkerEvictRequestsAction(WorkerActionBase):
    """请求驱逐操作。

    当 Worker 被抢占或需要清理时，将其所有运行中的请求驱逐。
    驱逐时可以可选地保存 KV cache（用于后续迁移到其他 Worker）。
    """

    def __init__(self, worker_id: int, action_id: int):
        self.worker_id = worker_id
        self.action_id = action_id

    def __repr__(self) -> str:
        return (f"WorkerEvictRequestsAction(worker={self.worker_id}, "
                f"action_id={self.action_id})")


class WorkerLoadAction(WorkerActionBase):
    """模型加载操作（仅 GPU Worker）。

    GPU Worker 初始不持有模型，收到此 Action 后将模型从磁盘加载到 GPU 显存。
    加载完成后 set hold_model_remote = True。
    """

    def __init__(self, worker_id: int, action_id: int):
        self.worker_id = worker_id
        self.action_id = action_id

    def __repr__(self) -> str:
        return (f"WorkerLoadAction(worker={self.worker_id}, "
                f"action_id={self.action_id})")


class WorkerOffloadAction(WorkerActionBase):
    """模型卸载操作（仅 GPU Worker）。

    将模型从 GPU 显存卸载，释放内存空间供其他 Worker 使用。
    卸载完成后 set hold_model_remote = False，并发出 WorkerGiveOutMemory 信号。
    """

    def __init__(self, worker_id: int, action_id: int):
        self.worker_id = worker_id
        self.action_id = action_id

    def __repr__(self) -> str:
        return (f"WorkerOffloadAction(worker={self.worker_id}, "
                f"action_id={self.action_id})")


class WorkerGiveOutMemory(WorkerActionBase):
    """内存释放通知信号。

    当 Worker 完成卸载或 KV 缩容后发送此信号到 Node.worker_actions_register_queue。
    Node 收到后重新遍历 pending_actions 列表，尝试提交之前因内存不足而阻塞的 Action。

    与其它 Action 不同，此信号不进入 dispatched_action_list。
    """

    def __init__(self, worker_id: int):
        self.worker_id = worker_id

    def __repr__(self) -> str:
        return f"WorkerGiveOutMemory(worker={self.worker_id})"


class WorkerSleepAction(WorkerActionBase):
    """Worker 睡眠操作（sllm+share 专用）。

    在 ServerlessLLM + sharing 模式下，Worker 执行完一轮后需要等待
    其计费周期内的时间配额，通过此 Action 实现异步休眠。
    """

    def __init__(self, worker_id: int, action_id: int, wakeup_time: float):
        self.worker_id = worker_id
        self.action_id = action_id
        self.wakeup_time = wakeup_time      # 唤醒时间戳 (time.time() 格式)

    def __repr__(self) -> str:
        return (f"WorkerSleepAction(worker={self.worker_id}, "
                f"action_id={self.action_id}, wakeup={self.wakeup_time:.2f})")
