"""枚举类型定义 — 调度系统中使用的所有枚举。

从原 worker.py 和 dist_gateway.py 中提取，便于全局引用。
"""

from enum import Enum, auto


class WorkerHangingReleaseType(Enum):
    """Worker 挂起释放类型 — 当 Worker 等待调度时收到的事件类型。

    being_scheduled: Worker 被调度器选中，获得执行配额
    finished_action: Worker 完成了一个异步 Action（加载/卸载/KV 扩缩容等）
    """
    being_scheduled = auto()
    finished_action = auto()
