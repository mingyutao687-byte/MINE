"""全局调度配置。

配置分为三类：
  1. 导入的模板配置（模型路径、模型信息、资源池信息）
  2. 运行时可通过 /set_config API 动态修改的配置
  3. 静态常量（SLO 参数、调度策略参数等）

使用方式：
  from Mine.config.settings import scheduler_config
  print(scheduler_config.keep_alive_time)

  # 模块级变量也保留，兼容原代码中的 import config 用法
"""

from dataclasses import dataclass, field
from typing import Literal, Optional

# 保留原有的 config_template 导入路径
import sys
import os

# 确保 config_template 目录在 Python path 中
_template_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              'config_template')
if _template_dir not in sys.path:
    sys.path.insert(0, _template_dir)

from models_path_template import models_path_template  # type: ignore
from models_info_template import models_info_template  # type: ignore
from pools_info_template import pools_info_template    # type: ignore


# ============================================================================
# 静态常量 (不可通过 API 修改)
# ============================================================================

# 模型文件路径映射: model_type → {'cpu': path, 'gpu': path}
models_path = models_path_template

# 模型元信息: model_type → {model_memory_GB, per_token_kv_memory_KB, ...}
models_info = models_info_template

# 资源池拓扑配置: {'gpu': {node_id: {...}}, 'cpu': {node_id: {...}}}
pools_config = pools_info_template


# ============================================================================
# SLO (Service Level Objective) 参数
# ============================================================================

# TTFT (Time To First Token) SLO 基线 (秒)
# 实际 SLO = max(TTFT_baseline, input_length / 512 * 0.95)，上限为 TTFT_max_threshold
TTFT_baseline: float = 0.5 * 0.95          # 0.475 秒
TTFT_max_threshold: float = 8 * 0.95        # 7.6 秒

# TPOT (Time Per Output Token) SLO (秒)
# 每个 decode token 的目标生成时间
TPOT: float = 0.25 * 0.95                   # 0.2375 秒


# ============================================================================
# 调度策略配置
# ============================================================================

# DDL (Deadline) 调度的安全阈值 (秒)
# 当所有 Worker 的 DDL 都大于此值时，切换为 batch-aware 调度（优先调 batch 最大的 Worker）
DDL_SAFE_THRESHOLD: float = 5.0


# ============================================================================
# 可通过 /set_config API 动态修改的配置
# ============================================================================

@dataclass
class SchedulerConfig:
    """调度器运行时配置 — 可通过 HTTP POST /set_config 动态修改。

    每个字段对应一个可配置项，修改后立即生效。
    """

    # --- 系统模式 ---
    # 'sota': MINE 原生调度模式
    # 'serverlessllm': 兼容 ServerlessLLM 的调度模式
    system: Literal['serverlessllm', 'sota'] = 'sota'

    # --- 资源管理 ---
    # Worker 空闲后保持存活的时间 (秒)，超时后自动清理
    # CPU Worker 始终为 0（立即清理），GPU Worker 默认 1 秒
    keep_alive_time: float = 1.0

    # 资源池调度优先级: 'cpu' 优先或 'gpu' 优先
    pool_priority: Literal['cpu', 'gpu'] = 'cpu'

    # 是否启用 CPU 节点参与推理
    enable_cpu: bool = True

    # --- 碎片整理与负载均衡 ---
    # True: 尽量将请求集中到已有 Worker（减少模型加载开销）
    # False: 尽量均匀分散到不同 Worker（负载均衡）
    enable_defragmentation: bool = True

    # 是否允许同一节点上多个 Worker 加载模型（多租户共享）
    enable_sharing: bool = True

    # 是否允许通过抢占已有 Worker 来接受新请求
    enable_preempt: bool = True

    # --- KV Cache 管理 ---
    # KV cache 扩容水位 (0~1)
    # 例: 当前使用 100 blocks, watermark=0.25 → 推荐扩容到 125 blocks
    kv_scale_watermark: float = 0.25

    # 每个实例保证的最小 token 容量
    minimal_tokens_per_instance: int = 4096

    # --- ServerlessLLM 集成 ---
    # 是否启用 sllm 的多租户模型共享
    sllm_enable_sharing: bool = False

    # sllm 共享模式下的最大共享数
    sllm_max_shares: int = 2

    # --- 抢占指标选择 ---
    # decode 抢占排序指标: 'batch' (按 running_requests 数量) | 'compute' (按 time_slice)
    decode_preempt_metric: Literal['batch', 'compute'] = 'batch'

    # 内存抢占排序指标: 'batch' | 'memory' (按实际内存占用)
    memory_preempt_metric: Literal['batch', 'memory'] = 'batch'

    # --- DDL 调度 ---
    # DDL-based 调度的详细配置
    ddl_based_schedule: dict = field(default_factory=lambda: {
        'enable_batch_aware': False,   # 当 DDL 都充裕时，是否切换到 batch 优先
        'safe_ddl_threshold': 5,       # DDL 安全阈值 (秒)
    })

    # --- 日志 ---
    # 是否启用详细的性能日志（会影响性能）
    enable_detailed_logging: bool = False


# 全局单例 — 模块加载时创建，运行时通过 /set_config 修改
scheduler_config = SchedulerConfig()


# ============================================================================
# 兼容旧代码的模块级变量导出
# 原代码中 import config 后直接使用 config.keep_alive_time 等
# ============================================================================

# 这些属性代理到 scheduler_config 实例
def __getattr__(name: str):
    """模块级属性访问代理 — 兼容原 config.py 的直接变量访问模式。

    例如: config.keep_alive_time → scheduler_config.keep_alive_time
    """
    if name == 'scheduler_config':
        return scheduler_config
    if hasattr(scheduler_config, name):
        return getattr(scheduler_config, name)
    # 静态常量 / 模板配置
    if name in ('models_path', 'models_info', 'pools_config',
                'TTFT_baseline', 'TTFT_max_threshold', 'TPOT',
                'DDL_SAFE_THRESHOLD'):
        return globals()[name]
    raise AttributeError(f"module 'Mine.config.settings' has no attribute '{name}'")
