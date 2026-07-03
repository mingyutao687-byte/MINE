"""A10 GPU 专用资源池配置 — 4×A10 (24GB) + 256GB CPU。

与原始 A100 配置的区别:
  - GPU 内存: 22GB 可用 (24GB - 2GB 系统保留)，而非 A100 的 78GB
  - 不支持 13B 模型在单 GPU 上运行（需张量并行跨 GPU）
  - CPU KV cache: 256GB 系统内存，但无 AMX 加速

建议的部署模式:
  1. GPU-only: 4×A10 各运行一个 7B 或 3B 模型实例
  2. GPU+CPU: GPU 做 decode，CPU 做 prefill (PD 分离模式)
  3. Multi-GPU: 2×A10 张量并行运行 13B 模型

使用方式:
  from Mine.config.a10_pools import POOLS_3B_4GPU, POOLS_7B_4GPU
"""

from Mine.config.settings import models_info_template as _models


# ============================================================================
# A10 GPU 模型信息 (调整显存限制)
# ============================================================================

A10_MODELS = {
    'llama-3.2-3b': {
        'model_type': 'llama-3.2-3b',
        'model_memory_GB': 6.1,              # 3B 模型约 6GB
        'per_token_kv_memory_KB': 112,
        'block_size': {'cpu': 32, 'gpu': 16},
        'cpu_kv_gb': 16,                      # CPU KV 预分配
    },
    'llama-2-7b': {
        'model_type': 'llama-2-7b',
        'model_memory_GB': 12.6,
        'per_token_kv_memory_KB': 512,
        'block_size': {'cpu': 32, 'gpu': 16},
        'cpu_kv_gb': 32,
    },
    'llama-3.1-8b': {
        'model_type': 'llama-3.1-8b',
        'model_memory_GB': 15.0,              # int4 量化约 8GB，fp16 需要 ~16GB
        'per_token_kv_memory_KB': 128,
        'block_size': {'cpu': 32, 'gpu': 16},
        'cpu_kv_gb': 32,
    },
    # 注意: 13B 模型无法放入单个 A10 (24GB)，需张量并行或 CPU-only
    'llama-2-13b': {
        'model_type': 'llama-2-13b',
        'model_memory_GB': 24.3,              # 超出单 A10 容量!
        'per_token_kv_memory_KB': 800,
        'block_size': {'cpu': 32, 'gpu': 16},
        'cpu_kv_gb': 32,
    },
}


# ============================================================================
# 3B 模型 — 4 GPU + 0 CPU 配置
# ============================================================================

POOLS_3B_4GPU_0CPU = {
    'gpu': {
        0: {
            'node_memory_capacity_GB': 22,    # A10: 24GB - 2GB overhead
            'node_ip': '127.0.0.1',
            'gateway_ip': '127.0.0.1',
            'dist_scheduler': False,
            'base_port': 8000,
            'node_label': 'gpu',
            'workers': {
                0: A10_MODELS['llama-3.2-3b'],
                1: A10_MODELS['llama-3.2-3b'],
            },
        },
        1: {
            'node_memory_capacity_GB': 22,
            'node_ip': '127.0.0.1',
            'gateway_ip': '127.0.0.1',
            'dist_scheduler': False,
            'base_port': 8100,
            'node_label': 'gpu',
            'workers': {
                0: A10_MODELS['llama-3.2-3b'],
                1: A10_MODELS['llama-3.2-3b'],
            },
        },
        2: {
            'node_memory_capacity_GB': 22,
            'node_ip': '127.0.0.1',
            'gateway_ip': '127.0.0.1',
            'dist_scheduler': False,
            'base_port': 8200,
            'node_label': 'gpu',
            'workers': {
                0: A10_MODELS['llama-3.2-3b'],
                1: A10_MODELS['llama-3.2-3b'],
            },
        },
        3: {
            'node_memory_capacity_GB': 22,
            'node_ip': '127.0.0.1',
            'gateway_ip': '127.0.0.1',
            'dist_scheduler': False,
            'base_port': 8300,
            'node_label': 'gpu',
            'workers': {
                0: A10_MODELS['llama-3.2-3b'],
                1: A10_MODELS['llama-3.2-3b'],
            },
        },
    },
    'cpu': {
        # 无 CPU 节点 — GPU-only 模式
    },
}


# ============================================================================
# 7B 模型 — 4 GPU + 0 CPU 配置 (每个 GPU 1 个 Worker)
# ============================================================================

POOLS_7B_4GPU_0CPU = {
    'gpu': {
        i: {
            'node_memory_capacity_GB': 22,
            'node_ip': '127.0.0.1',
            'gateway_ip': '127.0.0.1',
            'dist_scheduler': False,
            'base_port': 8000 + i * 100,
            'node_label': 'gpu',
            'workers': {
                0: A10_MODELS['llama-2-7b'],
                # 单个 A10 放 7B 模型 (12.6GB) + KV cache 约 10GB
                # 两个 Worker 会超出 22GB，故每 GPU 仅 1 个
            },
        } for i in range(4)
    },
    'cpu': {},
}


# ============================================================================
# 3B 模型 — 1 GPU + 1 CPU 配置 (调试用)
# ============================================================================

POOLS_3B_1GPU_1CPU_DEBUG = {
    'gpu': {
        0: {
            'node_memory_capacity_GB': 22,
            'node_ip': '127.0.0.1',
            'gateway_ip': '127.0.0.1',
            'dist_scheduler': False,
            'base_port': 8000,
            'node_label': 'gpu',
            'workers': {
                0: A10_MODELS['llama-3.2-3b'],
            },
        },
    },
    'cpu': {
        0: {
            'node_memory_capacity_GB': 240,   # 256GB - 16GB overhead
            'node_ip': '127.0.0.1',
            'gateway_ip': '127.0.0.1',
            'dist_scheduler': True,            # CPU 节点使用分布式调度器
            'base_port': 8400,
            'node_label': 'cpu',
            'workers': {
                0: A10_MODELS['llama-3.2-3b'],
                1: A10_MODELS['llama-3.2-3b'],
                2: A10_MODELS['llama-3.2-3b'],
                3: A10_MODELS['llama-3.2-3b'],
            },
        },
    },
}
