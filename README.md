# SLINFER — GPU+CPU 异构 LLM 推理调度系统

针对 **4× NVIDIA A10 (24GB) + 256GB 主存 + Intel CPU (无 AMX)** 环境
重构的完整推理系统。

## 目录结构

```
Mine/
├── README.md                    # 本文档 — 系统总览
│
├── scheduler/                   # ★ 调度器 (原 SLINFER_core，已重构)
│   ├── config/settings.py       #   全局调度配置 (SchedulerConfig dataclass)
│   ├── models/                  #   数据模型
│   │   ├── enums.py             #     WorkerHangingReleaseType 枚举
│   │   ├── actions.py           #     7 个 WorkerAction 类
│   │   ├── request_tracker.py   #     ReqTracker 请求生命周期
│   │   └── schemas.py           #     WorkerConfig/NodeConfig 等 dataclass
│   ├── core/                    #   核心调度逻辑
│   │   ├── worker.py            #     Worker 类 (vLLM 实例代理 + Action 系统)
│   │   ├── node.py              #     Node 类 + MiniNode/MiniWorker
│   │   ├── pool.py              #     Pool + PoolManager 全局资源管理
│   │   └── power_estimator.py   #     性能估算 (profiling 数据插值)
│   ├── api/                     #   HTTP API
│   │   ├── gateway.py           #     主调度网关 (FastAPI, 端口 7000)
│   │   └── dist_gateway.py      #     分布式调度网关 (CPU 节点本地)
│   ├── cli/                     #   命令行工具
│   │   ├── backend_starter.py   #     启动单个 vLLM Worker
│   │   └── batch_starter.py     #     批量启动 vLLM Worker
│   └── config_template/         #   Pool/Model 配置模板 (10 个变体)
│
├── engine/                      # ★ vLLM 推理引擎扩展 (SLINFER 修改层)
│   ├── README.md                #   修改清单和 API 文档
│   ├── core/
│   │   ├── kv_manager.py        #   全局 KV Cache 管理器 (单例)
│   │   └── kv_transfer.py       #   Gloo+TCP KV tensor 传输
│   └── api/
│       └── slinfer_endpoints.py #   自定义 API 端点 (FastAPI Router)
│
├── store/                       # ★ ServerlessLLM 模型存储集成
│   └── README.md                #   架构说明和配置指南
│
├── config/
│   ├── __init__.py              #   导出 A10 配置
│   └── a10_pools.py             #   A10 专用资源池配置 (3B/7B)
│
└── tools/                       # 测试与工具
    ├── test/                    #   测试脚本
    ├── trace/                   #   工作负载 trace 数据
    └── draw/                    #   可视化脚本
```

## 架构概览

```
                          ┌──────────────────────┐
                          │      Client          │
                          │  POST /v1/completions│
                          └──────────┬───────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────┐
│                    SLINFER Scheduler (Mine/scheduler/)          │
│                                                                │
│  gateway.py (:7000)              dist_gateway.py (每CPU节点)    │
│    │                               │                           │
│    └── PoolManager                 └── MiniNode (DDL-based)    │
│         ├── gpu_pool: Pool                                     │
│         │   └── Node[] ── Worker[] ── HTTP ──▶ vLLM GPU       │
│         └── cpu_pool: Pool                                     │
│             └── Node[] ── Worker[] ── HTTP ──▶ vLLM CPU       │
│                                                                │
│  调度策略: DDL-based → Batch-aware → Preemption → New Worker   │
│  检查维度: Memory budget → Decode time-slice → Prefill DDL     │
└────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────┐
│              vLLM Inference Engine (upstream + patches)         │
│                                                                │
│  SLINFER patches (Mine/engine/):                               │
│    ├── kv_manager.py       全局 KV cache 管理器                 │
│    ├── kv_transfer.py      Gloo+TCP KV 传输                    │
│    └── slinfer_endpoints.py  自定义 API 端点                    │
│                                                                │
│  vLLM upstream (installed as dependency):                      │
│    ├── GPU Worker (CUDA, FlashAttention)                       │
│    ├── CPU Worker (PyTorch CPU, no OpenVINO/AMX)               │
│    ├── PagedAttention KV cache                                 │
│    └── Continuous batching                                     │
└────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────┐
│           ServerlessLLM Store (Mine/store/ + upstream)          │
│                                                                │
│  sllm-store-server (C++ gRPC):                                 │
│    ├── Pinned memory pool (CPU)                                │
│    ├── CUDA memory pool (GPU)                                  │
│    └── DMA disk→CPU→GPU 快速加载                               │
└────────────────────────────────────────────────────────────────┘
```

## 快速开始

### 1. 安装依赖

```bash
# vLLM (CUDA 版本，不包含 OpenVINO/TPU/Neuron)
pip install vllm==0.5.0.post1

# ServerlessLLM Store (C++ CUDA)
cd ServerlessLLM_modify/sllm_store && pip install -e .

# Python 依赖
pip install fastapi uvicorn aiohttp ray requests websockets
```

### 2. 配置

```python
# 使用 A10 4-GPU 3B 模型配置
from Mine.config.a10_pools import POOLS_3B_4GPU_0CPU as pools_config

# 或使用 7B 4-GPU 配置
from Mine.config.a10_pools import POOLS_7B_4GPU_0CPU as pools_config
```

### 3. 启动

```bash
# 1. 启动 vLLM Worker (每个 GPU)
python -m Mine.cli.backend_starter --device gpu --gpu 0 --port 8000 \
    --model /path/to/model --block_num 256 --use_sllm False --no_load True

# 2. 启动调度网关
python -m Mine.api.gateway

# 3. 发送测试请求
python -m Mine.tools.test.test_3B_ultra_lite
```

## 已移除的组件

以下原始 SLINFER 组件已从 Mine/ 中移除（不兼容 A10/无 AMX 环境）:

### vLLM 移除
- `vllm/worker/openvino_worker.py` — 需要 Intel AMX 指令集
- `vllm/worker/openvino_model_runner.py` — 需要 OpenVINO 运行时
- `vllm/worker/tpu_worker.py` — 需要 Google TPU 硬件
- `vllm/worker/tpu_model_runner.py` — 需要 torch_xla
- `vllm/worker/neuron_worker.py` — 需要 AWS Trainium/Inferentia
- `vllm/worker/neuron_model_runner.py` — 需要 transformers_neuronx
- `vllm/worker/xpu_worker.py` — 需要 Intel GPU (XPU)
- `vllm/worker/xpu_model_runner.py` — 需要 Intel IPEX
- `vllm/executor/neuron_executor.py`
- `vllm/executor/openvino_executor.py`
- `vllm/executor/tpu_executor.py`
- `vllm/executor/xpu_executor.py`
- `vllm/executor/ray_xpu_executor.py`
- `vllm/attention/backends/openvino.py`
- `vllm/attention/backends/pallas.py` (TPU)
- `vllm/attention/backends/ipex_attn.py` (XPU)
- `vllm/model_executor/model_loader/neuron.py`
- `vllm/model_executor/model_loader/openvino.py`

### 保留的硬件后端 (NVIDIA A10 兼容)
- ✅ GPU Worker (CUDA, FlashAttention) — 主推理后端
- ✅ CPU Worker (PyTorch CPU, GLOO) — CPU 推理/Prefill
- ✅ FlashInfer backend — NVIDIA GPU 优化
- ✅ xFormers backend — NVIDIA GPU 备选
- ✅ ROCm backend — AMD GPU (保留，无依赖冲突)
- ✅ Torch SDPA backend — CPU fallback

## 与原始 SLINFER 的区别

| 方面 | 原始 SLINFER | Mine/ 重构版 |
|------|-------------|-------------|
| 代码组织 | 6 个平铺大文件 | ~35 个模块化小文件 |
| 类型标注 | 无 | 全量类型标注 |
| 文档 | 无 | 中英文 docstring |
| vLLM 后端 | 全部后端 | 仅 NVIDIA CUDA + CPU |
| 目标 GPU | A100 80GB | A10 24GB |
| CPU 推理 | OpenVINO (AMX) | PyTorch CPU (通用) |
| 测试环境 | 4×A100 + 4×Xeon | 4×A10 + 1×Xeon |
| 池配置 | 每 GPU 2 worker (13B/7B) | 每 GPU 1-2 worker (3B/7B) |
| 最大模型 | 13B 单 GPU | 7B 单 GPU, 13B 需 TP |

## 技术细节

### 配额控制 (Quota-based Scheduling)

vLLM Worker 不会自由运行，而是通过配额系统由 SLINFER Scheduler 精确控制:

```
Worker 空闲 → wait_for_new_requests()
  → check_and_consume_quota(was_being_scheduled=False)
    → HTTP POST /ask_for_quota → Gateway
      → node.schedule() 选中的 Worker 获得配额
  → 执行 inference step
  → check_and_consume_quota(was_being_scheduled=True)
  → 继续或 worker_go_idle()
```

### KV Cache 迁移

```
源 Worker (CPU)                    目标 Worker (GPU)
  save_kv_cache_chxu()              restore_kv_cache_chxu()
    ↓                                 ↑
  kv_manager.add_request()    kv_manager.pop_request()
    ↓                                 ↑
  kv_manager.send_kv() ──Gloo──→ kv_manager (receive_kv)
```

### PD 分离 (Prefill-Decode Disaggregation)

```
启用 enable_PD=True:
  1. 请求到达 → model_id < 256 → CPU/GPU 做 Prefill (1 token)
  2. transform_to_decode_only_request() → model_id += 256
  3. KV cache 迁移到 GPU Decode 专用 Worker
  4. 继续 Decode 生成剩余 token
```
