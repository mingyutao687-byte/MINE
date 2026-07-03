# SLINFER

GPU+CPU 异构 LLM 推理调度系统，支持 DDL 感知调度、KV cache 跨节点迁移、配额控制推理。

适用于 **4× NVIDIA A10 (24GB) + 256GB 主存 + Intel CPU** 环境。

## 架构

```
Client  →  Gateway (:7000)  →  PoolManager  →  Worker  →  vLLM Engine
                                      ├── gpu_pool (CUDA)
                                      └── cpu_pool (PyTorch CPU)
```

调度流程：新请求到达 → 尝试加入已有 Worker → 尝试抢占 → 分配新 Worker → KV 扩容 → 推理执行。每个 Worker 的运行步数由调度器通过配额精确控制。

## 目录

```
├── api/                HTTP 网关 (gateway :7000, dist_gateway)
├── core/               调度核心 (worker, node, pool, power_estimator)
├── models/             数据模型 (enums, actions, request_tracker, schemas)
├── config/             运行时配置 + A10 资源池配置
├── cli/                命令行工具 (worker 启动器)
├── config_template/    Pool/Model 配置模板
│
├── engine/             推理引擎
│   ├── vllm/           完整 vLLM (CUDA GPU + PyTorch CPU)
│   ├── core/           KV cache 管理器 + Gloo 传输层
│   └── api/            SLINFER 自定义端点
│
├── store/              ServerlessLLM 模型存储
│   ├── sllm/           Python 服务层 (controller, backends, routers)
│   └── sllm_store/     C++/CUDA gRPC 后端 (pinned memory, DMA)
│
├── tools/              测试脚本 + 工作负载 trace + 可视化
├── run_gateway.py      启动调度网关
├── run_worker.py       启动单个 vLLM Worker
├── run_cluster.py      一键批量启动集群 (多 GPU 多 Worker)
└── RUN_GUIDE.md        详细运行指南
```

## 快速开始

```bash
# 1. 安装
cd engine/vllm && pip install -e .
cd store/sllm_store && pip install -e .
pip install fastapi uvicorn aiohttp ray

# 2. 配置 (编辑 config/a10_pools.py 选择拓扑)

# 3. 一键启动集群 (4 GPU, 每 GPU 2 worker, 含 Gateway)
export PROJECT_BASE=/path/to/SLINFER
python run_cluster.py --mode local --model llama-3.2-3b --workers-per-gpu 2
```

```bash
# 单 Worker 启动
python run_worker.py --device gpu --gpu 0 --port 8000 --model $PROJECT_BASE/gpu_models/Llama-3.2-3B-Instruct

# 单独启动网关
python run_gateway.py          # → http://0.0.0.0:7000
```

## 核心特性

### DDL 感知调度
每个 token 有 deadline（基于 SLO），调度器优先调度最紧急的 Worker。当所有 Worker DDL 充裕时切换到 batch-aware 模式最大化吞吐。

### KV Cache 跨节点迁移
通过 Gloo + TCP 实现 CPU↔GPU 之间的 KV cache 传输，支持请求抢占和负载均衡。

### 配额控制推理
vLLM Worker 不自由运行，每步推理需向调度器请求配额。Worker 空闲时自动释放配额，调度器重新分配给其他 Worker。

### Worker Action 系统
模型加载/卸载、KV 扩缩容、请求驱逐等操作通过版本化 Action 队列管理，避免竞态条件。

### ServerlessLLM 集成
C++/CUDA gRPC 存储后端，支持 pinned memory + DMA 的快速模型加载（磁盘→CPU→GPU）。

## 配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `system` | `sota` | 调度模式: `sota` / `serverlessllm` |
| `keep_alive_time` | 1s | Worker 空闲后保留时间 |
| `pool_priority` | `cpu` | 优先调度 CPU 还是 GPU |
| `enable_defragmentation` | True | 负载集中 vs 负载均衡 |
| `enable_preempt` | True | 允许抢占已有 Worker |
| `kv_scale_watermark` | 0.25 | KV cache 扩容水位 |
| `enable_cpu` | True | 启用 CPU 节点 |
| `ddl_based_schedule` | `{batch_aware: False, threshold: 5}` | DDL 调度策略 |

运行时可通过 `POST /set_config` 动态修改。

## 已移除

- OpenVINO (需要 Intel AMX)
- TPU / XPU / Neuron (需要专用硬件)
- Prefill-Decode 分离 (A10 环境无加速收益)
