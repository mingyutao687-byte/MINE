# SLINFER Engine — vLLM 扩展层

本目录包含 SLINFER 对 vLLM 推理引擎的所有修改。这些修改将标准 vLLM
转变为一个支持 GPU+CPU 异构调度、KV cache 迁移和配额控制推理的分布式系统。

## 修改概览

SLINFER 修改了 vLLM 的以下 22 个文件（基于 v0.5.x 分支）:

### 新增文件 (2 个)

| 文件 | 说明 |
|------|------|
| `vllm/core/kv_manager.py` | 全局 KV cache 管理器（单例），管理跨 Worker 的 KV 迁移 |
| `vllm/core/kv_transfer.py` | Gloo+TCP 底层 KV tensor 传输（sender/receiver/manager） |

### 引擎层修改 (2 个文件)

| 文件 | 修改内容 |
|------|----------|
| `vllm/engine/async_llm_engine.py` | +200 行。新增: `register_worker`, `clear_worker`, `load_model`, `offload_model`, `scale_kv_cache_chxu`, `perform_kv_offload_chxu`, `check_and_consume_quota`, `worker_go_idle`, `set_traffic_light`, `run_engine_loop` 配额控制 |
| `vllm/engine/llm_engine.py` | +3 行。`need_kv_cache_restore_chxu()` 调用 |
| `vllm/engine/arg_utils.py` | +1 行。添加 `serverless_llm` loader 选项 |

### 调度器修改 (2 个文件)

| 文件 | 修改内容 |
|------|----------|
| `vllm/core/scheduler.py` | +55 行。新增: `scale_logical_kv_cache_chxu`, `restore_seq_group_for_continue_decode_chxu`, `get_requests_partial_metadata_chxu`, `abort_all_seqs_chxu`, `get_all_seqs_chxu` |
| `vllm/core/block_manager_v1.py` | +35 行。`UncachedBlockAllocator.scale_logical_kv_cache_chxu`, `BlockSpaceManagerV1.scale_logical_kv_cache_chxu` |

### Worker 层修改 (5 个文件)

| 文件 | 修改内容 |
|------|----------|
| `vllm/worker/worker_base.py` | +90 行。抽象方法: `scale_physical_kv_cache_chxu`, `save_kv_cache_chxu`, `restore_kv_cache_chxu` + 实现 |
| `vllm/worker/worker.py` | +20 行。`load_model`, `offload_model`, `scale_physical_kv_cache_chxu`, `save_serverless_llm_state`, `NO_MODEL_LOADING_AT_START` 检查 |
| `vllm/worker/model_runner.py` | +15 行。模型 `offload_model` (del + gc + cuda.empty_cache)，`save_serverless_llm_state` |
| `vllm/worker/cache_engine.py` | +23 行。`scale_kv_cache` 动态张量 resize |
| `vllm/worker/openvino_worker.py` | +90 行。`save_kv_cache_chxu`, `restore_kv_cache_chxu` (已移除，仅 CPU Worker 使用) |

### Executor 层修改 (3 个文件)

| 文件 | 修改内容 |
|------|----------|
| `vllm/executor/executor_base.py` | +18 行。抽象方法: `load_model`, `offload_model`, `scale_physical_kv_cache_chxu`, `save_kv_cache_chxu` |
| `vllm/executor/gpu_executor.py` | +30 行。上述抽象方法的实现 + `NO_MODEL_LOADING_AT_START` 检查 |
| `vllm/executor/distributed_gpu_executor.py` | +10 行。`save_serverless_llm_state` |

### API Server 修改 (1 个文件)

| 文件 | 修改内容 |
|------|----------|
| `vllm/entrypoints/openai/api_server.py` | +100 行。9 个自定义端点（见下文） |

### 数据模型修改 (1 个文件)

| 文件 | 修改内容 |
|------|----------|
| `vllm/sequence.py` | +30 行。`kv_cache_need_restore`, `prompt_hash`, KV restore 相关方法 |

### 配置修改 (2 个文件)

| 文件 | 修改内容 |
|------|----------|
| `vllm/config.py` | +2 行。`LoadFormat.SERVERLESS_LLM` 枚举值 |
| `vllm/envs.py` | +2 行。`NO_MODEL_LOADING_AT_START` 环境变量 |

### 模型加载器修改 (1 个文件)

| 文件 | 修改内容 |
|------|----------|
| `vllm/model_executor/model_loader/loader.py` | +30 行。`ServerlessLLMLoader` 类 |

## 自定义 API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/register_worker` | POST | 注册 Worker 信息 |
| `/clear_worker` | POST | 清除 Worker 状态 |
| `/load_model` | POST | 加载模型到 GPU |
| `/offload_model` | POST | 卸载模型 |
| `/kv_info` | POST | 获取 KV cache 信息 |
| `/kv_scale` | POST | KV cache 扩缩容 |
| `/kv_send` | POST | 发送 KV cache |
| `/kv_receive` | POST | 接收 KV cache |
| `/evict_requests` | POST | 驱逐请求 |
| `/set_traffic_light` | POST | 流量控制 |

## 已移除的组件（不兼容 A10/无 AMX 环境）

以下 vLLM 组件已移除:
- **Worker**: neuron_worker, openvino_worker, tpu_worker, xpu_worker (及对应 model_runner)
- **Executor**: neuron_executor, openvino_executor, tpu_executor, xpu_executor, ray_xpu_executor
- **Attention**: openvino 后端, pallas (TPU) 后端, ipex_attn (XPU) 后端
- **Model Loader**: neuron 加载器, openvino 加载器

## A10 GPU 特殊注意事项

1. **显存限制**: A10 24GB vs A100 80GB。单个 A10 只能放 7B 模型 + KV cache。
   13B 模型需要跨 GPU 张量并行或 CPU offload。
2. **KV cache 块大小**: 建议 block_size=16（与 A100 相同）。
3. **无 AMX**: CPU 推理使用 PyTorch CPU 后端（非 OpenVINO），prefill 速度较慢。
4. **模型选择**: 建议主力使用 3B/7B 模型。8B 使用 GPTQ int4 量化可放入单 A10。
