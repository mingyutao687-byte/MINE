# SLINFER Store — ServerlessLLM 模型存储集成层

本目录包含 SLINFER 对 ServerlessLLM 的集成代码。
ServerlessLLM 提供快速的模型加载/卸载能力（基于 C++ CUDA 内存池和 gRPC）。

## 架构

```
┌─────────────────────────────────────────────────────────┐
│                 SLINFER Scheduler                        │
│  (Mine/scheduler/)                                      │
│  Worker.register_load_action() → HTTP /load_model       │
│                                   → vLLM loads from     │
│                                     serverless_llm fmt  │
└──────────────────────────────┬──────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────┐
│              ServerlessLLM Store                         │
│                                                         │
│  sllm/serve/controller.py                               │
│    └── SllmController (Ray Actor)                       │
│        ├── StoreManager (Ray Actor)                     │
│        │   └── SllmLocalStore[] (每个节点)              │
│        │       ├── sllm-store-server (C++ gRPC)         │
│        │       └── pinned memory pool                   │
│        ├── Scheduler (FcfsScheduler|StorageAware)       │
│        └── Router (RoundRobinRouter|MigrationRouter)     │
│                                                         │
│  sllm_store/ (C++/CUDA)                                 │
│    ├── CheckpointStore — 磁盘→CPU→GPU 快速加载          │
│    ├── CUDA Memory Pool — GPU 显存管理                  │
│    └── Pinned Memory Pool — DMA 加速                    │
└─────────────────────────────────────────────────────────┘
```

## 关键文件

| 文件 | 说明 |
|------|------|
| `sllm/serve/controller.py` | 中心控制器: 管理模型注册和实例生命周期 |
| `sllm/serve/store_manager.py` | 存储管理器: 集群初始化、模型下载、内存池管理 |
| `sllm/serve/backends/vllm_backend.py` | vLLM 后端: 使用 "serverless_llm" load_format |
| `sllm/serve/backends/transformers_backend.py` | Transformers 后端: 直接使用 HuggingFace 模型 |
| `sllm/serve/routers/roundrobin_router.py` | 轮询路由器: 负载均衡 + 自动扩缩容 |
| `sllm/serve/routers/migration_router.py` | 迁移路由器: 支持实时模型迁移 |
| `sllm/serve/schedulers/fcfs_scheduler.py` | FCFS 调度器: 先来先服务 GPU 分配 |
| `sllm/serve/schedulers/storage_aware_scheduler.py` | 存储感知调度器: 优先使用本地已有模型的节点 |
| `sllm/cli/` | CLI 工具: deploy, delete, generate, encode, replay, update |
| `sllm_store/` | C++ 存储后端: gRPC server, CUDA 内存, pinned memory |

## A10 注意事项

1. **A10 兼容**: Ampere 架构 (SM 8.6)，所有 CUDA 代码兼容
2. **PCIe 带宽**: A10 通常为 PCIe Gen4 x16，比 A100 SXM 的 NVLink 慢，
   模型加载时间会更长
3. **内存池大小**: 建议 `--mem_pool_size` 设为 80-160GB (256GB 总内存的 30-60%)
4. **chunk_size**: 建议 64MB (与原始 A100 配置相同)
