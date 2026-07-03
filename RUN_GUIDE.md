# SLINFER Mine/ 运行指南

基于 **4× NVIDIA A10 (24GB) + 256GB 主存 + Intel CPU (无 AMX)** 环境。
代码位于 `SLINFER_core/Mine/` 目录。

## 与原版 SLINFER 的目录对应关系

| 原版路径 | Mine/ 中的路径 | 说明 |
|----------|---------------|------|
| `SLINFER_core/scheduler/gateway.py` | `Mine/api/gateway.py` | 调度网关 |
| `SLINFER_core/scheduler/*.py` | `Mine/core/*.py` + `Mine/models/*.py` | 调度核心 (已重构拆分) |
| `vLLM_modify/vllm/` | `Mine/engine/vllm/` | vLLM 推理引擎 (完整 fork) |
| `vLLM_modify/csrc/` | `Mine/engine/vllm/csrc/` | CUDA 内核 |
| `ServerlessLLM_modify/sllm_store/` | `Mine/store/sllm_store/` | 模型快速加载 C++ 后端 |
| `ServerlessLLM_modify/sllm/` | `Mine/store/sllm/` | ServerlessLLM Python 服务层 |
| `SLINFER_core/tools/` | `Mine/tools/` | 测试脚本和 trace 数据 |

## 0. 环境准备

```bash
# 设置项目根目录
export PROJECT_BASE=/ABSOLUTE_PATH/TO/SLINFER
export MINE_BASE=$PROJECT_BASE/SLINFER_core

# 创建 Conda 环境
conda create -n slinfer python=3.11
conda activate slinfer

# GPU 机器: 确认 CUDA
nvcc --version
```

## 1. 安装 ServerlessLLM Store (C++/CUDA)

```bash
cd $MINE_BASE/Mine/store/sllm_store
rm -rf build
pip install .
```

验证:
```bash
pip list | grep serverless-llm-store
```

## 2. 安装 vLLM (SLINFER 修改版)

```bash
cd $MINE_BASE/Mine/engine/vllm
pip install -e .
```

验证:
```bash
python -c "from vllm.entrypoints.openai.api_server import app; print('OK')"
```

## 3. 安装 Python 依赖

```bash
pip install fastapi uvicorn aiohttp ray requests websockets websocket-client
```

## 4. 导出模型

```bash
cd $PROJECT_BASE/huggingface_models
bash export_gpu_models.sh   # GPU 模型
# A10 环境下不需要 CPU OpenVINO 模型 (export_cpu_models.sh)
```

## 5. 配置 Pool (资源池)

编辑 `Mine/config/a10_pools.py` 选择你的部署拓扑:

```python
# 3B 模型 × 4 GPU (每 GPU 2 worker)
from Mine.config.a10_pools import POOLS_3B_4GPU_0CPU as MY_POOLS

# 或 7B 模型 × 4 GPU (每 GPU 1 worker)
from Mine.config.a10_pools import POOLS_7B_4GPU_0CPU as MY_POOLS
```

然后在 `Mine/config/settings.py` 中修改:
```python
pools_config = MY_POOLS
```

## 6. 启动

### 6.1 启动 vLLM Worker (每个 GPU 启动一个或多个)

```bash
cd $MINE_BASE

# GPU 0: 启动 2 个 3B Worker (端口 8000, 8001)
python Mine/run_worker.py --device gpu --gpu 0 --port 8000 --model /path/to/Llama-3.2-3B-Instruct
python Mine/run_worker.py --device gpu --gpu 0 --port 8001 --model /path/to/Llama-3.2-3B-Instruct

# GPU 1,2,3: 同上...
```

### 6.2 启动调度网关

```bash
cd $MINE_BASE
python Mine/run_gateway.py
# 网关在 http://0.0.0.0:7000 启动
```

### 6.3 运行测试

```bash
cd $MINE_BASE/Mine/tools/test
python test_3B_ultra_lite.py
```

## 关键 PYTHONPATH 说明

`run_worker.py` 自动设置:
```
PYTHONPATH=
  Mine/engine/          → import vllm → Mine/engine/vllm/
  SLINFER_core/         → import Mine → Mine/
```

`run_gateway.py` 自动设置:
```
PYTHONPATH=
  SLINFER_core/         → import Mine → Mine/
  Mine/config_template/ → import pools_info_template
```

## A10 注意事项

1. **模型大小限制**: 单个 A10 (24GB) 最多放 7B 模型 (13GB+KV)。13B 需要跨 GPU 张量并行
2. **无 AMX**: CPU 推理使用 PyTorch CPU 后端（非 OpenVINO），prefill 速度慢
3. **建议**: 主力使用 3B/7B，4 GPU 各跑 1-2 个 worker
4. **block_size**: 保持 16 (与原始 A100 配置相同)

## 测试不同的系统配置

```python
# ServerlessLLM 模式
{'system': 'serverlessllm'}

# SLINFER 原生模式 (GPU only)
{'system': 'sota', 'enable_cpu': False}

# SLINFER 原生模式 (GPU + CPU)
{'system': 'sota', 'enable_cpu': True}
```
