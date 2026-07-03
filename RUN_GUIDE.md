# SLINFER 部署指南

适用于 **4× NVIDIA A10 (24GB) + 256GB 主存 + Intel CPU** 环境。

---

## 1. 环境准备

### 1.1 克隆仓库

```bash
git clone git@github.com:mingyutao687-byte/MINE.git
cd MINE
```

### 1.2 创建虚拟环境

```bash
conda create -n slinfer python=3.11 -y
conda activate slinfer
```

### 1.3 确认 CUDA

```bash
nvcc --version    # 应为 12.x
nvidia-smi        # 确认 GPU 可见
```

### 1.4 设置环境变量（写入 ~/.bashrc）

```bash
echo 'export PROJECT_BASE=/path/to/MINE' >> ~/.bashrc
source ~/.bashrc
```

---

## 2. 安装

```bash
# vLLM 推理引擎
cd engine/vllm
pip install -e .

# ServerlessLLM 模型存储 (C++/CUDA gRPC)
cd ../store/sllm_store
pip install -e .

# Python 依赖
pip install fastapi uvicorn aiohttp ray requests websockets websocket-client

# 回到项目根目录
cd ../..
```

验证安装：

```bash
python -c "import vllm; print(vllm.__file__)"
python -c "from Mine.api.gateway import app; print('gateway OK')"
```

---

## 3. 准备模型

```bash
# GPU 模型 (HuggingFace 格式)
mkdir -p gpu_models
cd gpu_models

# 示例: 下载 Llama-3.2-3B
huggingface-cli download meta-llama/Llama-3.2-3B-Instruct --local-dir Llama-3.2-3B-Instruct

# 或从已有路径软链接
ln -s /existing/path/to/model Llama-3.2-3B-Instruct

cd ..
```

---

## 4. 配置集群

### 4.1 单机 4 GPU + 3B 模型 (推荐)

`config/a10_pools.py` 中已有预置配置，直接使用：

```python
# config/settings.py 末尾
from Mine.config.a10_pools import POOLS_3B_4GPU_0CPU
pools_config = POOLS_3B_4GPU_0CPU
```

### 4.2 自定义拓扑

编辑 `config/a10_pools.py` 添加你自己的配置，然后像上面一样引入。

---

## 5. 启动

### 5.1 一键启动 (推荐)

```bash
# 4 GPU, 每 GPU 2 worker, 自动启动 Gateway
python run_cluster.py --mode local --model llama-3.2-3b --workers-per-gpu 2

# 调试模式: 1 GPU + 4 CPU
python run_cluster.py --mode debug --model llama-3.2-3b
```

### 5.2 分步启动

```bash
# 终端1: 逐个启动 GPU Worker
python run_worker.py --device gpu --gpu 0 --port 8000 --model $PROJECT_BASE/gpu_models/Llama-3.2-3B-Instruct
python run_worker.py --device gpu --gpu 0 --port 8001 --model $PROJECT_BASE/gpu_models/Llama-3.2-3B-Instruct
# ... 每个 GPU 启动你需要数量的 worker

# 终端2: 启动调度网关
python run_gateway.py
```

---

## 6. 验证

```bash
# 检查网关健康
curl http://localhost:7000/ping

# 发送测试请求
curl -X POST http://localhost:7000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "request_info": {
      "request_id": "test-001",
      "model_id": 0,
      "model_type": "llama-3.2-3b",
      "input_length": 128,
      "expect_output_length": 64
    }
  }'

# 运行完整测试套件
cd tools/test
python test_3B_ultra_lite.py
```

---

## 7. 多机部署

每台 GPU 机器执行步骤 1-4，然后：

```bash
# 在每台机器上分别启动 (不同 base_port)
# 机器1: python run_cluster.py --mode local --model llama-3.2-3b --workers-per-gpu 2 --base-port 8000
# 机器2: python run_cluster.py --mode local --model llama-3.2-3b --workers-per-gpu 2 --base-port 8100
# ...

# 在网关机器上启动 Gateway (只启动 Gateway，不启动 Worker)
python run_gateway.py
```

修改 `config/a10_pools.py` 添加多机节点信息（IP + port）。

---

## 8. 切换调度策略

```bash
# ServerlessLLM 模式
curl -X POST http://localhost:7000/set_config \
  -H "Content-Type: application/json" \
  -d '{"system": "serverlessllm"}'

# SLINFER 原生模式 (GPU only)
curl -X POST http://localhost:7000/set_config \
  -H "Content-Type: application/json" \
  -d '{"system": "sota", "enable_cpu": false}'

# 查看当前配置
curl -X POST http://localhost:7000/get_config
```

---

## 常见问题

**Q: nvcc 版本不对？**
```bash
conda install -c nvidia cuda-toolkit=12.4
```

**Q: 端口被占用？**
```bash
lsof -i :7000-8999    # 查看占用
kill -9 <PID>          # 释放端口
```

**Q: GPU 显存不足？**
减少 `--workers-per-gpu` 参数，3B 模型每 GPU 最多 3 个 worker，7B 最多 1 个。
