#!/usr/bin/env python
"""vLLM Worker 启动器 — 设置正确的 PYTHONPATH 后启动 vLLM API server。

Mine/engine/vllm/ 是一个完整的 vLLM fork，但其 import 使用 `from vllm.xxx`
而非 `from Mine.engine.vllm.xxx`。因此需要将 Mine/engine/ 加入 PYTHONPATH
使得 `import vllm` 解析到 Mine/engine/vllm/。

用法:
  # GPU Worker (CUDA)
  python Mine/run_worker.py --device gpu --gpu 0 --port 8000 --model /path/to/model

  # CPU Worker (PyTorch CPU, 无 OpenVINO)
  python Mine/run_worker.py --device cpu --port 8400 --model /path/to/model --numa 0 --kv_gb 32
"""

import sys
import os
import argparse
import subprocess
import shlex

# 将 Mine/engine/ 加入 Python path（使得 import vllm → Mine/engine/vllm/）
_MINE_DIR = os.path.dirname(os.path.abspath(__file__))
_ENGINE_DIR = os.path.join(_MINE_DIR, 'engine')
_SLINFER_CORE = os.path.dirname(_MINE_DIR)

ENV = {
    **os.environ,
    'PYTHONPATH': os.pathsep.join([
        _ENGINE_DIR,          # import vllm → Mine/engine/vllm/
        _SLINFER_CORE,        # import Mine → SLINFER_core/Mine/
        os.environ.get('PYTHONPATH', ''),
    ]),
    'VLLM_ENGINE_ITERATION_TIMEOUT_S': '3600',
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Start a vLLM worker with SLINFER patches')
    parser.add_argument('--device', required=True, choices=['cpu', 'gpu'])
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--numa', type=int, default=None)
    parser.add_argument('--kv_gb', type=int, default=None)
    parser.add_argument('--block_num', type=int, default=256)
    parser.add_argument('--max_model_len', type=int, default=4096)
    parser.add_argument('--no_load', action='store_true', default=True,
                       help='Do not load model at startup (GPU only)')
    args = parser.parse_args()

    if args.device == 'gpu':
        assert args.gpu is not None, '--gpu required for GPU mode'
        ENV['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        ENV['NO_MODEL_LOADING_AT_START'] = 'True'
        block_override = f'--num-gpu-blocks-override {args.block_num}' if args.block_num >= 0 else ''
        cmd = (
            f'python -m vllm.entrypoints.openai.api_server '
            f'--port {args.port} --max-model-len {args.max_model_len} '
            f'--model {args.model} --enforce-eager '
            f'--gpu-memory-utilization 0.975 --block-size 16 '
            f'{block_override}'
        )
    elif args.device == 'cpu':
        assert args.kv_gb is not None, '--kv_gb required for CPU mode'
        ENV['VLLM_OPENVINO_KVCACHE_SPACE'] = str(args.kv_gb)
        numa_cmd = ''
        if args.numa is not None and args.numa >= 0:
            numa_cmd = f'numactl --cpunodebind={args.numa} --membind={args.numa}'
        cmd = (
            f'{numa_cmd} python -m vllm.entrypoints.openai.api_server '
            f'--port {args.port} --max-model-len {args.max_model_len} '
            f'--model {args.model}'
        )

    print(f'Starting vLLM worker: {cmd}')
    print(f'PYTHONPATH={ENV["PYTHONPATH"]}')
    proc = subprocess.Popen(shlex.split(cmd), env=ENV)
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.kill()
        proc.wait()
