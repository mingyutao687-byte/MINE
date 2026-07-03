"""vLLM 后端启动器 — 单个 vLLM Worker 进程的启动脚本。

支持 CPU (OpenVINO) 和 GPU (CUDA) 两种后端。
CPU 模式使用 numactl 绑定 NUMA 节点；GPU 模式使用 CUDA_VISIBLE_DEVICES 选择 GPU。

原文件: tools/vllm_backend_starter.py
"""

import subprocess
import argparse
import shlex
import os


def _build_env_vars(args) -> dict:
    """构建 vLLM 进程的环境变量。"""
    env_vars = {
        'VLLM_ENGINE_ITERATION_TIMEOUT_S': '3600',
    }
    return env_vars


def _build_cpu_command(args) -> str:
    """构建 CPU 模式的启动命令。"""
    if args.numa is None:
        raise ValueError('Argument numa is needed for CPU mode.')
    if args.kv_gb is None:
        raise ValueError('Argument kv_gb is needed for CPU mode.')

    numa_command = ''
    if args.numa >= 0:
        numa_command = f'numactl --cpunodebind={args.numa} --membind={args.numa}'

    max_model_len = args.max_model_len or 4096

    return (
        f'{numa_command} '
        f'python -m vllm.entrypoints.openai.api_server '
        f'--port {args.port} --max-model-len {max_model_len} '
        f'--model {args.model} '
    )


def _build_gpu_command(args) -> str:
    """构建 GPU 模式的启动命令。"""
    if args.gpu is None:
        raise ValueError('Argument gpu is needed for GPU mode.')
    if args.block_num is None:
        raise ValueError('Argument block_num is needed for GPU mode.')
    if args.use_sllm is None:
        raise ValueError('Argument use_sllm is needed for GPU mode.')
    if args.no_load is None:
        raise ValueError('Argument no_load is needed for GPU mode.')

    max_model_len = args.max_model_len or 4096
    block_num_override = ''
    if args.block_num >= 0:
        block_num_override = f'--num-gpu-blocks-override {args.block_num}'

    assert args.use_sllm in ['True', 'False']
    loader_override = ''
    if args.use_sllm == 'True':
        loader_override = '--load-format serverless_llm'

    assert args.no_load in ['True', 'False']

    return (
        f'python -m vllm.entrypoints.openai.api_server '
        f'--port {args.port} --max-model-len {max_model_len} '
        f'--model {args.model} '
        f'--enforce-eager --gpu-memory-utilization 0.975 '
        f'{block_num_override} --block-size 16 '
        f'{loader_override} '
    )


def _build_env_vars_for_mode(args, env_vars: dict) -> dict:
    """根据设备类型添加特定环境变量。"""
    if args.device == 'cpu':
        env_vars.update({'VLLM_OPENVINO_KVCACHE_SPACE': f'{args.kv_gb}'})
    elif args.device == 'gpu':
        env_vars.update({
            'CUDA_VISIBLE_DEVICES': f'{args.gpu}',
            'NO_MODEL_LOADING_AT_START': f'{args.no_load}',
        })
    return env_vars


def main():
    parser = argparse.ArgumentParser(description='Start a vLLM backend instance.')
    parser.add_argument('--port', type=int, required=True, help='HTTP server port')
    parser.add_argument('--device', type=str, required=True, choices=['cpu', 'gpu'])
    parser.add_argument('--model', type=str, required=True, help='Model path or name')
    parser.add_argument('--max_model_len', type=int, required=False, help='Max model length')

    # CPU-only parameters
    parser.add_argument('--kv_gb', type=int, required=False, help='KV cache size in GB (CPU)')
    parser.add_argument('--numa', type=int, required=False, help='NUMA node binding (CPU)')

    # GPU-only parameters
    parser.add_argument('--gpu', type=int, required=False, help='GPU device ID')
    parser.add_argument('--block_num', type=int, required=False, help='GPU block override count')
    parser.add_argument('--use_sllm', type=str, required=False, choices=['True', 'False'])
    parser.add_argument('--no_load', type=str, required=False, choices=['True', 'False'])

    args = parser.parse_args()

    env_vars = _build_env_vars(args)

    if args.device == 'cpu':
        command_str = _build_cpu_command(args)
    elif args.device == 'gpu':
        command_str = _build_gpu_command(args)
    else:
        raise ValueError(f"Unknown device: {args.device}")

    env_vars = _build_env_vars_for_mode(args, env_vars)

    print(command_str)
    process = subprocess.Popen(
        shlex.split(command_str),
        env={**os.environ, **env_vars},
    )
    try:
        process.wait()
    except KeyboardInterrupt:
        print('Main process received Ctrl-C.')
        process.kill()
        process.wait()


if __name__ == "__main__":
    main()
