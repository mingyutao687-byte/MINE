#!/usr/bin/env python
"""MINE 集群一键启动 — 支持多 GPU 机器批量启动 Worker + Gateway。

用法:
  # 单机模式 (4 GPU, 每 GPU 2 worker)
  python run_cluster.py --mode local --model llama-3.2-3b --workers-per-gpu 2

  # 集群模式 (多台机器, SSH 远程启动)
  python run_cluster.py --mode cluster --hosts host1,host2,host3,host4 --model llama-3.2-3b

  # 调试模式 (1 GPU + 1 CPU)
  python run_cluster.py --mode debug --model llama-3.2-3b
"""

import argparse
import subprocess
import os
import sys
import time
import shlex
from pathlib import Path

# Mine 目录
MINE_DIR = Path(__file__).parent.resolve()
MINE_PARENT = MINE_DIR.parent.resolve()

# 模型类型 → 推荐配置
MODEL_PRESETS = {
    "llama-3.2-3b": {
        "gpu_memory_gb": 6.1,
        "workers_per_gpu_max": 3,  # 24GB / 8GB ≈ 3
        "block_num": 512,
    },
    "llama-2-7b": {
        "gpu_memory_gb": 12.6,
        "workers_per_gpu_max": 1,  # 24GB / 14GB ≈ 1
        "block_num": 256,
    },
    "llama-3.1-8b": {
        "gpu_memory_gb": 15.0,
        "workers_per_gpu_max": 1,
        "block_num": 256,
    },
}


def _env_for_worker(gpu_id: int) -> dict:
    """构建 Worker 进程的环境变量。"""
    return {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([
            str(MINE_DIR / "engine"),    # import vllm → engine/vllm/
            str(MINE_PARENT),            # import Mine
            os.environ.get("PYTHONPATH", ""),
        ]),
        "CUDA_VISIBLE_DEVICES": str(gpu_id),
        "VLLM_ENGINE_ITERATION_TIMEOUT_S": "3600",
        "NO_MODEL_LOADING_AT_START": "True",
    }


def _build_gpu_cmd(port: int, model_path: str, block_num: int, max_model_len: int) -> str:
    """构建 GPU Worker 启动命令。"""
    block_override = f"--num-gpu-blocks-override {block_num}" if block_num >= 0 else ""
    return (
        f"python -m vllm.entrypoints.openai.api_server "
        f"--port {port} --max-model-len {max_model_len} "
        f"--model {model_path} --enforce-eager "
        f"--gpu-memory-utilization 0.975 --block-size 16 "
        f"{block_override}"
    )


def _build_cpu_cmd(port: int, model_path: str, kv_gb: int, numa: int, max_model_len: int) -> str:
    """构建 CPU Worker 启动命令。"""
    numa_cmd = f"numactl --cpunodebind={numa} --membind={numa}" if numa >= 0 else ""
    return (
        f"{numa_cmd} python -m vllm.entrypoints.openai.api_server "
        f"--port {port} --max-model-len {max_model_len} "
        f"--model {model_path}"
    )


def start_local_gpu_workers(model: str, workers_per_gpu: int, gpu_count: int,
                            model_path: str, base_port: int, block_num: int,
                            max_model_len: int, dry_run: bool) -> list[subprocess.Popen]:
    """在本机启动所有 GPU Worker。"""
    preset = MODEL_PRESETS.get(model, {})
    max_workers = preset.get("workers_per_gpu_max", 2)
    if workers_per_gpu > max_workers:
        print(f"WARNING: {workers_per_gpu} workers/GPU exceeds recommended max {max_workers} for {model}")

    procs = []
    worker_id = 0
    for gpu_id in range(gpu_count):
        for w in range(workers_per_gpu):
            port = base_port + worker_id
            cmd = _build_gpu_cmd(port, model_path, block_num, max_model_len)
            env = _env_for_worker(gpu_id)

            print(f"[GPU{gpu_id}-W{w}] :{port}  {model_path}")
            if not dry_run:
                proc = subprocess.Popen(
                    shlex.split(cmd), env=env,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                procs.append(proc)
            worker_id += 1

    return procs


def start_local_cpu_workers(worker_count: int, model_path: str, kv_gb: int,
                            base_port: int, max_model_len: int, dry_run: bool) -> list[subprocess.Popen]:
    """在本机启动 CPU Worker。"""
    procs = []
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([
            str(MINE_DIR / "engine"),
            str(MINE_PARENT),
            os.environ.get("PYTHONPATH", ""),
        ]),
        "VLLM_ENGINE_ITERATION_TIMEOUT_S": "3600",
        "VLLM_OPENVINO_KVCACHE_SPACE": str(kv_gb),
    }

    for w in range(worker_count):
        port = base_port + w
        cmd = _build_cpu_cmd(port, model_path, kv_gb, numa=-1, max_model_len=max_model_len)
        print(f"[CPU-W{w}] :{port}  {model_path}")
        if not dry_run:
            proc = subprocess.Popen(
                shlex.split(cmd), env=env,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            procs.append(proc)

    return procs


def start_remote_workers(hosts: list[str], model: str, workers_per_gpu: int,
                         model_path: str, base_port: int, block_num: int,
                         max_model_len: int, dry_run: bool):
    """通过 SSH 在远程主机上启动 Worker。"""
    for host in hosts:
        remote_cmd = (
            f"cd {MINE_PARENT} && "
            f"python Mine/run_cluster.py --mode local --model {model} "
            f"--workers-per-gpu {workers_per_gpu} --gpu-count 1 "
            f"--model-path {model_path} --base-port {base_port} "
            f"--block-num {block_num} --max-model-len {max_model_len}"
        )
        print(f"[{host}] {remote_cmd}")
        if not dry_run:
            subprocess.Popen(
                ["ssh", host, remote_cmd],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )


def start_gateway(dry_run: bool):
    """启动调度网关。"""
    print(f"\n[Gateway] :7000")
    if not dry_run:
        import uvicorn
        sys.path.insert(0, str(MINE_PARENT))
        sys.path.insert(0, str(MINE_DIR / "config_template"))
        from Mine.api.gateway import app
        # 在新线程中启动，避免阻塞
        import threading
        def _run():
            uvicorn.run(app, host="0.0.0.0", port=7000, log_level='warning')
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return t
    return None


def main():
    parser = argparse.ArgumentParser(description="MINE cluster launcher")
    parser.add_argument("--mode", choices=["local", "cluster", "debug"], default="local",
                       help="local=单机多GPU, cluster=SSH多机, debug=1GPU+1CPU")
    parser.add_argument("--model", default="llama-3.2-3b",
                       choices=["llama-3.2-3b", "llama-2-7b", "llama-3.1-8b"])
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--gpu-count", type=int, default=4)
    parser.add_argument("--cpu-workers", type=int, default=0)
    parser.add_argument("--model-path", type=str, default=None,
                       help="模型路径 (默认从 PROJECT_BASE 推断)")
    parser.add_argument("--base-port", type=int, default=8000)
    parser.add_argument("--cpu-base-port", type=int, default=8400)
    parser.add_argument("--block-num", type=int, default=None)
    parser.add_argument("--cpu-kv-gb", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--hosts", type=str, default=None,
                       help="远程主机列表, 逗号分隔 (cluster 模式)")
    parser.add_argument("--dry-run", action="store_true",
                       help="只打印命令, 不实际启动")
    args = parser.parse_args()

    # 模型路径
    project_base = os.environ.get("PROJECT_BASE", str(MINE_PARENT.parent))
    model_path = args.model_path
    if not model_path:
        model_path_map = {
            "llama-3.2-3b": f"{project_base}/gpu_models/Llama-3.2-3B-Instruct",
            "llama-2-7b": f"{project_base}/gpu_models/Llama-2-7b-chat-hf",
            "llama-3.1-8b": f"{project_base}/gpu_models/Llama-3.1-8B-Instruct",
        }
        model_path = model_path_map.get(args.model, model_path)

    # block_num 默认值
    block_num = args.block_num
    if block_num is None:
        block_num = MODEL_PRESETS.get(args.model, {}).get("block_num", 256)

    print(f"=== MINE Cluster Launcher ===")
    print(f"  Mode: {args.mode}, Model: {args.model}, Workers/GPU: {args.workers_per_gpu}")
    print(f"  Model path: {model_path}")
    if args.dry_run:
        print("  *** DRY RUN — no processes will be started ***")
    print()

    procs = []

    if args.mode == "cluster":
        hosts = args.hosts.split(",") if args.hosts else []
        if not hosts:
            print("ERROR: --hosts required for cluster mode")
            sys.exit(1)
        start_remote_workers(hosts, args.model, args.workers_per_gpu,
                            model_path, args.base_port, block_num, args.max_model_len, args.dry_run)
    elif args.mode == "local":
        gpu_procs = start_local_gpu_workers(
            args.model, args.workers_per_gpu, args.gpu_count,
            model_path, args.base_port, block_num, args.max_model_len, args.dry_run,
        )
        procs.extend(gpu_procs)
        if args.cpu_workers > 0:
            cpu_procs = start_local_cpu_workers(
                args.cpu_workers, model_path, args.cpu_kv_gb,
                args.cpu_base_port, args.max_model_len, args.dry_run,
            )
            procs.extend(cpu_procs)
    elif args.mode == "debug":
        gpu_procs = start_local_gpu_workers(
            args.model, 1, 1, model_path, 8000, block_num, args.max_model_len, args.dry_run,
        )
        cpu_procs = start_local_cpu_workers(
            4, model_path, args.cpu_kv_gb, 8400, args.max_model_len, args.dry_run,
        )
        procs = gpu_procs + cpu_procs

    if args.dry_run:
        print("\nDry run complete. Remove --dry-run to actually start.")
        return

    if args.mode == "local" or args.mode == "debug":
        print(f"\nStarted {len(procs)} workers. Waiting... (Ctrl+C to stop all)")
        start_gateway(False)
        try:
            for p in procs:
                p.wait()
        except KeyboardInterrupt:
            print("\nStopping all workers...")
            for p in procs:
                p.kill()
            for p in procs:
                p.wait()
            print("All stopped.")


if __name__ == "__main__":
    main()
