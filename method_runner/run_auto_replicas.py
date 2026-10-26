#!/usr/bin/env python3
"""Run dataset experiments with live worker logs, status snapshots and a single-run lock."""

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHOD_RUNNER_DIR = Path(__file__).resolve().parent
DATASETS = ("hupd", "tender", "ledgar")

EXPERIMENT_SCRIPTS = [
     #"dense_parent_child_setup.py",
     #"dense_parent_setup.py",
     "method_runner/hybrid_setup.py",
     #"sparse_parent_setup.py",
     #"sparse_parent_child_setup.py",
    #"sparse_parent_setup_reversed.py",
    # "sparse_parent_child_setup_reversed.py",
]
PRINT_LOCK = threading.Lock()


def timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def emit(message: str) -> None:
    with PRINT_LOCK:
        print(f"[{timestamp()}] {message}", flush=True)


def parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def args_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-dataset runner with live worker output.")
    parser.add_argument("--gpus", required=True, help="Physical GPU IDs, e.g. 0,1,2,3")
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--configs", default="", help="Overrides: hupd=path,tender=path,ledgar=path")
    parser.add_argument("--config", default=None, help="Legacy: only with one dataset")
    parser.add_argument("--generators", default=None)
    parser.add_argument("--scripts", default=None)
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--pass-config-arg", action="store_true",
        help="Pass --config to experiment scripts ONLY if they support this argument.",
    )
    parser.add_argument(
        "--status-interval", type=float, default=15.0,
        help="Seconds between worker status lines (default: 15).",
    )
    parser.add_argument(
        "--no-worker-output", action="store_true",
        help="Hide individual worker lines on terminal (still log to worker files).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show dataset/config/generator/GPU plan without starting workers.",
    )
    return parser.parse_args()


def resolve_path(raw: str | Path) -> Path:
    path = Path(raw)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def dataset_selection(raw: str) -> list[str]:
    selected = parse_csv(raw.lower())
    if not selected or len(selected) != len(set(selected)) or set(selected) - set(DATASETS):
        raise ValueError(f"--datasets must contain unique names from {DATASETS}")
    return selected


def config_overrides(raw: str, selected: list[str], legacy: str | None) -> dict[str, Path]:
    overrides = {}
    for entry in parse_csv(raw):
        if "=" not in entry:
            raise ValueError(f"Invalid --configs entry: {entry!r}")
        name, path = entry.split("=", 1)
        name = name.strip().lower()
        if name not in selected or name in overrides or not path.strip():
            raise ValueError(f"Invalid/duplicate config override: {name!r}")
        overrides[name] = resolve_path(path.strip())
    if legacy is not None:
        if len(selected) != 1 or selected[0] in overrides:
            raise ValueError("--config requires exactly one dataset and no --configs override")
        overrides[selected[0]] = resolve_path(legacy)
    return overrides


def find_config(dataset: str, overrides: dict[str, Path]) -> Path:
    if dataset in overrides:
        path = overrides[dataset]
        if not path.is_file():
            raise FileNotFoundError(f"Config for {dataset} not found: {path}")
        return path
    folder = PROJECT_ROOT / f"{dataset}_dataset"
    for candidate in [folder / f"{dataset}_config.yaml", folder / f"{dataset}_config.yml"]:
        if candidate.is_file():
            return candidate.resolve()
    candidates = sorted(set(folder.glob("*config*.yaml")) | set(folder.glob("*config*.yml")))
    if len(candidates) == 1:
        return candidates[0].resolve()
    raise FileNotFoundError(
        f"Cannot identify config for {dataset}; candidates={candidates}. Use --configs."
    )


def read_generators(config_path: Path, names: str | None) -> list[dict[str, Any]]:
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or not isinstance(config.get("generation"), dict):
        raise ValueError(f"Invalid generation config: {config_path}")
    generation = config["generation"]
    entries = generation.get("generators")
    if not isinstance(entries, list):
        raise ValueError(f"generation.generators must be a list: {config_path}")
    providers = generation.get("providers", {})
    if not isinstance(providers, dict):
        raise ValueError(f"generation.providers must be a mapping: {config_path}")
    vllm_defaults = providers.get("vllm", {})
    if not isinstance(vllm_defaults, dict):
        raise ValueError(f"generation.providers.vllm must be a mapping: {config_path}")
    generators = []
    for generator in entries:
        if not isinstance(generator, dict):
            raise ValueError(f"Invalid generator entry in {config_path}: {generator!r}")
        if str(generator.get("provider", "vllm")).strip().lower() != "vllm":
            continue
        merged = dict(vllm_defaults)
        merged.update(generator)
        if not isinstance(merged.get("name"), str) or not merged["name"]:
            raise ValueError(f"Generator missing name: {config_path}")
        try:
            tp = int(merged.get("tensor_parallel_size", 1))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid tensor_parallel_size for {merged['name']} in {config_path}"
            ) from error
        if tp < 1:
            raise ValueError(f"tensor_parallel_size must be positive for {merged['name']}")
        merged["tensor_parallel_size"] = tp
        generators.append(merged)
    if names:
        selected = set(parse_csv(names))
        missing = selected - {generator["name"] for generator in generators}
        if missing:
            raise ValueError(f"Unknown generators in {config_path}: {sorted(missing)}")
        generators = [generator for generator in generators if generator["name"] in selected]
    if not generators:
        raise ValueError(f"No matching vLLM generators in {config_path}")
    return generators


def select_scripts(raw: str | None) -> list[Path]:
    names = parse_csv(raw) if raw is not None else EXPERIMENT_SCRIPTS
    if not names:
        raise ValueError("No experiment scripts selected")
    scripts = []
    for name in names:
        path = resolve_path(name)
        if not path.is_file() and not Path(name).is_absolute():
            path = (METHOD_RUNNER_DIR / name).resolve()
        scripts.append(path)
    missing = [str(script) for script in scripts if not script.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing experiment scripts: {missing}")
    return scripts


def safe_name(name: str) -> str:
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in name)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(script: Path, config: Path, generator: dict[str, Any]) -> str:
    # Retain compatibility with markers created by the previous runner.
    payload = {
        "experiment_script": script.name,
        "experiment_script_hash": file_hash(script),
        "config_hash": file_hash(config),
        "generator": generator,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def completed(marker: Path, expected: str) -> bool:
    try:
        with marker.open(encoding="utf-8") as stream:
            data = json.load(stream)
        return data.get("status") == "completed" and data.get("fingerprint") == expected
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    temporary.replace(path)


def write_completion(
    marker: Path, run_hash: str, script: Path, generator: str,
    groups: list[list[str]], codes: list[int],
) -> None:
    write_json_atomic(marker, {
        "status": "completed", "completed_at": timestamp(),
        "experiment_script": script.name, "generator": generator,
        "gpu_groups": groups, "return_codes": codes, "fingerprint": run_hash,
    })


def gpu_groups(gpus: list[str], generator: dict[str, Any]) -> tuple[int, list[list[str]]]:
    tp = int(generator.get("tensor_parallel_size", 1))
    if tp < 1:
        raise ValueError(f"Invalid tensor_parallel_size={tp} for {generator['name']}")
    groups = [gpus[index:index + tp] for index in range(0, len(gpus), tp)]
    groups = [group for group in groups if len(group) == tp]
    return tp, groups


def environment(
    dataset: str, config: Path, script: Path, generator: str,
    group: list[str], index: int, count: int,
) -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "EXPERIMENT_DATASET": dataset,
        "EXPERIMENT_CONFIG": str(config),
        "CONFIG_PATH": str(config),
        "EXPERIMENT_SCRIPT": script.name,
        "RUN_ONLY_GENERATOR": generator,
        "RUN_ONLY_GENERATORS": generator,
        "CUDA_VISIBLE_DEVICES": ",".join(group),
        "WORKER_INDEX": str(index),
        "WORKER_COUNT": str(count),
        "PYTHONUNBUFFERED": "1",
    })
    prefix = env.get("CONDA_PREFIX", "")
    if prefix:
        conda_lib = Path(prefix) / "lib"
        libstdcpp = conda_lib / "libstdc++.so.6"
        if libstdcpp.is_file():
            env["LD_PRELOAD"] = ":".join(filter(None, [str(libstdcpp), env.get("LD_PRELOAD", "")]))
        env["LD_LIBRARY_PATH"] = ":".join(filter(None, [str(conda_lib), env.get("LD_LIBRARY_PATH", "")]))
    return env


def worker_reader(
    process: subprocess.Popen, log_path: Path, label: str, print_output: bool,
) -> None:
    # Each process has one dedicated thread to avoid blocking on full stdout pipes.
    try:
        with log_path.open("w", encoding="utf-8") as output:
            assert process.stdout is not None
            for line in process.stdout:
                output.write(line)
                output.flush()
                if print_output:
                    emit(f"[{label}] {line.rstrip()}")
    except Exception as error:
        emit(f"[LOG_ERROR] worker={label}: {type(error).__name__}: {error}")
    finally:
        if process.stdout is not None:
            process.stdout.close()


def stop_workers(workers: list[dict[str, Any]]) -> None:
    # start_new_session ensures that subprocess descendants share the worker's process group.
    for worker in workers:
        pid = worker["process"].pid
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    for worker in workers:
        process = worker["process"]
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    for worker in workers:
        worker["thread"].join(timeout=2)


def worker_status(workers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = time.monotonic()
    return [
        {
            "index": worker["index"], "pid": worker["process"].pid,
            "gpus": worker["gpus"],
            "state": "running" if worker["process"].poll() is None else "exited",
            "exit_code": worker["process"].poll(),
            "elapsed_seconds": round(now - worker["started"], 1),
            "log": str(worker["log"]),
        }
        for worker in workers
    ]


def run_generator(
    dataset: str, config: Path, script: Path, generator: dict[str, Any],
    gpus: list[str], log_root: Path, pass_config_arg: bool,
    status_interval: float, print_worker_output: bool, status_file: Path,
) -> tuple[list[int], list[list[str]]]:
    name = generator["name"]
    tp, groups = gpu_groups(gpus, generator)
    if not groups:
        # Resource skips are handled by main(); never spawn a worker without a full TP group.
        raise RuntimeError(f"Insufficient GPUs for {name}: required={tp}, available={len(gpus)}")
    log_dir = log_root / safe_name(script.stem) / safe_name(name)
    log_dir.mkdir(parents=True, exist_ok=True)
    emit(f"[PLAN] dataset={dataset} script={script.name} generator={name} TP={tp} "
         f"workers={len(groups)} groups={groups}")
    emit("[TASK_DISTRIBUTION] The experiment's existing database/SQL task assignment is "
         "unchanged; this runner only assigns GPUs and starts workers.")
    unused = gpus[len(groups) * tp:]
    if unused:
        emit(f"[UNUSED_GPUS] {unused}: incomplete TP group")

    workers: list[dict[str, Any]] = []
    started_at = timestamp()
    try:
        for index, group in enumerate(groups):
            env = environment(dataset, config, script, name, group, index, len(groups))
            cmd = [sys.executable, "-u", str(script)]
            if pass_config_arg:
                cmd += ["--config", str(config)]
            log_path = log_dir / f"worker_{index}.log"
            process = subprocess.Popen(
                cmd, cwd=PROJECT_ROOT, env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                bufsize=1, start_new_session=True,
            )
            label = f"{dataset}/{script.stem}/{safe_name(name)}/w{index}/pid{process.pid}/gpu{','.join(group)}"
            worker = {
                "index": index, "gpus": group, "process": process,
                "started": time.monotonic(), "log": log_path,
            }
            workers.append(worker)
            thread = threading.Thread(
                target=worker_reader, args=(process, log_path, label, print_worker_output),
                name=f"log-worker-{index}", daemon=True,
            )
            worker["thread"] = thread
            thread.start()
            emit(f"[WORKER_STARTED] index={index} pid={process.pid} gpus={','.join(group)} "
                 f"command={' '.join(cmd)} log={log_path}")
        last_report = 0.0
        while True:
            now = time.monotonic()
            codes = [worker["process"].poll() for worker in workers]
            if now - last_report >= status_interval or all(code is not None for code in codes):
                statuses = worker_status(workers)
                details = " | ".join(
                    f"w{item['index']}:pid={item['pid']},gpus={','.join(item['gpus'])},"
                    f"{item['state']},exit={item['exit_code']},elapsed={item['elapsed_seconds']}s"
                    for item in statuses
                )
                emit(f"[WORKER_STATUS] {dataset}/{script.stem}/{name} | {details}")
                write_json_atomic(status_file, {
                    "updated_at": timestamp(), "runner_pid": os.getpid(),
                    "dataset": dataset, "script": script.name, "generator": name,
                    "started_at": started_at, "tensor_parallel_size": tp,
                    "workers": statuses,
                })
                last_report = now
            if all(code is not None for code in codes):
                for worker in workers:
                    worker["thread"].join(timeout=2)
                    if worker["thread"].is_alive():
                        emit(f"[LOG_WARNING] w{worker['index']} log thread still active (descendant may own stdout)")
                emit(f"[RUN_FINISHED] dataset={dataset} script={script.name} generator={name} "
                     f"exit_codes={codes}")
                return [int(code) for code in codes], groups
            time.sleep(0.25)
    except BaseException:
        stop_workers(workers)
        raise


def acquire_lock() -> Any:
    path = PROJECT_ROOT / ".run_all_datasets.lock"
    stream = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.seek(0)
        other = stream.read().strip()
        stream.close()
        raise RuntimeError(f"Another runner holds the project lock: {path}; holder={other}") from None
    stream.seek(0)
    stream.truncate()
    stream.write(f"pid={os.getpid()} started={timestamp()}\n")
    stream.flush()
    return stream


def main() -> int:
    args = args_parser()
    if args.status_interval <= 0:
        raise ValueError("--status-interval must be positive")
    datasets = dataset_selection(args.datasets)
    overrides = config_overrides(args.configs, datasets, args.config)
    gpus = parse_csv(args.gpus)
    if not gpus or len(gpus) != len(set(gpus)) or not all(gpu.isdecimal() for gpu in gpus):
        raise ValueError("--gpus must contain distinct numeric IDs")
    scripts = select_scripts(args.scripts)
    if args.dry_run:
        emit("[DRY_RUN] No processes will start")
        for dataset in datasets:
            config = find_config(dataset, overrides)
            generators = read_generators(config, args.generators)
            for script in scripts:
                for generator in generators:
                    tp, groups = gpu_groups(gpus, generator)
                    if not groups:
                        emit(f"[SKIP_INSUFFICIENT_GPUS] dataset={dataset} script={script.name} "
                             f"generator={generator['name']} required={tp} available={len(gpus)}")
                        continue
                    emit(f"[PLAN] dataset={dataset} config={config} script={script.name} "
                         f"generator={generator['name']} TP={tp} worker_groups={groups}")
        return 0

    lock = acquire_lock()
    status_file = PROJECT_ROOT / "run_all_datasets_status.json"
    completed_runs: list[tuple[str, str, str]] = []
    skipped_runs: list[tuple[str, str, str]] = []
    resource_skipped_runs: list[tuple[str, str, str, int, int]] = []
    failed_runs: list[tuple[str, str, str, str]] = []
    try:
        emit(f"[RUNNER_STARTED] pid={os.getpid()} root={PROJECT_ROOT} datasets={datasets} "
             f"gpus={gpus} scripts={[script.name for script in scripts]} "
             f"status_file={status_file}")
        emit("[CONFIG_WARNING] Each experiment script must read EXPERIMENT_CONFIG (or use "
             "--pass-config-arg if supported); setting the environment variable alone does not "
             "change a hardcoded dataset path.")
        for dataset in datasets:
            emit(f"[DATASET_START] {dataset}")
            try:
                config = find_config(dataset, overrides)
                with config.open(encoding="utf-8") as stream:
                    config_data = yaml.safe_load(stream)
                if isinstance(config_data, dict):
                    configured_dataset = str(config_data.get("dataset", dataset)).strip().lower()
                    if configured_dataset != dataset:
                        raise ValueError(
                            f"Config dataset '{configured_dataset}' does not match selected "
                            f"dataset '{dataset}': {config}"
                        )
                generators = read_generators(config, args.generators)
                emit(f"[CONFIG] dataset={dataset} file={config} "
                     f"generators={[generator['name'] for generator in generators]}")
                result_root = PROJECT_ROOT / f"{dataset}_dataset" / "results"
                for script in scripts:
                    for generator in generators:
                        name = generator["name"]
                        marker = result_root / "experiment_state" / safe_name(script.stem) / f"{safe_name(name)}.json"
                        try:
                            tp, groups = gpu_groups(gpus, generator)
                            if not groups:
                                resource_skipped_runs.append((dataset, script.name, name, tp, len(gpus)))
                                emit(f"[SKIP_INSUFFICIENT_GPUS] dataset={dataset} script={script.name} "
                                     f"generator={name} required={tp} available={len(gpus)}")
                                continue
                            run_hash = fingerprint(script, config, generator)
                            if not args.rerun_completed and completed(marker, run_hash):
                                skipped_runs.append((dataset, script.name, name))
                                emit(f"[SKIP_COMPLETED] dataset={dataset} script={script.name} generator={name}")
                                continue
                            if args.rerun_completed:
                                marker.unlink(missing_ok=True)
                            codes, groups = run_generator(
                                dataset, config, script, generator, gpus,
                                result_root / "logs", args.pass_config_arg,
                                args.status_interval, not args.no_worker_output, status_file,
                            )
                            if any(code != 0 for code in codes):
                                raise RuntimeError(f"Worker exit codes: {codes}")
                            write_completion(marker, run_hash, script, name, groups, codes)
                            completed_runs.append((dataset, script.name, name))
                            emit(f"[COMPLETED] dataset={dataset} script={script.name} generator={name}")
                        except Exception as error:
                            failed_runs.append((dataset, script.name, name, str(error)))
                            emit(f"[FAILED] dataset={dataset} script={script.name} generator={name} "
                                 f"reason={type(error).__name__}: {error}")
                            if not args.continue_on_error:
                                return 1
            except Exception as error:
                failed_runs.append((dataset, "CONFIG_OR_SETUP", "-", str(error)))
                emit(f"[DATASET_FAILED] dataset={dataset} reason={type(error).__name__}: {error}")
                if not args.continue_on_error:
                    return 1
            emit(f"[DATASET_DONE] {dataset}")
        emit(f"[SUMMARY] completed={len(completed_runs)} skipped={len(skipped_runs)} "
             f"skipped_insufficient_gpus={len(resource_skipped_runs)} failed={len(failed_runs)}")
        for dataset, script, generator, required, available in resource_skipped_runs:
            emit(f"[SKIPPED_GPU_RUN] dataset={dataset} script={script} generator={generator} "
                 f"required={required} available={available}")
        for dataset, script, generator, reason in failed_runs:
            emit(f"[FAILED_RUN] dataset={dataset} script={script} generator={generator} reason={reason}")
        return 1 if failed_runs else 0
    finally:
        lock.close()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        emit("[STOPPED_BY_USER] Active workers have been terminated")
        sys.exit(130)
    except Exception as error:
        emit(f"[FATAL_ERROR] {type(error).__name__}: {error}")
        sys.exit(1)
