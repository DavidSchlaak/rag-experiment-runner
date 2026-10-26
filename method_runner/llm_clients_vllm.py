"""Direct vLLM client used by the experiment method runners.

vLLM installation is hardware-specific. Install a vLLM build that supports the host's
operating system before running experiments.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any, Dict, List


def _default_cache_root() -> Path:
    """Return a per-user cache path without assuming a particular machine."""
    configured = os.environ.get("LLM_CACHE_DIR")
    if configured:
        return Path(configured).expanduser()

    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))

    return base.expanduser() / "experiment_runner"


_CACHE_ROOT = _default_cache_root()
_HF_HOME = Path(os.environ.get("HF_HOME", _CACHE_ROOT / "huggingface")).expanduser()
_HF_HUB_CACHE = Path(os.environ.get("HF_HUB_CACHE", _HF_HOME / "hub")).expanduser()
_HF_XET_CACHE = Path(os.environ.get("HF_XET_CACHE", _HF_HOME / "xet")).expanduser()
_VLLM_CACHE_ROOT = Path(os.environ.get("VLLM_CACHE_ROOT", _CACHE_ROOT / "vllm")).expanduser()

# Set cache locations before importing vLLM or Hugging Face libraries.
os.environ.setdefault("HF_HOME", str(_HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(_HF_HUB_CACHE))
os.environ.setdefault("HF_XET_CACHE", str(_HF_XET_CACHE))
os.environ.setdefault("VLLM_CACHE_ROOT", str(_VLLM_CACHE_ROOT))


def build_generator_configs(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return configured generators, merging shared vLLM defaults if present."""
    generation = config.get("generation", {})
    generators = generation.get("generators", [])
    if not isinstance(generators, list):
        raise ValueError("generation.generators must be a list.")

    providers = generation.get("providers", {})
    if not isinstance(providers, dict):
        raise ValueError("generation.providers must be a dictionary.")

    defaults = providers.get("vllm", {})
    if not isinstance(defaults, dict):
        raise ValueError("generation.providers.vllm must be a dictionary.")

    merged_generators: List[Dict[str, Any]] = []
    for entry in generators:
        if not isinstance(entry, dict):
            raise ValueError("Each generator entry must be a dictionary.")
        generator = dict(defaults)
        generator.update(entry)
        provider = str(generator.get("provider", "vllm")).strip().lower()
        if provider != "vllm":
            raise ValueError(
                f"Generator '{generator.get('name', '<unnamed>')}' uses "
                f"provider '{provider}'. This client runs vLLM directly; "
                "set provider: vllm and use a vLLM/Hugging Face model ID."
            )
        generator["provider"] = "vllm"
        merged_generators.append(generator)

    return merged_generators


def _require_string(config: Dict[str, Any], key: str) -> str:
    value = str(config.get(key, "")).strip()
    if not value:
        raise ValueError(f"Missing required generator field: {key}")
    return value


def setup_generator(generator_config: Dict[str, Any]) -> Dict[str, Any]:
    """Load a vLLM model using per-generator engine options from YAML."""
    provider = str(generator_config.get("provider", "vllm")).strip().lower()
    if provider != "vllm":
        raise ValueError(
            f"Unsupported provider '{provider}'. This client supports direct vLLM inference only."
        )

    name = _require_string(generator_config, "name")
    model_name = _require_string(generator_config, "model")

    engine_options = generator_config.get("vllm_args", {})
    if not isinstance(engine_options, dict):
        raise ValueError(f"vllm_args must be a mapping for generator '{name}'.")

    configured_quantization = generator_config.get("quantization")
    engine_quantization = engine_options.get("quantization")
    if configured_quantization not in (None, "", "none", "None") or engine_quantization not in (
        None,
        "",
        "none",
        "None",
    ):
        raise ValueError(
            f"Generator '{name}' requests quantization, but this project "
            "requires unquantized model weights. Remove the quantization option."
        )

    # Do not pass null quantization options through to vLLM either.
    engine_options = {
        key: value for key, value in engine_options.items()
        if key != "quantization"
    }

    # Common options are configurable, while backend-specific options can be
    # supplied in the vllm_args mapping. Hardware-specific defaults are left
    # to vLLM so CPU, CUDA, ROCm, XPU, and supported plugins can choose them.
    llm_kwargs: Dict[str, Any] = {
        "model": model_name,
        "download_dir": str(
            Path(generator_config.get("download_dir", _HF_HUB_CACHE)).expanduser()
        ),
        "dtype": str(generator_config.get("dtype", "auto")),
        "trust_remote_code": bool(generator_config.get("trust_remote_code", False)),
    }

    for key in (
        "device",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "gpu_memory_utilization",
        "max_model_len",
        "enforce_eager",
        "max_num_seqs",
        "max_num_batched_tokens",
        "swap_space",
        "cpu_offload_gb",
        "kv_cache_dtype",
        "distributed_executor_backend",
        "enable_prefix_caching",
    ):
        if key in generator_config:
            llm_kwargs[key] = generator_config[key]

    # Allow explicitly selected options for hardware backends and vLLM versions.
    llm_kwargs.update(engine_options)

    tensor_parallel_size = int(llm_kwargs.get("tensor_parallel_size", 1))
    if tensor_parallel_size < 1:
        raise ValueError("tensor_parallel_size must be a positive integer.")
    llm_kwargs["tensor_parallel_size"] = tensor_parallel_size

    cache_dir = Path(llm_kwargs["download_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("-" * 72)
    print(f"[MODEL] {name}")
    print(f"[MODEL_ID] {model_name}")
    print(f"[VLLM_OPTIONS] { {key: value for key, value in llm_kwargs.items() if key != 'model'} }")
    print("-" * 72)

    try:
        from vllm import LLM
    except ImportError as exc:
        raise RuntimeError(
            "vLLM is not installed in this Python environment. Install a vLLM "
            "build compatible with this operating system and hardware; see "
            "the official vLLM installation guide."
        ) from exc

    model = LLM(**llm_kwargs)

    return {
        "name": name,
        "provider": provider,
        "model_name": model_name,
        "model": model,
        "seed": int(generator_config.get("seed", 42)),
        "max_output_tokens": generator_config.get("max_output_tokens"),
    }


def generate_answers(
    prompts: List[str],
    generator_state: Dict[str, Any],
    max_output_tokens: int,
    temperature: float,
) -> List[str]:
    if not prompts:
        return []

    try:
        from vllm import SamplingParams
    except ImportError as exc:
        raise RuntimeError("vLLM is required to generate answers.") from exc

    configured_limit = generator_state.get("max_output_tokens")
    if configured_limit is not None:
        max_output_tokens = min(int(max_output_tokens), int(configured_limit))

    sampling_params = SamplingParams(
        temperature=float(temperature),
        max_tokens=int(max_output_tokens),
        seed=int(generator_state.get("seed", 42)),
    )
    conversations = [
        [{"role": "user", "content": str(prompt)}]
        for prompt in prompts
    ]
    outputs = generator_state["model"].chat(
        conversations,
        sampling_params=sampling_params,
        use_tqdm=False,
    )

    if len(outputs) != len(prompts):
        raise RuntimeError(
            f"vLLM returned {len(outputs)} outputs for {len(prompts)} prompts."
        )

    answers: List[str] = []
    for output in outputs:
        if not output.outputs:
            raise RuntimeError("vLLM returned a request without generated text.")
        answers.append(output.outputs[0].text.strip())
    return answers


def generate_answer(
    prompt: str,
    generator_state: Dict[str, Any],
    max_output_tokens: int,
    temperature: float,
) -> str:
    return generate_answers(
        prompts=[prompt],
        generator_state=generator_state,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )[0]


def release_generator(generator_state: Dict[str, Any]) -> None:
    name = generator_state.get("name", "unknown")
    model = generator_state.pop("model", None)
    if model is not None:
        del model

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass

    print(f"[MODEL_RELEASED] {name}")
