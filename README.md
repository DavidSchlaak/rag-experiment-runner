# Experiment Runner

This repository contains Python scripts for RAG experiments, dataset preparation, metric evaluation, and analysis for Tender, LEDGAR, and HUPD. Method runners use direct vLLM inference. Users are responsible for observing the licenses and access conditions of the Hugging Face models they choose.

### Tender dataset use restrictions

Tender files may be used only for non-commercial purposes. Commercial use and republication are prohibited.

## Reproduction workflows

The workflows below move from analysis of existing outputs back to source data. For a full rebuild, run **5 → 4 → 3 → 2 → 1**. If experiment JSONs and metric tables already exist, start at level 1. Run commands from the repository root; in PyCharm, set it as the **Working directory**.

### 1. Analyze existing results

This level reads saved generation metrics, AlignScore tables, and human-evaluation workbooks. It creates figures, tables, and significance tests; it does not launch experiments or recalculate metrics.

```bash
python evaluation/analysis/generation_quality/generation_quality_evaluation.py --datasets tender,ledgar,hupd
python evaluation/analysis/evidence_support/evidence_support_evaluation.py --datasets tender,ledgar,hupd
python evaluation/analysis/effective_k_analysis/effective_k_context_tokens.py
python evaluation/analysis/significance_testing/master_significance_testing.py --datasets tender,ledgar,hupd
python evaluation/analysis/significance_testing/human_evaluation_siginifcance_testing.py
```

Inputs are under each dataset's `results/metric_analysis/` and `results/retrieval_analysis/`. Human-evaluation significance testing also needs the evaluator workbooks and administrator mapping workbook under `tender_dataset/results/human_evaluation/`. Fixed-name analysis outputs may be replaced on rerun. These scripts do not automatically read `results_test/`.

The generation-quality, evidence-support, and master significance scripts accept `--dataset` or comma-separated `--datasets`; without a selection they process all three datasets. In PyCharm, put the desired option in **Script parameters**. The human-evaluation significance script uses the Tender workbooks.

### 2. Recalculate generation and retrieval metrics

These runners read experiment JSON/JSONL files from each selected dataset's `results/`; they do not use vLLM.

```bash
python evaluation/evaluation_runner/generation_metric_runner.py --datasets tender,ledgar,hupd
python evaluation/evaluation_runner/AlignScore_runner.py --datasets tender,ledgar,hupd
```

Select datasets with `--datasets tender,ledgar` and optionally filter either runner with `--experiment dense_parent_child`. Generation metrics go to `*_dataset/results_test/metric_analysis/`; AlignScore goes to `*_dataset/results_test/retrieval_analysis/`. The existing `results/` inputs are preserved, but reruns can replace same-named files in `results_test/`. To analyze these new tables, change the analysis input paths or copy the tables into `results/` (which can replace matching files).

AlignScore requires `evaluation/evaluation_runner/checkpoints/AlignScore-base.ckpt` and spaCy's `en_core_web_sm` model. The setup script can download the checkpoint; install the spaCy model with `python -m spacy download en_core_web_sm`.

### 3. Generate experiment answers with vLLM

This stage requires prepared test cases, parent/child JSONL and embedding arrays, plus a vLLM installation suited to the host's hardware. The supplied dataset configs select models from 7B to 235B parameters. Check available RAM/VRAM before running; weights download from Hugging Face on first use. Llama models require Hugging Face access and license acceptance.

For one Tender dense-parent-child run, configure `method_runner/run_auto_replicas.py` in PyCharm with the repository root as **Working directory**, a Linux/WSL interpreter with vLLM, and these **Script parameters**:

```text
--gpus 0 --datasets tender --scripts method_runner/dense_parent_child_setup.py --generators Qwen/Qwen2.5-7B-Instruct
```

Use actual GPU IDs. Dataset and generator lists accept comma-separated names; generator names must match `generation.generators[].name` in the YAML config. To use other hardware, copy the relevant YAML, adjust its model list, `generation.providers.vllm.max_model_len`, and `generation.max_output_tokens`, then select it with `--configs tender=path,ledgar=path,hupd=path`. Config paths are resolved from the repository root.

The orchestrator requires Linux/WSL and numeric GPU IDs; it cannot launch CPU-only runs. For CPU-only vLLM, run a setup script directly and set `EXPERIMENT_CONFIG` and `RUN_ONLY_GENERATOR`; its YAML must select one model and configure CPU execution. Install the matching Linux CPU build of vLLM.

A run processes every test-case row and configured `retrieval.top_k`. Results and resumable state are written under the dataset's `results/`; a run can replace a same-named result JSON. The orchestrator skips matching completed runs unless `--rerun-completed` is set.

### 4. Rebuild parent/child JSONL and embeddings

```bash
python dataset_preparation/chunking_and_embedding.py
```

This processes all three datasets. Tender parent JSONL is derived from `tender_dataset/parsed_concepts/dataset_*.json`; LEDGAR and HUPD use the prepared parent JSONL in each `rag_artifacts/` folder. Child JSONL and parent/child embedding `.npy` files are written to those folders. The script downloads/loads `BAAI/bge-m3` and overwrites same-named artifacts.

### 5. Prepare datasets and synthetic queries

For Tender, restore the external asset ZIP with `python setup_resources.py`, then run `python dataset_preparation/tender_concept_parser.py`. The parser reads the PDF selected by `INPUT_FILENAME` from `tender_dataset/concepts_and_RFPs_raw/` and writes concept JSONs to `tender_dataset/parsed_concepts/`.

For LEDGAR and HUPD, create prepared parent data and baseline test cases with:

```bash
python dataset_preparation/hupd_ledgar_dataset_preparation.py
```

Then generate synthetic queries for either dataset:

```bash
python dataset_preparation/synthetic_query_generation/ledgar_generate_synthetic_queries.py
python dataset_preparation/synthetic_query_generation/hupd_generate_synthetic_queries.py
```

The query scripts require a separately running OpenAI-compatible vLLM server. For example, start this in a Linux/WSL terminal and leave it running:

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct --served-model-name qwen-query-model --host 127.0.0.1 --port 8000
```

Set these environment variables in the PyCharm configuration for the query script; `QWEN_MODEL` must match `--served-model-name`:

```text
SYNTHETIC_API_BACKEND=vllm
QWEN_BASE_URL=http://localhost:8000/v1
QWEN_MODEL=qwen-query-model
QWEN_API_KEY=EMPTY
```

Each query script updates its dataset's test-case CSV, audit log, and checkpoint/backup files. Rerunning preparation or query generation can overwrite or update these files; back up prepared data first.

## Environment setup

Use Linux for the supported vLLM path. On Windows, use WSL2 with a supported Linux distribution. vLLM installation is hardware/backend-specific; follow the [official vLLM installation guide](https://docs.vllm.ai/en/stable/getting_started/installation/).

Create a clean environment (Python 3.12 is a practical cross-backend choice):

```bash
uv venv --python 3.12
source .venv/bin/activate
```

Install vLLM first using the instructions for the host's hardware. For a standard NVIDIA setup, the vLLM guide documents:

```bash
uv pip install vllm --torch-backend=auto
```

For AMD ROCm, Intel XPU, or CPU-only use, follow the corresponding official instructions; installation may require a specific wheel/index, driver, or build. Then install the remaining project dependencies:

```bash
uv pip install -r requirements.txt
```

The dataset configs use unquantized Hugging Face model IDs; `dtype: auto` selects the checkpoint's supported native dtype and does not quantize weights. Model downloads can require substantial disk space and memory. Start with `Qwen/Qwen2.5-7B-Instruct` and enable larger models only if the host has enough resources. Meta Llama models require accepting Meta's license and obtaining Hugging Face access. Authenticate with `hf auth login` when needed.

To keep model weights outside the repository, configure a cache location, for example:

```bash
export LLM_CACHE_DIR=/path/to/large/model-cache
```

The client also respects `HF_HOME`, `HF_HUB_CACHE`, `HF_XET_CACHE`, and `VLLM_CACHE_ROOT`. Hardware-specific settings and a minimal starter config are in [vllm_hardware_config.example.yaml](method_runner/vllm_hardware_config.example.yaml).

## Download project data and model assets

Analysis tables and human-evaluation workbooks are tracked in Git. The external asset ZIP contains all 75 raw experiment-result JSONs (including agentic results) and 43 Tender source documents. Their exact project-relative paths are listed in [EXTERNAL_ASSETS.md](EXTERNAL_ASSETS.md).

Set the ZIP's direct HTTPS URL in `setup_resources.py`, through `EXPERIMENT_RUNNER_ASSET_URL`, or with `--asset-url`, then run `python setup_resources.py` from the repository root. This downloads the configured Hugging Face scorer/embedding assets and restores the 118 ZIP files to their project-relative paths. Setup keeps existing files unless `--overwrite` is passed. The archive is needed to recalculate metrics from raw experiment outputs or rerun Tender parsing; it is not needed for analysis of checked-in result tables. Analysis of checked-in tables needs no setup downloads; `python setup_resources.py --analysis-only` exits without downloading assets. Use `python setup_resources.py --hf-only` to download Hugging Face assets without the ZIP.

The selected generation model is downloaded by vLLM when an experiment starts. Setup does not preload these large, hardware-dependent models; some require Hugging Face authentication or license acceptance.

## Agentic RAG

The agentic runner loads workflow settings and prompts from the selected dataset's `multi_agent/` folder. From the repository root, run the default LEDGAR workflow with:

```bash
python method_runner/agentic_RAG_setup.py
```

Select another dataset by passing its main config; its matching workflow block is loaded automatically:

```bash
python method_runner/agentic_RAG_setup.py --config tender_dataset/tender_config.yaml
python method_runner/agentic_RAG_setup.py --config hupd_dataset/hupd_config.yaml
```

You may also pass a dataset's `agentic_workflow_*_config_block.yaml` directly. Agentic outputs and resumable SQLite state are written to that dataset's configured `results` folder.

