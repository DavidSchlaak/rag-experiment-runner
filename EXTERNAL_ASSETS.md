# External assets and setup

## External data ZIP

The ZIP contains 118 files, about 14.42 GiB uncompressed (14767.0 MiB): all 75 raw experiment-result JSONs and 43 original Tender source documents. Keep their project-relative paths exactly as shown. The AlignScore checkpoint is excluded because setup downloads it directly from Hugging Face.

The raw generation JSONs from all three datasets, including each dataset's agentic result, are needed to recalculate metric and AlignScore tables. Tender source documents are needed to rerun dataset_preparation/tender_concept_parser.py. Existing analysis can run from the smaller result tables tracked in Git without this ZIP.

The 75 result JSONs total 14.31 GiB.

### Raw experiment-result JSONs (75)

| Size (MiB) | Project-relative path | Purpose |
|---:|---|---|
| 72.68 | `hupd_dataset/multi_agent/multi_agent_results/hupd_agentic_hupd_answers_Qwen__Qwen2.5-32B-Instruct-agentic-final.json` | Raw experiment result JSON |
| 264.20 | `hupd_dataset/results/hupd_dense_parent_child_Qwen__Qwen2.5-32B-Instruct.json` | Raw experiment result JSON |
| 270.20 | `hupd_dataset/results/hupd_dense_parent_child_Qwen__Qwen2.5-72B-Instruct.json` | Raw experiment result JSON |
| 255.44 | `hupd_dataset/results/hupd_dense_parent_child_Qwen__Qwen2.5-7B-Instruct.json` | Raw experiment result JSON |
| 309.78 | `hupd_dataset/results/hupd_dense_parent_Qwen__Qwen2.5-32B-Instruct.json` | Raw experiment result JSON |
| 303.74 | `hupd_dataset/results/hupd_dense_parent_Qwen__Qwen2.5-72B-Instruct.json` | Raw experiment result JSON |
| 312.22 | `hupd_dataset/results/hupd_dense_parent_Qwen__Qwen2.5-7B-Instruct.json` | Raw experiment result JSON |
| 277.93 | `hupd_dataset/results/hupd_hybrid_parent_child_Qwen__Qwen2.5-32B-Instruct.json` | Raw experiment result JSON |
| 275.25 | `hupd_dataset/results/hupd_hybrid_parent_child_Qwen__Qwen2.5-72B-Instruct.json` | Raw experiment result JSON |
| 279.75 | `hupd_dataset/results/hupd_hybrid_parent_child_Qwen__Qwen2.5-7B-Instruct.json` | Raw experiment result JSON |
| 760.41 | `hupd_dataset/results/hupd_sparse_parent_BM25L_Qwen__Qwen2.5-32B-Instruct.json` | Raw experiment result JSON |
| 764.27 | `hupd_dataset/results/hupd_sparse_parent_BM25L_Qwen__Qwen2.5-72B-Instruct.json` | Raw experiment result JSON |
| 762.13 | `hupd_dataset/results/hupd_sparse_parent_BM25L_Qwen__Qwen2.5-7B-Instruct.json` | Raw experiment result JSON |
| 214.46 | `hupd_dataset/results/hupd_sparse_parent_child_Qwen__Qwen2.5-32B-Instruct.json` | Raw experiment result JSON |
| 213.16 | `hupd_dataset/results/hupd_sparse_parent_child_Qwen__Qwen2.5-72B-Instruct.json` | Raw experiment result JSON |
| 216.24 | `hupd_dataset/results/hupd_sparse_parent_child_Qwen__Qwen2.5-7B-Instruct.json` | Raw experiment result JSON |
| 532.70 | `hupd_dataset/results/hupd_sparse_parent_Qwen__Qwen2.5-32B-Instruct.json` | Raw experiment result JSON |
| 528.84 | `hupd_dataset/results/hupd_sparse_parent_Qwen__Qwen2.5-72B-Instruct.json` | Raw experiment result JSON |
| 534.28 | `hupd_dataset/results/hupd_sparse_parent_Qwen__Qwen2.5-7B-Instruct.json` | Raw experiment result JSON |
| 124.92 | `ledgar_dataset/multi_agent/multi_agent_results/ledgar_agentic_ledgar_answers_Qwen__Qwen2.5-32B-Instruct-agentic-final.json` | Raw experiment result JSON |
| 300.40 | `ledgar_dataset/results/ledgar_dense_parent_child_Qwen__Qwen2.5-32B-Instruct.json` | Raw experiment result JSON |
| 301.37 | `ledgar_dataset/results/ledgar_dense_parent_child_Qwen__Qwen2.5-72B-Instruct.json` | Raw experiment result JSON |
| 299.96 | `ledgar_dataset/results/ledgar_dense_parent_child_Qwen__Qwen2.5-7B-Instruct.json` | Raw experiment result JSON |
| 325.88 | `ledgar_dataset/results/ledgar_dense_parent_Qwen__Qwen2.5-32B-Instruct.json` | Raw experiment result JSON |
| 328.72 | `ledgar_dataset/results/ledgar_dense_parent_Qwen__Qwen2.5-72B-Instruct.json` | Raw experiment result JSON |
| 326.39 | `ledgar_dataset/results/ledgar_dense_parent_Qwen__Qwen2.5-7B-Instruct.json` | Raw experiment result JSON |
| 0.00 | `ledgar_dataset/results/ledgar_hybrid_parent_child_Qwen__Qwen2.5-32B-Instruct.json` | Raw experiment result JSON |
| 420.64 | `ledgar_dataset/results/ledgar_hybrid_parent_child_Qwen__Qwen2.5-72B-Instruct.json` | Raw experiment result JSON |
| 419.91 | `ledgar_dataset/results/ledgar_hybrid_parent_child_Qwen__Qwen2.5-7B-Instruct.json` | Raw experiment result JSON |
| 514.80 | `ledgar_dataset/results/ledgar_sparse_parent_BM25L_Qwen__Qwen2.5-32B-Instruct.json` | Raw experiment result JSON |
| 515.92 | `ledgar_dataset/results/ledgar_sparse_parent_BM25L_Qwen__Qwen2.5-72B-Instruct.json` | Raw experiment result JSON |
| 514.27 | `ledgar_dataset/results/ledgar_sparse_parent_BM25L_Qwen__Qwen2.5-7B-Instruct.json` | Raw experiment result JSON |
| 315.96 | `ledgar_dataset/results/ledgar_sparse_parent_child_Qwen__Qwen2.5-32B-Instruct.json` | Raw experiment result JSON |
| 316.93 | `ledgar_dataset/results/ledgar_sparse_parent_child_Qwen__Qwen2.5-72B-Instruct.json` | Raw experiment result JSON |
| 314.95 | `ledgar_dataset/results/ledgar_sparse_parent_child_Qwen__Qwen2.5-7B-Instruct.json` | Raw experiment result JSON |
| 383.17 | `ledgar_dataset/results/ledgar_sparse_parent_Qwen__Qwen2.5-32B-Instruct.json` | Raw experiment result JSON |
| 383.21 | `ledgar_dataset/results/ledgar_sparse_parent_Qwen__Qwen2.5-72B-Instruct.json` | Raw experiment result JSON |
| 381.67 | `ledgar_dataset/results/ledgar_sparse_parent_Qwen__Qwen2.5-7B-Instruct.json` | Raw experiment result JSON |
| 4.97 | `tender_dataset/multi_agent/multi_agent_results/tender_agentic_tender_answers_Qwen__Qwen2.5-32B-Instruct-agentic-final.json` | Raw experiment result JSON |
| 15.63 | `tender_dataset/results/tender_dense_parent_child_meta-llama__Llama-3.1-70B-Instruct.json` | Raw experiment result JSON |
| 15.68 | `tender_dataset/results/tender_dense_parent_child_meta-llama__Llama-3.1-8B-Instruct.json` | Raw experiment result JSON |
| 15.58 | `tender_dataset/results/tender_dense_parent_child_Qwen__Qwen2.5-32B-Instruct.json` | Raw experiment result JSON |
| 15.67 | `tender_dataset/results/tender_dense_parent_child_Qwen__Qwen2.5-72B-Instruct.json` | Raw experiment result JSON |
| 15.54 | `tender_dataset/results/tender_dense_parent_child_Qwen__Qwen2.5-7B-Instruct.json` | Raw experiment result JSON |
| 15.62 | `tender_dataset/results/tender_dense_parent_child_Qwen__Qwen3-235B-A22B-Instruct-2507.json` | Raw experiment result JSON |
| 16.58 | `tender_dataset/results/tender_dense_parent_meta-llama__Llama-3.1-70B-Instruct.json` | Raw experiment result JSON |
| 16.67 | `tender_dataset/results/tender_dense_parent_meta-llama__Llama-3.1-8B-Instruct.json` | Raw experiment result JSON |
| 16.50 | `tender_dataset/results/tender_dense_parent_Qwen__Qwen2.5-32B-Instruct.json` | Raw experiment result JSON |
| 16.59 | `tender_dataset/results/tender_dense_parent_Qwen__Qwen2.5-72B-Instruct.json` | Raw experiment result JSON |
| 16.47 | `tender_dataset/results/tender_dense_parent_Qwen__Qwen2.5-7B-Instruct.json` | Raw experiment result JSON |
| 16.54 | `tender_dataset/results/tender_dense_parent_Qwen__Qwen3-235B-A22B-Instruct-2507.json` | Raw experiment result JSON |
| 19.87 | `tender_dataset/results/tender_hybrid_parent_child_meta-llama__Llama-3.1-70B-Instruct.json` | Raw experiment result JSON |
| 19.94 | `tender_dataset/results/tender_hybrid_parent_child_meta-llama__Llama-3.1-8B-Instruct.json` | Raw experiment result JSON |
| 19.71 | `tender_dataset/results/tender_hybrid_parent_child_Qwen__Qwen2.5-32B-Instruct.json` | Raw experiment result JSON |
| 19.81 | `tender_dataset/results/tender_hybrid_parent_child_Qwen__Qwen2.5-72B-Instruct.json` | Raw experiment result JSON |
| 19.68 | `tender_dataset/results/tender_hybrid_parent_child_Qwen__Qwen2.5-7B-Instruct.json` | Raw experiment result JSON |
| 19.75 | `tender_dataset/results/tender_hybrid_parent_child_Qwen__Qwen3-235B-A22B-Instruct-2507.json` | Raw experiment result JSON |
| 56.00 | `tender_dataset/results/tender_sparse_parent_BM25L_meta-llama__Llama-3.1-70B-Instruct.json` | Raw experiment result JSON |
| 28.21 | `tender_dataset/results/tender_sparse_parent_BM25L_meta-llama__Llama-3.1-8B-Instruct.json` | Raw experiment result JSON |
| 78.59 | `tender_dataset/results/tender_sparse_parent_BM25L_Qwen__Qwen2.5-32B-Instruct.json` | Raw experiment result JSON |
| 78.69 | `tender_dataset/results/tender_sparse_parent_BM25L_Qwen__Qwen2.5-72B-Instruct.json` | Raw experiment result JSON |
| 78.56 | `tender_dataset/results/tender_sparse_parent_BM25L_Qwen__Qwen2.5-7B-Instruct.json` | Raw experiment result JSON |
| 78.63 | `tender_dataset/results/tender_sparse_parent_BM25L_Qwen__Qwen3-235B-A22B-Instruct-2507.json` | Raw experiment result JSON |
| 17.70 | `tender_dataset/results/tender_sparse_parent_child_meta-llama__Llama-3.1-70B-Instruct.json` | Raw experiment result JSON |
| 17.75 | `tender_dataset/results/tender_sparse_parent_child_meta-llama__Llama-3.1-8B-Instruct.json` | Raw experiment result JSON |
| 17.61 | `tender_dataset/results/tender_sparse_parent_child_Qwen__Qwen2.5-32B-Instruct.json` | Raw experiment result JSON |
| 17.72 | `tender_dataset/results/tender_sparse_parent_child_Qwen__Qwen2.5-72B-Instruct.json` | Raw experiment result JSON |
| 17.58 | `tender_dataset/results/tender_sparse_parent_child_Qwen__Qwen2.5-7B-Instruct.json` | Raw experiment result JSON |
| 17.66 | `tender_dataset/results/tender_sparse_parent_child_Qwen__Qwen3-235B-A22B-Instruct-2507.json` | Raw experiment result JSON |
| 31.94 | `tender_dataset/results/tender_sparse_parent_meta-llama__Llama-3.1-70B-Instruct.json` | Raw experiment result JSON |
| 32.02 | `tender_dataset/results/tender_sparse_parent_meta-llama__Llama-3.1-8B-Instruct.json` | Raw experiment result JSON |
| 31.98 | `tender_dataset/results/tender_sparse_parent_Qwen__Qwen2.5-32B-Instruct.json` | Raw experiment result JSON |
| 32.04 | `tender_dataset/results/tender_sparse_parent_Qwen__Qwen2.5-72B-Instruct.json` | Raw experiment result JSON |
| 31.82 | `tender_dataset/results/tender_sparse_parent_Qwen__Qwen2.5-7B-Instruct.json` | Raw experiment result JSON |
| 31.92 | `tender_dataset/results/tender_sparse_parent_Qwen__Qwen3-235B-A22B-Instruct-2507.json` | Raw experiment result JSON |

These 43 Tender source documents total 117.1 MiB and are restored to `tender_dataset/concepts_and_RFPs_raw/`.

### Tender source documents (43)

| Size (MiB) | Project-relative path | Purpose |
|---:|---|---|
| 4.87 | `tender_dataset/concepts_and_RFPs_raw/2024-07-13 Angebotskonzept Einbeck final.pdf` | Tender source document |
| 9.76 | `tender_dataset/concepts_and_RFPs_raw/2024-08-14 Angebotskonzept Bad Honnef.pdf` | Tender source document |
| 0.35 | `tender_dataset/concepts_and_RFPs_raw/20240207 LV_Ausschreibung digitales Monitoringtool_Stadt_Celle (1).pdf` | Tender source document |
| 8.94 | `tender_dataset/concepts_and_RFPs_raw/2024_03_10 Angebotskonzept Dietzenbach.docx` | Tender source document |
| 4.12 | `tender_dataset/concepts_and_RFPs_raw/22 05 17_Urbane Datenplattform_Aalen.pdf` | Tender source document |
| 0.28 | `tender_dataset/concepts_and_RFPs_raw/7886452_Leistungsbeschreibung der Anforderungen.pdf` | Tender source document |
| 0.30 | `tender_dataset/concepts_and_RFPs_raw/Angebotskonzept Fulda v1.pdf` | Tender source document |
| 0.27 | `tender_dataset/concepts_and_RFPs_raw/Ausschreibung Datenplattform BHAG.pdf` | Tender source document |
| 7.93 | `tender_dataset/concepts_and_RFPs_raw/Konzept Augsburger Verkehrsverbund.pdf` | Tender source document |
| 6.24 | `tender_dataset/concepts_and_RFPs_raw/Konzept Ausschreibung Heilbronn-Franken.docx` | Tender source document |
| 6.89 | `tender_dataset/concepts_and_RFPs_raw/Konzept Bamberg Mobilitätsplattform.pdf` | Tender source document |
| 3.76 | `tender_dataset/concepts_and_RFPs_raw/Konzept Hamburger Hochbahn.docx` | Tender source document |
| 0.10 | `tender_dataset/concepts_and_RFPs_raw/Konzept Hof.pdf` | Tender source document |
| 1.23 | `tender_dataset/concepts_and_RFPs_raw/Konzept KielRegion.pdf` | Tender source document |
| 5.12 | `tender_dataset/concepts_and_RFPs_raw/Konzept KSD Hildesheim (1).pdf` | Tender source document |
| 1.69 | `tender_dataset/concepts_and_RFPs_raw/Konzept NAH.SH.pdf` | Tender source document |
| 0.94 | `tender_dataset/concepts_and_RFPs_raw/Konzept Smartboard_Stadt Andernach.pdf` | Tender source document |
| 5.83 | `tender_dataset/concepts_and_RFPs_raw/Konzept Stadt Celle.pdf` | Tender source document |
| 7.20 | `tender_dataset/concepts_and_RFPs_raw/Konzept_Flensburg.pdf` | Tender source document |
| 1.22 | `tender_dataset/concepts_and_RFPs_raw/Konzept_Geestland.pdf` | Tender source document |
| 4.93 | `tender_dataset/concepts_and_RFPs_raw/Konzept_Jena.pdf` | Tender source document |
| 0.70 | `tender_dataset/concepts_and_RFPs_raw/Konzept_Landkreis_Harz.docx` | Tender source document |
| 0.90 | `tender_dataset/concepts_and_RFPs_raw/Konzept_Stadt Friedrichshafen.pdf` | Tender source document |
| 0.81 | `tender_dataset/concepts_and_RFPs_raw/Konzept_Stadt Halle.pdf` | Tender source document |
| 0.30 | `tender_dataset/concepts_and_RFPs_raw/Konzept_Stadt Mainz.pdf` | Tender source document |
| 5.09 | `tender_dataset/concepts_and_RFPs_raw/Konzept_Stadt Oldenburg.docx` | Tender source document |
| 0.61 | `tender_dataset/concepts_and_RFPs_raw/Konzept_Stadt Paderborn_v1.pdf` | Tender source document |
| 0.68 | `tender_dataset/concepts_and_RFPs_raw/Konzept_Stadt Regensburg.pdf` | Tender source document |
| 1.71 | `tender_dataset/concepts_and_RFPs_raw/Konzept_StWendel Social Gardening.pdf` | Tender source document |
| 3.40 | `tender_dataset/concepts_and_RFPs_raw/Konzept_StWendelEnergieweg.pdf` | Tender source document |
| 7.11 | `tender_dataset/concepts_and_RFPs_raw/Konzept_Traunreut_final.pdf` | Tender source document |
| 0.91 | `tender_dataset/concepts_and_RFPs_raw/Konzept_UDP_Digitalagentur Smarte Grenzregion (1).pdf` | Tender source document |
| 1.29 | `tender_dataset/concepts_and_RFPs_raw/Konzept_WFG Heinsberg__KI-P GmbH.pdf` | Tender source document |
| 1.83 | `tender_dataset/concepts_and_RFPs_raw/Konzeptbeschreibung_Menden Digital.pdf` | Tender source document |
| 0.58 | `tender_dataset/concepts_and_RFPs_raw/Leistungsbeschreibung Hof.pdf` | Tender source document |
| 0.44 | `tender_dataset/concepts_and_RFPs_raw/Leistungsbeschreibung KielRegion.pdf` | Tender source document |
| 5.42 | `tender_dataset/concepts_and_RFPs_raw/Leistungsbeschreibung_Flensburg.pdf` | Tender source document |
| 1.38 | `tender_dataset/concepts_and_RFPs_raw/Leistungsbeschreibung_Fulda.pdf` | Tender source document |
| 0.16 | `tender_dataset/concepts_and_RFPs_raw/Leistungsbeschreibung_Smartboard_Smarte_Bahnhofstrasse_TG.pdf` | Tender source document |
| 0.62 | `tender_dataset/concepts_and_RFPs_raw/Leistungsbeschreibung_StWendel Social Gardening.pdf` | Tender source document |
| 0.82 | `tender_dataset/concepts_and_RFPs_raw/Leistungsbeschreibung_StWendelEnergieweg.pdf` | Tender source document |
| 0.26 | `tender_dataset/concepts_and_RFPs_raw/Leistungsverzeichnis Einbeck.pdf` | Tender source document |
| 0.06 | `tender_dataset/concepts_and_RFPs_raw/Leistungsverzeichnis_Augsburg.xlsx` | Tender source document |

## Files downloaded directly from Hugging Face

- AlignScore checkpoint: `yzha/AlignScore/AlignScore-base.ckpt`, saved to `evaluation/evaluation_runner/checkpoints/AlignScore-base.ckpt`.
- Metric and embedding model repositories: `BAAI/bge-m3`, `WENGSYX/Multilingual_SimCSE`, and `Siran-Li/MATCHA`, placed in the standard Hugging Face cache used by the Python libraries.
- LEDGAR and HUPD source datasets: downloaded by `dataset_preparation/hupd_ledgar_dataset_preparation.py` from their Hugging Face dataset sources during preparation.
- Configured Qwen/Llama generation models: vLLM downloads the selected model from Hugging Face when that run starts. Setup does not preload these large, hardware-dependent models. Llama access and license acceptance may be required.

## Configure and restore the external ZIP

1. Upload exactly the 118 files listed above to a stable HTTPS host. Preserve their project-relative paths in the ZIP. Do not include the AlignScore checkpoint, code, `.git`, result tables, `rag_artifacts`, or other dataset folders.
2. Configure the direct ZIP download URL in `setup_resources.py`, set `EXPERIMENT_RUNNER_ASSET_URL`, or pass `--asset-url`. Optionally configure `EXTERNAL_ASSET_ARCHIVE_SHA256`.
3. Run `python setup_resources.py` from the project root. It downloads Hugging Face assets, then restores the 118 external files into their project-relative paths. Existing files are kept unless `--overwrite` is passed.
4. Run `python setup_resources.py --inventory` to list local files at or above 100 MiB.

For analysis of checked-in result tables, run `python setup_resources.py --analysis-only`; it performs no downloads. Use `python setup_resources.py --hf-only` when only Hugging Face assets are needed.
