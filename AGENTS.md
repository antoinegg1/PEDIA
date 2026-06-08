<!-- Project brief for AI coding agents. README.md is the human-facing version. -->

# AGENTS.md — PERIA Repo Guide

## TL;DR

- **Project**: PERIA (*Perceive, Interact, Reason: Building Tool-Augmented Visual Agents for Spatial Reasoning*).
- **Default eval model**: `./pedia_model/PEDIA_8B`.
- **Naming rule**: project name is **PERIA**, but on-disk prefixes remain `pedia_*` for data, model dirs, and training configs. Do not rename them.
- **Path rule**: keep state under `./pedia_model/`, `./pedia_data/`, and `./outputs/{mixed_rl,eval_results,eval_output,eval_logs,trajectories}/`. Never hard-code absolute paths; use `PEDIA_MODEL` / `PEDIA_DATA` overrides.
- **Runtime rule**: use Conda envs, not Singularity, for new work.

## Repo Layout

| Path | Purpose | Env |
|---|---|---|
| `pedia/` | Tool catalog, eval/inference, data synthesis utilities | `peria-inference` |
| `llamafactory/` | SFT training via LLaMA-Factory | `peria-sft` |
| `train_tool_server/` | HTTP tool server for RL rollouts | `peria-tools` |
| `verl-tool/` | Vendored verl-tool RL training | `peria-rl` |

## Hugging Face Sources

- Models and tool backends: `Changyeli03/pedia_model`
- Released data: `Changyeli03/pedia_data`
- Public SFT base model: `Qwen/Qwen3-VL-8B-Thinking`

Model paths in `./pedia_model/`:

| Path | Purpose |
|---|---|
| `PEDIA_8B/` | Default 8B RL checkpoint |
| `pedia_8b_SFT/` | 8B SFT checkpoint used as RL start |
| `pedia_4b/`, `pedia_2b/` | Optional RL checkpoints |
| `PaddleOCR-VL-1.5/`, `sam3.1/`, `grounding-dino-base/` | Tool backends |

Data paths in `./pedia_data/`:

| Path | Purpose |
|---|---|
| `pedia_sft.tar` | SFT data archive; extract to `./pedia_data/pedia_sft/` |
| `pedia_rl.tar` | RL data archive; extract to `./pedia_data/pedia_rl/` |
| `eval/id/*.parquet`, `eval/ood/*.parquet` | Evaluation parquet files |

## Environments

| Env | Install | Used For |
|---|---|---|
| `peria-inference` | `pip install -U -r pedia/requirements.txt -e ./pedia` | Evaluation, scoring, data synthesis utilities |
| `peria-sft` | `pip install -r llamafactory/requirements.txt` | SFT training |
| `peria-tools` | `cd train_tool_server && pip install -r requirements.txt && cd ..` | RL HTTP tool server |
| `peria-rl` | `cd verl-tool && TORCH_CUDA_ARCH_LIST="8.9" MAX_JOBS=48 NVCC_THREADS=4 pip install flash-attn==2.7.4.post1 --no-build-isolation && pip install -r requirements.txt && cd ..` | RL training |

`peria-rl` is independent from `peria-tools`; do not reuse the Node A tool-server env on Node B.

## Canonical Workflows

### Evaluation

- Example dataset in README: `visual_probe_easy`.
- Use `peria-inference`.
- Download `PEDIA_8B/*` plus the three tool backend dirs.
- Download the needed eval parquet, e.g. `eval/id/visual_probe_easy.parquet`.
- Run:

```bash
DATASET=visual_probe_easy bash pedia/scripts/run_inference.sh
DATASET=visual_probe_easy bash pedia/scripts/run_eval.sh
```

`run_inference.sh` starts local Ray tool actors on GPUs `0,1,2,3` and vLLM `DP_SIZE=4` on GPUs `4,5,6,7`. Eval does **not** use `train_tool_server/scripts/launch_tool_server.sh`.

### SFT

- Use `peria-sft`.
- Download `Qwen/Qwen3-VL-8B-Thinking` to `./pedia_model/Qwen3-VL-8B-Thinking`.
- Download `pedia_sft.tar` and extract under `./pedia_data`.
- Run `bash llamafactory/train.sh`.
- Output: `./pedia_model/pedia_8b_SFT/`.

### RL

- Two-node default.
- Node A uses `peria-tools` and starts the HTTP tool server:

```bash
bash train_tool_server/scripts/launch_tool_server.sh
hostname -i
```

- Node B uses a fresh `peria-rl` env, downloads `pedia_8b_SFT/*`, downloads/extracts `pedia_rl.tar`, then runs:

```bash
TOOL_SERVER_IP=<node-a-ip> \
    bash verl-tool/examples/train/pedia/run_pedia_rl_singlenode.sh
```

The RL launcher builds `http://<node-a-ip>:30888/get_observation`. It also still accepts a full `TOOL_SERVER_URL`.

### Data Synthesis

Do not document or assume a one-command recipe. Data synthesis is dataset-specific because source records, image columns, answer formats, IDs, and prompts must be normalized per dataset. The core Python stages are:

- `pedia.scripts.iterative_sampling_generate`
- `pedia.data_preprocess.augment_traj_data`
- `pedia.data_preprocess.convert_trajectory_to_sft`

Use `peria-inference` for these utilities.

## Evaluation Registry

`pedia/evaluation/eval_datasets.py` is the single source of truth for eval ids.

Registered ids:

- ID: `visual_probe_easy`, `visual_probe_medium`, `visual_probe_hard`, `reason_map`, `reason_map_plus`, `map_trace`
- OOD: `visworld_cube`, `visworld_mmsi`, `visworld_ballgame`, `visworld_paperfolding`, `mapeval_visual`, `babyvision`, `vstar_bench`

To add a dataset, update `DATASET_REGISTRY`; avoid editing shell launchers for dataset paths.

## Gotchas

1. Run `unset ROCR_VISIBLE_DEVICES` before Ray starts on NVIDIA machines.
2. Eval tool calls use local **Ray actors** created by `ToolRouter`; RL tool calls use the **HTTP router** via `TOOL_SERVER_IP` / `TOOL_SERVER_URL`.
3. `train_tool_server/train_tool_server/tools/pedia_base.py` should resolve `_AREAL_ROOT` with 3 `..` levels after repo flattening.
4. Keep `pedia/tool_definitions/agents/paddleocr_tool.py` at `num_replicas: 2`; increasing replicas can OOM.
5. Do not add wandb; training scripts use console logging.
6. `launch_cantainer.sh` is legacy and should not be recommended for new setup.
