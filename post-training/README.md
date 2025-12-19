# Screencoder Train: MLLM SFT + RL Training Stack

This repo combines supervised fine-tuning (SFT) via `LLaMA-Factory` and reinforcement learning (GRPO) via `VLM-R1`, plus a minimal vLLM serving setup.

## Contents
- `LLaMA-Factory/`: general SFT for LLM/VLM and evaluation
- `VLM-R1/`: RL training (GRPO) for VLM tasks like REC/OVD/Math
- `vllm/`: minimal scripts for OpenAI-style API with vLLM
- `conda_envs/`: example conda env YAMLs and optional env archives
- `scripts/`: convenience wrappers

## Quickstart

0) Optional bootstrap
```bash
bash scripts/bootstrap_envs.sh         # installs rl, vllm, data, sft
```

1) Environments
```bash
# Option A: from YAMLs (recommended to customize as needed)
conda env create -f conda_envs/sft_env.yml
conda env create -f conda_envs/rl_env.yml
# Optional: a separate env for vLLM
conda create -n vllm python=3.10 -y
```

2) Data
- SFT: see `LLaMA-Factory/data/get_data.sh` and `LLaMA-Factory/data/dataset_info.json`.
- RL: follow dataset links in `VLM-R1/README.md` (COCO, RefCOCO/+/g, LISA-Grounding, etc.).

3) Train
```bash
# SFT (LoRA example)
conda activate sft
bash scripts/run_sft.sh LLaMA-Factory/examples/train_lora/llama3_lora_sft.yaml

# RL (REC GRPO example)
conda activate rl
bash scripts/run_rl.sh VLM-R1/run_scripts/run_grpo_rec.sh
```

4) Inference (OpenAI-style API via vLLM)
```bash
conda activate vllm
bash scripts/run_vllm.sh LLaMA-Factory/examples/inference/llama3_vllm.yaml 8000
```

- Test clients in `LLaMA-Factory/scripts/api_example/`.

## Notes
- Upstream project docs remain authoritative:
  - LLaMA-Factory: see its `README.md` and `examples/` for many model/task recipes.
  - VLM-R1: see its `README.md` and `run_scripts/` for GRPO variants, multi-node, LoRA.
- Large assets (data, checkpoints, env tar parts) are ignored via `.gitignore` by default.

## License
- Each subproject keeps its own license. Follow model/checkpoint licenses accordingly.

## Acknowledgements
- This project builds upon and thanks the following open-source efforts:
  - LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory
  - VLM-R1: https://github.com/om-ai-lab/VLM-R1

