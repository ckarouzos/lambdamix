# LambdaMix

Code for **Steerable Multi-Objective Group-Relative Preference Optimization with Jointly Trained Low-Rank Adapters**.

[Constantinos Karouzos](https://ckarouzos.github.io/) · [Nikos Aletras](https://nikosaletras.com/)

LambdaMix jointly trains low-rank adapters on a frozen language model. A preference vector controls both the adapter mixture used to generate responses and the objective advantages used during optimization. At inference, changing this vector selects an operating point without additional training.

## Navigate

[Quick start](#quick-start) · [Repository map](#repository-map) · [Tasks and data](#tasks-and-data) · [Training](#training) · [Evaluation](#evaluation) · [Baselines and ablations](#baselines-and-ablations) · [Export](#export-and-vllm-evaluation) · [Results](#results)

## Quick start

Use a GPU environment compatible with the pinned dependencies in [pyproject.toml](pyproject.toml). The package declares Python 3.10–3.14 support; GPU package availability depends on your Python version and platform.

```bash
git clone https://github.com/ckarouzos/lambdamix.git
cd lambdamix
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Inspect the first training command without loading models or downloading data:

```bash
python scripts/train_paper.py configs/paper/qwen3_4b/gsm8k/lambdamix.yaml --dry-run
```

The launcher uses each configuration's policy process count and task environment from [the index](configs/paper/index.json). Remove `--dry-run` to train. The command above can also be inspected before installing dependencies: the launcher uses only Python's standard library.

The recorded environment uses PyTorch 2.11.0, Transformers 5.15.0, TRL 0.29.1, PEFT 0.19.1, Accelerate 1.14.0, Datasets 4.0.0, and vLLM 0.20.2. Training uses Transformers generation; vLLM is used for evaluation of base models and exported dense checkpoints. Authenticate with `hf auth login` if a model requires access.

## Repository map

| Path | Contents |
| --- | --- |
| [src/lambdamix/adapters/](src/lambdamix/adapters/) | LambdaMix and Panacea adapters, initialization, and checkpoint loading |
| [src/lambdamix/training/](src/lambdamix/training/) | Training methods and shared setup |
| [src/lambdamix/evaluation/](src/lambdamix/evaluation/) | Preference sweeps, model merging, metrics, and export |
| [src/lambdamix/tasks/](src/lambdamix/tasks/) | Dataset selection and task scoring |
| [configs/paper/](configs/paper/) | Recipes organized by model, task, and method |
| [scripts/](scripts/) | Indexed training launcher, baseline evaluation, and specialist-front aggregation |

## Tasks and data

Each task has two objectives. Preference coordinates follow the order shown below; for example, `(1, 0)` gives all weight to the first objective and `(0.5, 0.5)` gives equal weight to both. GSM8K and PKU load through Hugging Face Datasets.

| Task directory | Dataset key | Source | Objectives | Evaluation protocol |
| --- | --- | --- | --- | --- |
| `gsm8k` | `gsm8k_ifeval` | `openai/gsm8k` | Math correctness; response-constraint compliance | First 300 test problems; 1,024-token cap |
| `pku` | `pku_safety` | `PKU-Alignment/PKU-SafeRLHF` | Beaver reward; negative Beaver cost | All 1,354 prompt-disjoint test prompts; 1,024-token cap |
| `synthpai` | `synthpai` | `RobinSta/SynthPAI` | ROUGE-1 retention; attribute-support rejection | 111 test comments; 512-token cap |

GSM8K uses all 7,473 training problems, each with two deterministic response constraints. Correctness uses the last complete `<answer>...</answer>` span; missing or unclosed spans receive zero. Compliance is the fraction of two constraints satisfied over the full response; the JSON check accepts any parseable JSON value. OLMo uses `gsm8k_ifeval_nativecot`, which omits Qwen's `/no_think` switch while keeping the problems and constraints unchanged.

PKU online training uses 38,641 unique official training prompts. Evaluation removes exact train/test prompt overlap, deduplicates the remaining test prompts, and orders them by the SHA-256 hash of their verbatim text for a fixed order over all 1,354 held-out prompts. The scorers are `PKU-Alignment/beaver-7b-unified-reward` and `PKU-Alignment/beaver-7b-unified-cost`. Preference Vector DPO instead uses the 73,870 annotated training pairs.

Download the unchanged [SynthPAI source JSONL](https://huggingface.co/datasets/RobinSta/SynthPAI):

```bash
python -c 'from huggingface_hub import hf_hub_download; hf_hub_download(repo_id="RobinSta/SynthPAI", filename="synthpai.jsonl", repo_type="dataset", local_dir="data")'
export LAMBDAMIX_SYNTHPAI_SOURCE="$PWD/data/synthpai.jsonl"
```

The loader takes eligible comments in release order: 728 for training followed by 111 for testing. Comments are disjoint across splits; authors may overlap. Selection is defined in [the SynthPAI loader](src/lambdamix/tasks/synthpai.py). It scores stemmed ROUGE-1 against the source text and uses frozen `Qwen/Qwen3-8B` judgments of whether the rewrite supports the supplied attribute values. Invalid rewrites receive zero on both axes. A parsed `NO` judgment scores one for rejection; every other verdict scores zero.

SynthPAI is included only for Qwen3.5-9B and Qwen3.6-27B. The parser requires a complete, nonempty `<anonymized>...</anonymized>` block. The scores measure wording retention and attribute-support rejection.

PKU and SynthPAI also require memory for frozen scoring models. `LAMBDAMIX_PKU_DEVICE` and `LAMBDAMIX_SYNTHPAI_JUDGE_DEVICE` control their placement, for example `cuda:1` or `cpu`. Reserve scoring devices separately from the policy processes in the index. Local scoring snapshots can be selected with `LAMBDAMIX_PKU_REWARD_MODEL`, `LAMBDAMIX_PKU_COST_MODEL`, and `LAMBDAMIX_SYNTHPAI_JUDGE_MODEL`.

## Training

| Configuration directory | Base model | Tasks |
| --- | --- | --- |
| `qwen3_4b` | `Qwen/Qwen3-4B-Instruct-2507` | GSM8K, PKU |
| `qwen35_9b` | `Qwen/Qwen3.5-9B` | GSM8K, PKU, SynthPAI |
| `qwen36_27b` | `Qwen/Qwen3.6-27B` | GSM8K, PKU, SynthPAI |
| `olmo3_7b` | `allenai/Olmo-3-1025-7B` | GSM8K, PKU |

Train LambdaMix on one of the three tasks:

```bash
python scripts/train_paper.py configs/paper/qwen3_4b/gsm8k/lambdamix.yaml
python scripts/train_paper.py configs/paper/qwen35_9b/pku/lambdamix.yaml
python scripts/train_paper.py configs/paper/qwen35_9b/synthpai/lambdamix.yaml
```

Run these commands independently after preparing each task. Checkpoints go to `outputs/<model>/<task>/<method>/`. Command-line options override YAML values; use a distinct `--output_dir` for each new run. Hosted experiment logging is disabled by default.

The index records the policy process count needed for each configuration's batch settings. Changing process count or accumulation requires preserving the effective optimizer batch; changing the generation batch also changes the batch over which rewards are normalized.

The RL budget is 500 updates, eight responses per prompt, and 64 completion draws per update. Defaults include learning rate `1e-6`, cosine scheduling, bfloat16, seed 42, temperature 0.6, top-p 0.95, KL coefficient 0.2, and clipping offsets 0.20/0.28. Warmup and generation batch sizes are recorded per configuration.

LambdaMix has two rank-8 adapters with alpha 128, zero B factors, and A entries sampled uniformly from `[-1/sqrt(8), 1/sqrt(8)]`. It applies adapters to `q_proj` and `v_proj`; larger Qwen models additionally use `in_proj_qkv`, `in_proj_z`, and `out_proj`, excluding visual and auxiliary prediction modules. Primary LambdaMix and GDPO use token-averaged `dapo` loss. AdamW uses betas `(0.9, 0.999)`, epsilon `1e-8`, zero weight decay, and gradient clipping at 1. Each rollout contributes to one update.

LambdaMix standardizes each objective within the eight responses to a prompt using sample standard deviation, weights them by the preference, and standardizes again over the gathered generation batch, with epsilon `1e-4`. Its four preference anchors are `(1,0)`, `(0,1)`, `(0.5,0.5)`, and a seed-0 Dirichlet draw. At update `s`, each anchor is annealed with exponent `s/(0.3*500)` after adding `1e-8`, then normalized. Anchors cycle within each data pass, with the starting position advanced between passes.

## Baselines and ablations

The launcher chooses the appropriate module for each method:

| Configuration | Training module | Role |
| --- | --- | --- |
| `lambdamix.yaml` | `training.lambdamix` | Joint adapter mixture and objective conditioning |
| `lambdamix_no_norm.yaml` | `training.lambdamix` | Raw reward weighting with final batch normalization |
| `prompt.yaml` | `training.prompt` | Preference directive with one LoRA adapter |
| `panacea.yaml` | `training.panacea` | Eight shared and two preference directions; orthogonality weight 0.1 |
| `moc.yaml` | `training.moc` | MOC adaptation with group-relative advantages |
| `prism.yaml` | `training.prism` | Three length-8 prefixes; negative-branch weight 0.1 |
| `hoe.yaml` | `training.hoe` | Top-2 routing over frozen endpoint adapters and a DARE+PCB midpoint |
| `dynamicrw.yaml` | `training.dynamicrw` | Gradient-based dynamic reward weighting |
| `paretopo.yaml` | `training.paretopo` | Two-stage ParetoPO adaptation |
| `grpo.yaml` | `training.fixed` | Balanced GRPO with standard initialization |
| `gdpo.yaml` | `training.fixed` | Fixed-preference GDPO specialists |
| `preference_vector_*.yaml` | `training.preference_vector` | Four positive/negative PKU DPO directions |

All modules above are under `lambdamix`. GRPO scalarizes before normalization; GDPO standardizes each objective before combining it. PKU MOC uses running reward normalization; GSM8K and SynthPAI use clipped `[0,1]` bounds. ParetoPO's first stage lasts 120 updates on GSM8K, four on PKU, and two on SynthPAI. Preference Vector trains for two epochs with global pair batch 128, learning rate `1e-6`, beta 0.1, and a 512-token sequence limit. All recipes use zero weight decay.

For five specialists, run a fixed-policy configuration at `w = 0, 0.25, 0.5, 0.75, 1`, setting `--reward_weights w 1-w` and distinct output directories. For example:

```bash
python scripts/train_paper.py configs/paper/qwen35_9b/pku/gdpo.yaml \
  --reward_weights 1 0 --output_dir outputs/qwen35_9b/pku/gdpo_w100
python scripts/train_paper.py configs/paper/qwen35_9b/pku/gdpo.yaml \
  --reward_weights 0 1 --output_dir outputs/qwen35_9b/pku/gdpo_w000
python scripts/train_paper.py configs/paper/qwen35_9b/pku/hoe.yaml
```

HoE configurations point to these GDPO endpoint directories. Its two frozen rank-8 endpoints and DARE+PCB midpoint are combined by top-2 routing. MOC uses a discrepancy threshold of 0.05; DynamicRW uses regularization `1e-5` and applies updated weights at the next optimizer step.

Qwen3.5 component ablations use `conditioning_mixture_only.yaml` and `conditioning_weight_only.yaml` for all three tasks, plus `conditioning_both_fixed.yaml` for PKU. They inherit the corresponding LambdaMix configuration and change only which preference is fixed to `(0.5,0.5)`. Evaluate every variant at the same eleven mixtures. `lambdamix_no_norm.yaml` removes only per-objective standardization and retains final batch normalization.

## Evaluation

After training, evaluate a LambdaMix checkpoint at eleven evenly spaced preferences:

```bash
LAMBDAMIX_PERSAMPLE_DIR=outputs/qwen3_4b/gsm8k/lambdamix/eval/persample \
lambdamix-eval \
  --model_path outputs/qwen3_4b/gsm8k/lambdamix \
  --benchmark gsm8k_ifeval --split test \
  --resolution 10 --max_eval_samples 300 --max_new_tokens 1024 \
  --batch_size 8 --attn_implementation sdpa \
  --output_path outputs/qwen3_4b/gsm8k/lambdamix/eval/results.json
```

This uses greedy generation and saves objective means, front metrics, generated text, and per-prompt scores. The task settings are:

| Task | Benchmark argument | Test samples | Maximum new tokens |
| --- | --- | --- | --- |
| GSM8K | `gsm8k_ifeval` | 300 | 1,024 |
| PKU | `pku_safety` | 1,354 | 1,024 |
| SynthPAI | `synthpai` | All 111; omit `--max_eval_samples` | 512 |

For OLMo on GSM8K, use `gsm8k_ifeval_nativecot`. Preserve the scoring models and stop-token settings between training and evaluation.

Panacea uses the same evaluator. Prompt checkpoints add `--prompt_conditioned`. MOC, PRISM, and HoE use `scripts/run_modern_baseline_eval.py --method moc|prism|hoe`. Fixed policies, including DynamicRW and ParetoPO, use `scripts/run_objective_point_eval.py --model_kind adapter`; base-model evaluation uses `--model_kind base`. `aggregate_fixed_policy_front_artifact.py` combines specialist points. HoE evaluation accepts `--endpoint_adapters ENDPOINT_1 ENDPOINT_2` when its frozen endpoints have moved.

Evaluate each fixed specialist with its training preference, then assemble a front. The point evaluator writes sidecars next to `results.json`:

```bash
python scripts/run_objective_point_eval.py \
  --model_path outputs/qwen35_9b/pku/gdpo_w100 --model_kind adapter \
  --benchmark pku_safety --split test \
  --reward_funcs pku_helpfulness,pku_harmlessness --lambda_vec 1 0 \
  --max_eval_samples 1354 --max_new_tokens 1024 \
  --output_path outputs/specialists/w100/results.json
# Repeat for w075, w050, w025, and w000 with their matching checkpoints and weights.
python scripts/aggregate_fixed_policy_front_artifact.py \
  --point-dirs outputs/specialists/w100 outputs/specialists/w075 \
    outputs/specialists/w050 outputs/specialists/w025 outputs/specialists/w000 \
  --output-dir outputs/specialist_front
```

The assembled `results.json` and `persample/point*.jsonl` use the same format as conditional sweeps and Preference Vector.

Endpoint merging uses `python -m lambdamix.evaluation.merging` with `--base_model_path`, `--adapter_paths`, and `--reward_funcs`. Select `--merge_strategy linear`, `ties`, `dare`, or `dare_ties` and `--resolution 10`. Paper settings are TIES retention 0.2 and DARE drop rate 0.9. Task arithmetic uses `--merge_strategy linear --coeff_values 0 0.25 0.5 0.75 1`, an independent grid for both endpoint coefficients.

Preference Vector uses `python -m lambdamix.evaluation.preference_vector`, supplying `--positive_adapter_paths` and `--negative_adapter_paths` in helpfulness/harmlessness order, `--reward_funcs pku_helpfulness pku_harmlessness`, `--benchmark pku_safety --split test --simplex_resolution 10 --max_eval_samples 1354 --max_new_tokens 1024`. Use `--help` for the remaining model and output arguments.

## Export and vLLM evaluation

For a fixed preference, merge LambdaMix into the frozen base and generate with vLLM:

```bash
lambdamix-export-merged \
  --model_path outputs/qwen35_9b/gsm8k/lambdamix \
  --lambda_vec 0.5 0.5 --output_dir outputs/merged_w050
python scripts/run_objective_point_eval.py \
  --model_path outputs/merged_w050 --model_kind full --backend vllm \
  --benchmark gsm8k_ifeval --reward_funcs correctness,ifeval_constraint \
  --lambda_vec 0.5 0.5 --output_path outputs/vllm_w050/results.json
```

Repeat at the eleven preferences and aggregate the point directories with the same aggregation script. Set `--tensor_parallel_size` to the number of policy GPUs and `--gpu_memory_utilization` to reserve memory for scorers. Prompt conditioning, PRISM, and HoE use their dedicated evaluators. The vLLM path uses the [offline generation API](https://docs.vllm.ai/en/v0.20.2/api/vllm/entrypoints/llm/).

## Results

Front results contain raw objective means, hypervolume, preference alignment (`inner_product`), distinct non-dominated-point counts (`num_pareto`), per-objective `span`, and `reversals`. GSM8K and SynthPAI use `[0,1]` scales; PKU uses unclipped calibration `g(s)=(s+9)/18` and reference `(0,0)`. Span uses raw means. A reversal is a decrease greater than `1e-6` between adjacent settings as that objective's weight increases. Alignment averages the preference-weighted calibrated scores. Task arithmetic's independent coefficient grid has no simplex preference ordering, so its results omit alignment and reversals.



## License

The code is released under the [MIT License](LICENSE). External dependencies, models, and datasets retain their own licenses.
