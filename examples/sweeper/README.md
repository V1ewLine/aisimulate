<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Sweeper Examples

> [!WARNING]
> **Experimental.** Sweeper is intended for evaluation and feedback, not production capacity
> planning. Its Python API, configuration schema, search behavior, and output may change without a
> standard deprecation period. Sweeper does not guarantee SLA compliance, prediction accuracy, or
> globally optimal configurations.

This directory is the canonical home for backend-neutral and Dynamo-integrated Sweeper examples.

## Run the Backend-Neutral Example

The backend-neutral example injects a deterministic replay runner and does not import Dynamo or
another application framework:

```bash
python examples/sweeper/run_sweep.py \
  --config examples/sweeper/sweep.yaml
```

## Run the Dynamo Integration Example

To use Dynamo Replay or the `dynamo.planner` and `dynamo.router` providers, follow Dynamo's
[Sweeper integration guide](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/developer-guide/knowledge-base/modular-components/ai-simulate-experimental/sweeper-experimental/dynamo-integration.md).
That guide owns the Dynamo build, installation, Planner, Router, and replay-composition steps.

With compatible AI Simulate and Dynamo packages installed, run the general search example with the
explicit Dynamo runner:

```bash
python examples/sweeper/tools/run_sweep.py \
  --config examples/sweeper/configs/smart_sweep.yaml
```

To use another replay implementation, pass its `RunnerFactory` to the `Sweeper` constructor.

The GLM-5-FP8 Pareto-front configuration captures a disaggregated search over `kv_load_ratio`.
Update `workload.trace_path` before running a trace-backed configuration.

## Generate a Synthetic Trace

Generate a Mooncake-format trace whose request rate follows a sine wave:

```bash
python examples/sweeper/tools/gen_sine_trace.py \
  --out /tmp/sweeper-sine-trace.jsonl
```

Compare Planner load predictors on that trace:

```bash
python examples/sweeper/tools/run_load_predictor_sweep.py \
  --trace /tmp/sweeper-sine-trace.jsonl \
  --policies throughput_180_5 throughput_600_5
```

## Search Configurations with the Predict CLI

The standalone [`predict_tuner`](tools/predict_tuner/) tool runs a finite grid through
`aisimulate predict`. Its [search YAML](tools/predict_tuner/search.yaml) contains the
ordinary prediction configuration under `prediction`, plus the search dimensions,
output directory, workload count, metric scope, SLOs, and execution options.
Use the Python environment that contains AISimulate:

```bash
conda activate aiperf
python examples/sweeper/tools/predict_tuner/tune_predict.py \
  -c examples/sweeper/tools/predict_tuner/search.yaml --dry-run
python examples/sweeper/tools/predict_tuner/tune_predict.py \
  -c examples/sweeper/tools/predict_tuner/search.yaml
```

All relative file paths are relative to the search YAML. The supplied GLM-5.2/B300
example uses a local 100K-input trace; edit `prediction.traffic.source.paths` for
another workload. The 36-point grid has 33 valid candidates and three incompatible
chunk/prefill budgets. Omitted search dimensions keep their prediction values.

| YAML field | Default | Behavior |
|---|---|---|
| `prediction` | Required | Ordinary predict configuration; this tool supports aggregated SGLang, one replica, PP=1, CP/DCP=1, and concurrency traffic. |
| `output_dir` | Required | New output directory, or an existing matching campaign with `--resume`. |
| `search.concurrency` | Base value | Nonempty list of positive concurrent request counts. |
| `search.dp_ep` | Base layout | List of `[attention_data, moe_expert]` pairs; both divide the fixed base GPU count; Attention/MoE TP is derived. |
| `search.chunked_prefill_size` | Base value | Nonempty positive list; global chunk before division by DP. Must be explicit here or in `prediction`. |
| `search.max_prefill_tokens` | Base value | Nonempty positive list; per-rank prefill budget. Must be explicit here or in `prediction`. |
| `search.max_sequences` | Base value | Nonempty positive list; running request capacity per DP rank. |
| `search.memory_fraction` | Base value | Nonempty list in `(0, 1]`; requires `kv_cache.capacity.type: default`. |
| `workload.requests_per_concurrency` | `null` | Set `4` for N=4C. A trace uses its first N records without repetition; synthetic traffic uses N requests. Omit to use the entire trace or the base synthetic `stop.requests`. |
| `metrics.ttft_scope` | `first_batch` | `first_batch` selects the initial C client arrivals; `all` selects all N. First-batch mode requires N≥C. |
| `slo.ttft_p50_ms` | `15000` | Strict upper bound on the selected TTFT P50. |
| `slo.tpot_p50_ms` | `33` | Strict upper bound on full-workload request TPOT P50. |
| `run.jobs` | `1` | Positive number of concurrent CPU simulations, independent of target GPU count. |
| `run.timeout_seconds` | `300` | Positive wall-clock timeout per candidate. |

Unknown fields, empty or duplicate choices, invalid types, and insufficient trace
records are rejected. Chunk must align to DP × KV page size; the per-rank chunk
must fit the prefill budget. Invalid combinations are recorded without prediction.
Each candidate sets `max_batched_tokens = chunked_prefill_size / attention_data`
for consistent capacity estimation. Flat Mooncake records and fixed-length
synthetic requests must each have at least two output tokens.

Each candidate runs once with per-request capture. TTFT and TPOT use exact medians
of the selected request records, rather than the CLI's approximate histogram
quantiles. TPOT is `(last_token_ms - first_token_ms) / (output_length - 1)` per
request. TPOT and input/output/total token throughput always cover the full N
requests. Input throughput counts all prompt tokens, including cache hits.
`first_batch` is the initial C client requests, not the scheduler's first microbatch;
this combination of metrics is a custom screening rule, not an all-request TTFT SLO.

`metrics.csv` contains all candidates and five metrics; `ranking.csv` contains
passing candidates ranked by output throughput. `best.yaml` is a normal predict
YAML, and `best.json` records its selected metrics and sample counts. Original CLI
reports, commands and logs live under `candidates/<id>/`. Trace slices are shared
under `workloads/`, prepared once per concurrency without loading the whole trace
into memory. `summary.json` and `manifest.json` record campaign results and identity.

Add `--resume` to reuse validated successful candidates and retry failures.
Only `run.jobs` and `run.timeout_seconds` may change during resume; changes to
workload, grid, SLOs, metric scope, model identity, code or profile metadata require
a new directory. Performance data identity uses paths, sizes and modification
times; do not edit it during a search. Exit code 2 means invalid configuration or
no qualifying result, and 130 means interruption. These are offline predictions;
the tool does not run GPU benchmarks or calibrate the cost model.

## Documentation

Read the canonical [Sweeper documentation](../../docs/sweeper/README.md)
for the search flow, workload schema, optimization goals, and search-space reference. Dynamo-specific
composition is documented in the [Dynamo integration
guide](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/developer-guide/knowledge-base/modular-components/ai-simulate-experimental/sweeper-experimental/dynamo-integration.md).

## 本地 GLM-5.2 / B300 性能采集

[`perf_collector/collect_perf.py`](tools/perf_collector/collect_perf.py) 读取
[`collect.yaml`](tools/perf_collector/collect.yaml)，串行调用仓库的算子、NCCL 和
SGLang AllReduce 采集接口。使用当前 Python 环境，不会安装或切换 SGLang。

```bash
conda activate aiperf
python examples/sweeper/tools/perf_collector/collect_perf.py \
  -c examples/sweeper/tools/perf_collector/collect.yaml --plan-only
# 检查 plan.json 中的 blocker、范围和命令后，执行采集：
python examples/sweeper/tools/perf_collector/collect_perf.py \
  -c examples/sweeper/tools/perf_collector/collect.yaml
```

GLM-5.2 NVFP4 / B300 的本地采集支持 **SGLang 0.5.19**，使用当前 conda Python。
`expected_sglang_version` 检查实际安装版本；新版本不会自动通过兼容检查。
此适配覆盖 BF16 GEMM、NVFP4 MoE、DSA 模块及其子算子，并适配新版 AllReduce 初始化。
预览成功只表示计划生成成功；`blocker` 非空的任务不能执行。
正式运行的 `summary.json` 记录实际成功、失败和产出行数。
通信使用独立接口，NCCL 测试仍需安装 `nccl-tests` 并配置其路径。

`categories: all` 是该模型在指定网格内的以下数据，不是整个模型库：

| 分类 | 内容 |
| --- | --- |
| `gemm` | BF16 router、共享专家、前 3 层 dense MLP、LM head；尺寸从模型配置和 TP 范围推导 |
| `moe` | GLM-5.2 NVFP4 路由专家，不扫描 BF16/FP8 专家或其他模型 |
| `attention` | DSA prefill/decode、skip-indexer、MQA logits、topk、稀疏 attention；投影已包含在 module 中 |
| `communication` | NCCL AllReduce/AllGather/ReduceScatter，以及 SGLang custom AllReduce，按实际通信组大小采集 |

配置只支持 GLM-5.2 NVFP4 / B300 / 单机 CP=1。调度排队、CPU launch、MTP 接受率、完整
speculative decoding 链路，以及模型中仍使用解析公式的算子不在上述采集接口内。
因此 `all` 不能保证端到端预测无误差。数据接入 predict 和覆盖数据库不在本工具内。

### YAML 字段

| 字段 | 含义 |
| --- | --- |
| `model.id` | 固定为 `nvidia/GLM-5.2-NVFP4`，避免扫描其他模型 |
| `model.config` | 本地 `config.json`；检查结构、量化、索引等关键参数与采集器缓存一致，不读取完整权重 |
| `model.kv_cache_dtype` | `fp8` 或 `bfloat16`；DSA compute/projection 使用 BF16 |
| `hardware` | 固定 `b300_sxm`；执行时检查选中设备均为 B300 / SM103 |
| `output_dir` | 输出根目录；所有相对路径以 YAML 所在目录为准 |
| `categories` | `all` 或分类列表，例如 `[moe, attention]` |
| `runtime.expected_sglang_version` | 预期实际安装版本；`null` 表示仅记录，不做此项相等检查 |
| `runtime.gpu_ids` | 传入 `CUDA_VISIBLE_DEVICES` 的本机 GPU 编号，数量必须等于 `num_gpus` |
| `parallelism.num_gpus` | 单个部署的总 GPU 数 |
| `parallelism.attention_tp_sizes` | Attention TP 列表；对应 DP=`num_gpus / TP`，CP=1 |
| `parallelism.moe_ep_sizes` | MoE EP 列表；对应 MoE TP=`num_gpus / EP` |
| `sweeps.gemm_tokens` / `moe_tokens` | 算子 token 数扫描，独立于请求总数 |
| `sweeps.moe_balanced` / `moe_power_law_alphas` | 是否测均匀路由，以及 power-law 路由的 alpha 列表 |
| `sweeps.attention.context_batch_sizes` / `context_sequence_lengths` | Prefill batch 数与本次 query 长度的扫描 |
| `sweeps.attention.prefix_lengths` | Prefill 已有前缀长度，单位 token；不是前缀命中率 |
| `sweeps.attention.generation_batch_sizes` / `generation_sequence_lengths` | Decode batch 数与 KV 长度扫描 |
| `sweeps.attention.context_max_tokens` / `generation_max_tokens` | 采集网格的 batch × sequence 上限；不是 serving `max_prefill_tokens` |
| `communication.collectors` | `[nccl]`、`[custom_allreduce]` 或两者 |
| `communication.nccl_ops` | `all_reduce`、`all_gather`、`reduce_scatter` 的子集 |
| `communication.nccl_tests_dir` | `nccl-tests/build` 路径；`null` 时从 PATH 查找可执行文件 |
| `communication.message_sizes.min_bytes` / `max_bytes_exclusive` / `factor` | 通信消息字节数起点、排他终点和整数倍率（必须 >1） |
| `collection.smoke` / `limit` / `shuffle` | 传入算子接口；smoke 默认最多 4 个顶层 case，显式 limit 优先；不限制通信网格 |
| `collection.measure_power` / `power_test_duration_sec` | 传入功耗采集选项；NCCL 的 duration 参数仅为接口提示 |
| `collection.profile` / `debug` | 传入算子采集器的 profiling/debug 选项 |
| `collection.keep_csv` | 算子默认转换为 parquet；true 保留原始 CSV。独立通信接口始终保留 `*_perf.txt` CSV |
| `collection.timeout_seconds` | 每项任务超时，结束该任务的整组子进程 |
| `collection.continue_on_error` | 失败后是否继续其他任务；任何失败都会使最终退出码非零 |

网格越大采集越久。每个 attention 顶层 case 内还会扫描多个形状，因此 `limit` 不是输出行数上限。
底层仍执行其模型边界、显存预算、硬件能力和失败记录逻辑；`plan.json.expanded` 是 CPU 可展开的
结构范围，不代表 GPU 测量覆盖率。启用 smoke/limit 时 `partial_grid=true`，只用于试运行。

工具将 sweep 写入运行目录内的 `cases/base_ops`，将相关结构写入 `cases/models`，并通过
`AISIM_COLLECTOR_CASES_ROOT` 传给每个采集进程。共享 generator 初始化需要的其他模型声明也会复制，
但展开时使用精确的 `COLLECTOR_MODEL_PATH`；工具检查 GEMM 无通用多余尺寸、MoE/DSA 无其他模型。
已有 `AIC_*`、`COLLECTOR_*` 或 `AISIM_COLLECTOR_*` 环境覆盖会被拒绝，避免 YAML 范围失效。

输出路径是 `output_dir/sglang/<实际版本>/b300_sxm/<UTC时间>/`：

- `config.input.yaml`、`config.resolved.yaml`、`cases/`：原始配置及本次展开配置。
- `plan.json`：范围、GEMM 精确尺寸、CPU 展开结果、命令及不兼容原因。
- `runtime.json`、`provenance.json`、`source_sha256.json`：Python/依赖版本、模型与采集源码校验值；
  正式执行还记录 CUDA/NCCL、GPU 和驱动信息。Git commit 无法取得时记录 `unknown`。
- `jobs/<任务>/`：`command.json`、`console.log`、原始性能文件、采集器 checkpoint/错误记录、`result.json`。
- `summary.json`：正式执行后的总体结果。有效性能行非空、latency 合法、错误摘要为零才会标记成功；
  通信还检查消息扫描的最低行数。失败时保留已有文件，不把部分数据标成完整。

NCCL 原始行保留自身的 NCCL 版本身份，外层目录记录本次 SGLang 环境，不修改数据行的 framework/version。
每次启动创建独立目录，不覆盖仓库自带数据库，也不会自动复用上次成功结果。
