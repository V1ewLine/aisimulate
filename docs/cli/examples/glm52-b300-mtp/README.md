<!--
SPDX-License-Identifier: Apache-2.0
-->

# GLM-5.2-NVFP4：单机 8×B300、MTP=3 初筛配置

本实验使用 `aiconfigurator cli exp`，通过固定 `isl`、`osl`、`prefix` 搜索，**无需 trace**。
两份 YAML 分别搜索合并部署的并行布局（含 CP）、batch 和 context token 预算，固定 MTP=3。
P50 和精度要求仍需在实际实验中验收。

已有共享前缀 trace 时，可使用第 5 节的统一 `aisimulate recommend` 搜索基线；该基线未启用 MTP。

## 1. 使用前核对假设

| 项目 | 示例采用的解释 |
|---|---|
| 后端 | `sglang`，性能数据版本 `0.5.17`，与当前实验配置一致 |
| K | `1000`，因此输入分别为 `100000` / `32000` token |
| “前缀 100K/32K” | 暂按总输入长度解释；如果后面还有独立 prompt，应把它计入总输入 |
| 缓存命中率 | 每个请求有 90% / 60% 的输入 token 已缓存，非“90% / 60% 的请求命中” |
| 两组接受率 | 按顺序分别属于用例 1 / 2，且为各位置的无条件接受概率 |
| 部署搜索 | 两个用例分别搜索；合并部署、PP=1、CP 候选为 1/2/4/8，暂不搜索 P/D 分离 |

性能数据必须覆盖所选布局与工作点；覆盖表中的 PASS 只表示预测路径可运行，
不证明本例的 100K 输入、MTP=3 或误差目标已经验证。
默认量化按模型元数据推断。仓库中该模型的量化元数据声明 KV 为 FP8；`NVFP4` 不表示 KV 也是 FP4，
应检查实际解析结果并与部署保持一致。

## 2. 两组工作负载怎样映射

| 字段/含义 | 用例 1 | 用例 2 |
|---|---:|---:|
| `isl`：总输入 token | 100000 | 32000 |
| `osl`：输出 token | 1000 | 300 |
| `prefix`：已缓存 token | 90000 | 19200 |
| 尚需 prefill 的输入 token | 10000 | 12800 |
| `nextn`：每轮草稿数 | 3 | 3 |
| `nextn_accepted`：平均接受草稿数 | 1.110 | 0.297 |
| 每轮预期输出进度（含保底 token） | 2.110 | 1.297 |

`prefix` 已包含在 `isl` 内，不能把 `isl` 改成仅未命中的长度。
`prefix: 90000` 表示每个请求已有 90000 个输入 token 的 KV 可用，并不是“90% 的请求命中”。
它把缓存命中转换成固定已缓存 token 数，不回放冷缓存、预热、淘汰或请求间复用过程。

无条件位置接受概率对应：

```text
nextn_accepted = p1 + p2 + p3
每轮预期输出 token = 1 + nextn_accepted
```

若输入其实是条件概率（后一个位置的概率以之前全接受为前提），应使用
`p1 + p1*p2 + p1*p2*p3`，两组分别为 `0.71568327` / `0.162534784`。
模型只使用这个平均值，不模拟接受长度随请求和时间的波动。

若 AL 指含保底 token 的平均输出长度，`2.0 ±5%` 的区间是 `[1.9, 2.1]`，
上述两组并不都落在区间内，而且这种 AL 不会接近 0。
若 AL 只统计接受的草稿，则应对比 `1.110` / `0.297`。确认口径后再与线上数据比较。

## 3. 运行

在仓库根目录、已安装 AISimulate 的环境执行：

```bash
aiconfigurator cli exp \
  --yaml-path docs/cli/examples/glm52-b300-mtp/case1.exp.yaml \
  --save-dir ./out/glm52-b300-case1 --top-n 5

aiconfigurator cli exp \
  --yaml-path docs/cli/examples/glm52-b300-mtp/case2.exp.yaml \
  --save-dir ./out/glm52-b300-case2 --top-n 5
```

分别查看输出目录下的 `best_config_topn.csv` 和 `pareto.csv`（实际子目录以终端输出为准）。
`total_gpus: 8` 提供整机预算；`agg_num_gpu_candidates` 指单个 worker 的 GPU 数，
搜索还会考虑用较小 worker 的多个副本占用这 8 张卡。无法放下模型或缺少所需性能数据的组合不会成为可用方案。

`pareto_sweep: false` 让搜索直接使用 `tpot: 33`，不额外扫默认 TPOT 阈值网格；仍会搜索并行布局和 batch 工作点。
`enable_chunked_prefill: true` 扫描解析模型的 `ctx_tokens` 预算，候选由内部规则生成；
这不是 SGLang 原生 `chunked_prefill_size` 与 `max_prefill_tokens` 的两个独立搜索维度。
`exp` 的筛选器按 GPU 预算内的吞吐表现选配置，无需再添加统一 CLI 的 `optimization` 字段。
两个用例分别找各自的优选配置；若要求共用一套配置，应在两组结果中对同一候选做双场景比较，不能把不同输出长度的结果直接混为一个排名。

## 4. 哪些要求不能由这份 YAML 验收

| 原始要求 | 当前处理方式 |
|---|---|
| P50 TPOT <33 ms、P50 TTFT <15 s | YAML 的 `tpot/ttft` 只用于模型工作点初筛，且筛选使用 `≤`。实测需分别统计请求 TPOT/TTFT 的 P50，并按原始严格 `<` 判断 |
| KVV 误差 ≤3% | 当前输入没有此约束字段；须先定义 KVV、参考值和误差计算方法，再以对应测量验收 |
| AL≈2.0（±5%） | 接受率是仿真输入，不是模型能预测或自动校准的输出；不能通过填写目标 AL 证明满足要求 |
| 两组场景最大吞吐 | 初筛只覆盖上述合并部署范围；在目标后端上增加并发/到达率，找到满足 P50 的最大实测吞吐 |

采用固定前缀的实测应记录缓存预热方式、实测 token 命中率、接受率口径和量化设置。
MTP 的 token 常成批输出，P50 TPOT 应按请求的平均每 token 耗时统计，不能直接拿所有 token 间隔的 P50 替代。

两份配置已在 `aiperf` 环境通过 `Task.from_yaml()`、`Task.validate()` 和并行布局枚举检查。
每份枚举得到 65 种不超过 8 卡的 worker 布局，覆盖 CP=1/2/4/8；
各布局的显存与性能数据可用性仍由实际搜索筛选，尚未执行完整搜索或实测精度验证。

## 5. 用已有 trace 搜索统一 CLI 参数

[case1.trace.recommend.yaml](case1.trace.recommend.yaml) 使用
`/loong/run/aisimulate/workload-prefix90.jsonl`，沿用已有实验的 SGLang `0.5.17`。
它搜索单机 8 卡合并部署的并行布局、并发、`max_batched_tokens`、`max_sequences`、KV 页大小和显存比例。
trace 的 `block_size: 1000` 固定不变；它描述输入数据，不作为调优参数。

```bash
aisimulate recommend \
  -c docs/cli/examples/glm52-b300-mtp/case1.trace.recommend.yaml \
  --output-dir ./out/glm52-b300-trace-search
```

下表的路径相对于 `engine.workers.aggregated.`，并发字段除外。

| 可搜索字段 | 含义 | 本例处理 |
|---|---|---|
| `parallelism.replicas` | 完整服务副本数量 | 从合法布局池中联合选择 |
| `parallelism.tensor` | 每个 Attention DP rank 的 TP | 从合法布局池中联合选择 |
| `parallelism.pipeline` | 流水线并行度 | 固定为 1 |
| `parallelism.attention_data` | Attention 数据并行度 | 从合法布局池中联合选择 |
| `parallelism.moe_tensor` | 专家内部的张量并行度 | 从合法布局池中联合选择 |
| `parallelism.moe_expert` | 专家并行度 | 从合法布局池中联合选择 |
| `scheduler.max_batched_tokens` | 通用 token 预算字段；SGLang 限制见下文 | 2048、3072、4096、6144、8192、12288、16384、20480、24576、28672、32768 |
| `scheduler.max_sequences` | 调度器运行请求数上限 | 16、32、64、128、256、512 |
| `kv_cache.block_size` | KV 页大小 | 64 |
| `kv_cache.capacity.memory_fraction` | 显存预算比例 | 0.80、0.85、0.88、0.90 |
| `traffic.load.concurrency`（完整路径） | 整个部署的在途请求数 | 16、32、48、64、80、96、112、128 |

本例用 `preset: default`，在满足模型、后端、显存和 GPU 预算的布局池中搜索，默认 PP=1。
副本数、TP、Attention DP、MoE TP 和 EP 仍会变化。本例总 GPU 数为
`replicas × tensor × pipeline × attention_data = 8`，MoE 划分不再额外乘一次。

不要把这些相关维度全部改成独立的 `[1, 2, 4, 8]`：`preset: false` 会直接组合各个采样值，
再检查组合是否合法。例如，本例还要求 `tensor × attention_data = moe_tensor × moe_expert`。
大量无效组合会被统一记为 `backend-unsupported`，并可能触发 `projection stalled` 提前停止。
这不表示后端只支持已输出的几个配置。需要手动限制布局时，可提供合法的完整 `preset` 列表。

**SGLang 调度限制：** `max_batched_tokens` 参与容量估算，但不会覆盖 SGLang 调度器独立的
`chunked_prefill_size` / `max_prefill_tokens`。当前原生默认值分别为 `8192` / `16384`；
Attention DP 开启后，chunk 还会除以 DP 数。`recommend` 尚未暴露这两个字段，不能据此声称已找到最优 chunk；
`predict` 已支持，固定配置示例见第 6 节。
`prefill_decode_interval` 目前仅在统一 `predict` 中可配，也不是内置搜索维度。

**筛选顺序：** 从 `recommendation.json` 取完成请求且有有效延迟样本的候选，先要求
`median_ttft_ms < 15000` 且 `median_tpot_ms < 33`，再按 `output_throughput_tok_s` 降序选择。
`strict_sla: true` 筛的是均值，不能代替 P50；`median_itl_ms` 也不能代替 `median_tpot_ms`。
随机搜索 1000 次不保证遍历全部组合或找到全局最优。

此 YAML 已通过配置模型校验，并在 `aiperf` 环境中以 `optimizer.max_trials=32` 完成回放检查：
32 个候选全部成功、0 个 `backend-unsupported`、0 个回放失败，覆盖 5 种并行布局；
每个候选使用 8 张 GPU 并完成 trace 中的 1024 条请求。该检查验证搜索可运行，不代表已找到吞吐最优解或满足 P50。
配置文件保留 1000 次搜索预算。MTP=3 及对应接受率需通过 SDK/底层回放参数接入后重新搜索，
才能用于原始完整实验目标；模型精度与 KVV 误差不由该搜索验收。

并行覆盖也有限：本例固定 PP=1，统一搜索和 trace 回放尚未接入 Prefill CP。
底层 GLM/DSA 成本模型支持 SGLang CP，兼容 `exp` 有 `agg_cp_candidates` / `prefill_cp_candidates`；
不能把该静态估算能力直接当作 trace 中 CP 调度、KV 布局和缓存复用已被建模。
若目标是包含 CP 的联合优化，须先补齐 CP 链路，并把 MTP、接受率和原生调度参数纳入每个候选，
再按上述 P50 条件选取候选；后端全部启动参数并非都已有模拟语义。

## 6. 用 predict 检查固定配置

[case1.trace.mtp.predict.yaml](case1.trace.mtp.predict.yaml) 回放同一份 100K trace，
固定全局 TP8 / Attention DP2 / EP2、并发 32、MTP=3。
`chunked_prefill_size: 32768` 在 DP2 下对应每 rank 16384 token，
`max_prefill_tokens: 458752` 保持每 rank 原值；`max_batched_tokens: 16384` 用于容量估算。
字段语义见 [SGLang prefill budgets](../../../replay/engine/workers.md#sglang-prefill-budgets)。

在仓库根目录运行：

```bash
conda activate aiperf
aisimulate predict \
  -c docs/cli/examples/glm52-b300-mtp/case1.trace.mtp.predict.yaml \
  --capture-per-request \
  --output-dir benchmarks/evidence/glm52-b300-predict/chunk32768
```

对比 chunk=16384 时使用新的输出目录，并同步调整每 rank 的容量估算预算：

```bash
aisimulate predict \
  -c docs/cli/examples/glm52-b300-mtp/case1.trace.mtp.predict.yaml \
  --set engine.workers.aggregated.scheduler.chunked_prefill_size=16384 \
  --set engine.workers.aggregated.scheduler.max_batched_tokens=8192 \
  --capture-per-request \
  --output-dir benchmarks/evidence/glm52-b300-predict/chunk16384
```

输出中的 `prediction.json` 包含汇总和逐请求结果。比较 `median_ttft_ms`、
`median_tpot_ms` 和 `output_throughput_tok_s`；`median_itl_ms` 不是请求 TPOT 的 P50。
本例从冷缓存开始，仍使用 0.5.17 性能数据和原用例的接受草稿数假设 `1.11`，
用于检查配置与回放链路，不代表已验证 0.5.19 的预测精度。

## 7. 自动遍历 predict 配置

工具已集中到 [`predict_tuner`](../../../../examples/sweeper/tools/predict_tuner/)，
使用一份完整的 [`search.yaml`](../../../../examples/sweeper/tools/predict_tuner/search.yaml)。
`prediction` 填普通 predict 配置，`search` 填搜索列表，`output_dir` 填输出目录。
所有相对路径以搜索 YAML 所在目录为基准。

```bash
conda activate aiperf
python examples/sweeper/tools/predict_tuner/tune_predict.py \
  -c examples/sweeper/tools/predict_tuner/search.yaml --dry-run
python examples/sweeper/tools/predict_tuner/tune_predict.py \
  -c examples/sweeper/tools/predict_tuner/search.yaml
```

示例搜索并发 `16,32,64`、chunk `8192,16384,32768,65536`、prefill 预算
`16384,65536,458752`，固定 DP2/EP2。共 36 个组合，3 个预算不兼容的组合记录为无效。
需要扩大搜索时直接修改 YAML：

```yaml
search:
  concurrency: [16, 32, 64, 128]
  dp_ep: [[2, 2], [2, 1], [4, 2], [4, 1], [8, 1]]
  chunked_prefill_size: [16384, 32768, 65536]
  max_prefill_tokens: [65536, 131072, 458752]
  memory_fraction: [0.80, 0.85, 0.90]
  max_sequences: [128, 256]
workload:
  requests_per_concurrency: 4
metrics:
  ttft_scope: first_batch
slo:
  ttft_p50_ms: 15000
  tpot_p50_ms: 33
run:
  jobs: 2
  timeout_seconds: 300
```

总请求数为并发的 4 倍，trace 按原始顺序取前 N 条，不足时报错。每个配置只仿真一次，
TTFT P50 取最初到达的 C 条请求，TPOT P50 和输入、输出、总吞吐使用全部 N 条。
将 `ttft_scope` 改为 `all` 可恢复全量 TTFT；省略 `requests_per_concurrency` 可使用完整 trace。
示例仍沿用原来的 100K 输入 / 90K 共享前缀 / 1K 输出 trace 和 `nextn_accepted: 1.11`。

全部请求完成、token 数和样本数检查通过后，先按所选 TTFT P50 和全量 TPOT P50 严格筛选，
再按输出吞吐降序排列。首批 TTFT 与全量 TPOT 的组合是自定义筛选口径，不是全请求 TTFT SLO。

`metrics.csv` 包含所有候选及五项指标，`ranking.csv` 包含达标排序，`best.yaml` 可直接交给
`aisimulate predict`。逐请求报告、候选配置和日志保存在 `candidates/`，共用的请求切片在
`workloads/`。完整字段表和输出说明见 [Sweeper 示例](../../../../examples/sweeper/README.md#search-configurations-with-the-predict-cli)。

续跑时在原命令末尾加 `--resume`；只有 YAML 中的 `run.jobs` 和 `run.timeout_seconds` 可调整。
配置、请求、统计口径、程序或性能数据变化需要换输出目录。此工具通过当前 Python 环境调用
公开 predict CLI，不运行 GPU 实测，也不校准成本模型。
