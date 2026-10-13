<!--
SPDX-License-Identifier: Apache-2.0
-->

# AISimulate 中文使用教程

AISimulate 用来提前估算：**某个模型放到某种 GPU 上，能有多快、能接多少请求、怎样配置更合适。**
仿真在 CPU 上运行，不需要手里真的有目标 GPU；输出是预测结果。

本文对应当前仓库源码。发布包可能比源码旧，安装与版本差异见[安装说明](../installation.md)。

按顺序读即可跑通一次“预测 → 调整负载 → 搜索配置”。查某个参数时，直接打开[中文参数速查](parameters.zh-CN.md)。

- [1. 选对命令并准备环境](#start)
- [2. 预测一套配置，并读懂结果](#predict)
- [3. 换成自己的业务负载](#workload)
- [4. 自动搜索合适的配置](#recommend)
- [5. 需要最低 GPU 数或部署文件时](#legacy)
- [6. 常见问题与进阶入口](#troubleshooting)

<a id="start"></a>

## 1. 选对命令并准备环境

| 你想知道什么 | 使用入口 |
|---|---|
| “这套配置在我的负载下表现如何？” | `aisimulate predict` |
| “给定 GPU 上限，哪套配置更合适？” | `aisimulate recommend` |
| “满足指定负载和延迟要求，最少要几张卡？” | `aiconfigurator cli recommend`，兼容入口 |
| “生成启动脚本、部署配置” | `aiconfigurator cli generate`，兼容入口 |
| “只估算一个固定 batch 的 prefill/decode 耗时” | `aiconfigurator cli estimate`，兼容入口 |

**两个 `recommend` 的目标不同**：统一 CLI 根据指定指标排序；兼容 CLI 提供最低 GPU 数估算。

已经安装好可以跳过。使用本仓库源码时，在仓库根目录执行：

```bash
# 前提：Python 3.11–3.13、uv、Rust/Cargo，以及 C/C++ 编译和链接工具
uv sync --project python/aisimulate --extra dev
source python/aisimulate/.venv/bin/activate
aisimulate predict --help
aisimulate recommend --help
```

安装会构建 Rust 扩展。仅想安装发布包时，按[安装说明](../installation.md#published-packages)选择适合平台的版本。
以下命令均在保存 YAML 的目录执行；每次运行使用不同输出目录。

<a id="predict"></a>

## 2. 预测一套配置，并读懂结果

### 2.1 保存配置

保存为 `prediction.yaml`：

```yaml
traffic:                         # 要模拟的业务
  source:
    type: synthetic              # 按固定 token 长度构造请求
    input_tokens: 1024            # 每个请求的输入长度
    output_tokens: 128            # 每个请求生成的长度
  load:
    type: concurrency
    concurrency: 8               # 最多保持 8 个请求在途，完成后补下一个
  stop:
    requests: 32                 # 总共发出 32 个请求

engine:                          # 要评估的部署
  mode: aggregated               # 同一组 worker 做 prefill 和 decode
  model: Qwen/Qwen3-32B-FP8
  hardware: h200_sxm             # 被模拟的 GPU 型号
  backend: vllm                  # 被模拟的推理框架
  backend_version: "0.24.0"       # 对应性能数据库版本
  workers:
    aggregated:
      parallelism:
        replicas: 1              # 1 份服务副本
        tensor: 1                # 每份模型用 1 张 GPU
      scheduler:
        max_batched_tokens: 8192  # 一轮调度的 token 预算
        max_sequences: 256       # 调度批次的序列数上限
      kv_cache:
        capacity: {type: default, memory_fraction: 0.9}
```

这里模拟的是 **1 张 H200、8 并发、共 32 个请求**。
token 是模型处理文本的单位，不等于字数；prefill 是处理输入，decode 是逐个生成输出 token。
KV cache 保存已处理 token 的注意力状态，减少重复计算，也会占用显存。

### 2.2 运行

```bash
aisimulate predict -c prediction.yaml \
  --output-dir ./out/predict \
  --capture-per-request \
  --detail summary,memory,time
```

终端显示指标；`out/predict/prediction.json` 保存完整报告，`requests.jsonl` 保存逐请求记录。
`--detail` 增加诊断信息，不改变业务负载。需要 JSON 标准输出时加 `--format json`。

### 2.3 按这个顺序看结果

| 先看什么 | 通俗解释 | 单位/方向 |
|---|---|---|
| 请求数、完成数 | 是否有请求失败或没完成；先确认统计对象完整 | 个 |
| TTFT / Time to first token | 发出请求后，等多久才出现第一个 token，包含排队 | ms，越小越好 |
| ITL / Inter-token latency | 相邻输出 token 的时间间隔，反映生成是否流畅 | ms，越小越好 |
| Request latency / E2E | 从发出请求到生成完毕的总时间 | ms，越小越好 |
| Total output | 整套部署每秒生成多少输出 token | tokens/s，越大越好 |
| Request throughput | 每秒完成多少请求 | requests/s，越大越好 |
| p99 | 该指标约 99% 的样本不超过此值，用来看尾部延迟 | 与原指标相同 |

`wall_time_ms` 是仿真程序在本机运行了多久；吞吐量使用的是仿真的时间区间 `duration_ms`。
32 个请求适合检查流程。正式比较时应增加请求量，并保持模型、输入/输出长度、负载和统计条件一致。

**ITL 与 TPOT 要分清**：ITL 统计每个相邻 token 间隔；TPOT 先算每个请求的平均间隔，再汇总。
两者平均值和 p99 都可能不同。详细口径见[结果解读](understand-your-prediction.md)。

<a id="workload"></a>

## 3. 换成自己的业务负载

先改三个数：**输入长度、输出长度、负载大小**。用 `--set` 临时覆盖 YAML，不用复制整份文件：

```bash
aisimulate predict -c prediction.yaml \
  --set traffic.source.input_tokens=4096 \
  --set traffic.source.output_tokens=512 \
  --set traffic.load.concurrency=16 \
  --set traffic.stop.requests=160 \
  --output-dir ./out/long-input
```

如果你知道业务是“每秒来 4 个请求”，改用请求到达率：

```bash
aisimulate predict -c prediction.yaml \
  --set 'traffic.load={type: constant_rate, requests_per_second: 4}' \
  --set traffic.stop.requests=160 \
  --output-dir ./out/rate-4
```

| 负载方式 | 模拟的行为 | 适合的问题 |
|---|---|---|
| `concurrency` | 在途数量有上限，完成一个才补一个 | “同时 16 个请求，体验怎样？” |
| `constant_rate` | 按固定间隔持续到达，即使服务忙也继续来 | “每秒 4 个请求会不会排队？” |
| `poisson` | 按指定平均速率随机到达 | 观察到达波动的影响 |

`requests_per_second` 是**输入的压力**，实际完成吞吐量要看结果。
切换负载类型时，像上例一样替换整个 `traffic.load`；只修改 `type` 会遗留不兼容的 `concurrency` 字段。
开启 `prefix_caching` 也不代表每个请求自动命中缓存；多轮对话和共享前缀需要相应的业务输入。

<a id="recommend"></a>

## 4. 自动搜索合适的配置

### 4.1 先做一个范围明确的小搜索

保存为 `recommendation.yaml`。它在同样的 8 并发下，比较 TP=1/2、副本数=1/2，最多使用 4 张卡：

```yaml
traffic:
  source: {type: synthetic, input_tokens: 1024, output_tokens: 128}
  load: {type: concurrency, concurrency: 8}
  stop: {requests: 32}

engine:
  mode: aggregated
  model: Qwen/Qwen3-32B-FP8
  hardware: h200_sxm
  backend: vllm
  backend_version: "0.24.0"
  workers:
    aggregated:
      parallelism:
        preset: false            # 手工指定要搜索的维度
        replicas: {choices: [1, 2]}
        tensor: {choices: [1, 2]}
        pipeline: 1
        attention_data: 1
        moe_tensor: 1
        moe_expert: 1
      scheduler: {max_batched_tokens: 8192, max_sequences: 256}
      kv_cache:
        block_size: 64
        capacity: {type: default, memory_fraction: 0.9}

optimization:                    # 什么样的结果算好
  target: throughput_per_gpu     # 每张 GPU 的输出吞吐量
  constraints: {max_candidate_gpus: 4}

optimizer:                       # 搜索花多少计算资源
  algorithm: random
  max_trials: 8
  parallelism: 1
  seed: 42
```

```bash
aisimulate recommend -c recommendation.yaml --output-dir ./out/search
```

搜索完成后看这三个位置：

| 文件 | 用途 |
|---|---|
| `out/search/recommendation.json` | 完整候选记录、指标、失败原因和选中结果 |
| `out/search/recommendation.csv` | 候选记录的表格形式，方便筛选比较 |
| `out/search/recommendations/0001.yaml` | 本例中排名第一的具体配置，可直接再次预测 |

**4 张卡是上限，优胜方案可能只用 1 张。** `max_trials` 是尝试预算，不保证枚举所有组合或找到全局最优。
这里比较的是固定负载；要探索容量，可把并发数改成 `concurrency: {choices: [8, 16, 32, 64]}`。

### 4.2 有延迟要求时，再加 SLA

SLA 是可接受的延迟上限。例如要求首 token 不超过 500 ms、每个请求的平均输出间隔不超过 50 ms：

```bash
aisimulate recommend -c recommendation.yaml \
  --set 'evaluation.sla={ttft_ms: 500, itl_ms: 50}' \
  --set optimization.target=goodput_per_gpu \
  --set optimization.strict_sla=true \
  --output-dir ./out/search-sla
```

`goodput_per_gpu` 只把满足 SLA 的已完成请求计入有效吞吐量，再除以 GPU 数。
`strict_sla: true` 额外按候选方案的**平均延迟**过滤，其中 `itl_ms` 检查平均 TPOT；它不保证 p99 或每个请求达标。
如果只设置 `evaluation.sla`，仍以普通 `throughput` 排序且不开 `strict_sla`，就不会自动执行这一过滤。

### 4.3 把选中的配置再跑一次

若搜索找到了可行配置：

```bash
aisimulate predict -c ./out/search/recommendations/0001.yaml \
  --capture-per-request --detail all \
  --output-dir ./out/best
```

若要查看上一步带 SLA 的搜索结果，把输入路径改为 `./out/search-sla/recommendations/0001.yaml`。
这些 YAML 已经去掉搜索范围，可直接输入 `predict`。生成启动脚本或 Kubernetes 部署文件请用下一节的入口。

<a id="legacy"></a>

## 5. 需要最低 GPU 数或部署文件时

以下兼容命令同样由 AISimulate 包提供。它们使用独立的命令行参数，不能直接照搬上面的 YAML 字段名。

**查模型和硬件覆盖情况：**

```bash
aiconfigurator cli support \
  --model-path Qwen/Qwen3-32B-FP8 --system h200_sxm --backend vllm
```

**估算每秒 4 个请求、满足给定延迟要求所需的最低 GPU 数：**

```bash
aiconfigurator cli recommend \
  --model-path Qwen/Qwen3-32B-FP8 --system h200_sxm \
  --backend vllm --backend-version 0.24.0 \
  --target-request-rate 4 --isl 1024 --osl 128 \
  --ttft 500 --tpot 50 --strict-sla
```

**生成一份基础部署配置：**

```bash
aiconfigurator cli generate \
  --model-path Qwen/Qwen3-32B-FP8 --system h200_sxm \
  --backend vllm --total-gpus 4 \
  --deployment-target dynamo-j2 --save-dir ./out/deployment
```

`generate` 提供起点，不执行完整性能搜索，也不启动 GPU 服务。生成后的部署步骤见[部署指南](../../python/aisimulate/docs/dynamo_deployment_guide.md)。
已有 GPU 预算、想沿用 AIC 的搜索方式时用 `aiconfigurator cli default`；固定 batch 分析用 `estimate`，已有实验 YAML 用 `exp`。
各模式参数含义见[兼容 CLI 参数速查](parameters.zh-CN.md#legacy)。

<a id="troubleshooting"></a>

## 6. 常见问题与进阶入口

| 遇到的问题 | 先检查什么 |
|---|---|
| `command not found` | 是否执行了 `source python/aisimulate/.venv/bin/activate` |
| Rust 扩展 `_runtime` 无法导入 | 是否完整安装/构建了当前源码；只设置 `PYTHONPATH` 不够 |
| 未知字段、参数不兼容 | 拼写、YAML 层级和本机版本；`--set` 也只能使用已有字段 |
| `predict` 拒绝 `choices`、`preset` 或 `optimization` | 搜索配置应交给 `recommend`；预测使用保存下来的具体配置 |
| 输出目录非空 | 换目录，或加 `--overwrite` 替换已有仿真结果 |
| 搜索没有可行结果 | 打开 `recommendation.json` 看失败原因，再核对显存、GPU 上限、SLA、性能数据覆盖 |
| 性能数据/版本不存在 | 核对模型、硬件、框架和 `backend_version`；一个模型被支持不等于所有组合都被支持 |
| 仿真很慢 | 先减少请求量和搜索次数；`optimizer.parallelism` 控制本机同时跑多少候选 |

退出码：`0` 成功；`1` 执行失败或搜索无可行结果；`2` 参数/配置错误；`130` 手动中断。
搜索无可行结果时仍会写报告，但不会产生可供预测的选中 YAML。

需要更多功能时，按场景继续：

| 场景 | 阅读入口 |
|---|---|
| 查参数默认值、取值、怎么选 | [中文参数速查](parameters.zh-CN.md) |
| Prefill/Decode 分离、两端使用不同 GPU | [P/D 示例](migrate-from-aiconfigurator.md#migrate-heterogeneous-pd-hardware-with-sweeper) |
| 多轮会话、共享前缀、真实流量 trace | [流量配置](user-guide.md#traffic) |
| Dynamo 路由与自动扩缩容 | [执行栈](user-guide.md#choose-an-execution-stack)、[Planner 示例](examples/dynamo-planner/README.md)；需匹配的 Dynamo 集成与依赖 |
| KV cache 卸载到主机内存 | [Host offload](user-guide.md#native-vllm-host-offload-prediction) |
| 多模态编码器分离 / Attention 与 FFN 分离 | [EPD](../sweeper/epd.md)、[AFD](../sweeper/afd-topology.md)；两者目前使用分析模型 |
