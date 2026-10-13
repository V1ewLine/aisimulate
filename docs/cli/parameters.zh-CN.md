<!--
SPDX-License-Identifier: Apache-2.0
-->

# AISimulate 中文参数速查

配套[中文使用教程](tutorial.zh-CN.md)。先用教程中的完整配置跑通，再到这里查字段。
表格覆盖日常预测与搜索；特殊场景链接到对应指南。**YAML 中的字段名必须完整匹配，未知字段会报错。**

- [1. 命令行开关](#cli)
- [2. 业务负载 traffic](#traffic)
- [3. 模型与部署 engine](#engine)
- [4. SLA 与搜索控制](#search)
- [5. 搜索范围的写法](#domains)
- [6. aiconfigurator 兼容参数](#legacy)

<a id="cli"></a>

## 1. 命令行开关

`predict` 和 `recommend` 共用：

| 参数 | 默认值 | 含义与用法 |
|---|---|---|
| `-c PATH` / `--config PATH` | 必填 | YAML 输入文件 |
| `--stack NAME` | `engine` | `engine` 使用内置仿真；`dynamo` 使用另行安装的 Dynamo 集成。此项写在命令行 |
| `--set PATH=VALUE` | 无 | 覆盖 YAML 字段，可重复；如 `--set traffic.load.concurrency=16` |
| `--output-dir PATH` | `./aisimulate-output` | 报告输出目录 |
| `--overwrite` | 关闭 | 替换目录中已知的仿真输出，保留无关文件 |
| `--format table\|json` | `table` | 控制标准输出格式；完整报告仍会写到文件 |
| `-h` / `--help` | — | 查看本机版本的命令帮助；YAML 字段另见下面各表 |

只有 `predict` 接受：

| 参数 | 默认值 | 含义与用法 |
|---|---|---|
| `--capture-per-request` | 关闭 | 保存 `requests.jsonl`，用于检查单个请求 |
| `--detail SECTIONS` | 无 | `summary` 总览、`memory` 显存、`time` 时间、`energy` 能耗；逗号分隔，`all` 全选。无相应数据时会显示不可用 |
| `--diagnostics power` | 无 | 保留的功耗诊断入口；新命令优先用 `--detail energy` |
| `--diagnostics-top-n N` | `12` | 能耗诊断每阶段最多展示的操作数，正整数 |
| `--online` | 关闭 | 让仿真按真实时钟推进，需要执行栈支持；普通离线预测无需开启 |

`--set` 的值按 YAML 解析。列表和映射建议用单引号包住整个赋值，例如
`--set 'traffic.load={type: poisson, requests_per_second: 4, seed: 42}'`。
它会替换指定位置的值；不支持 `workers.0` 这样的数组下标，也不支持透传任意推理框架参数。

<a id="traffic"></a>

## 2. 业务负载 `traffic`

### 2.1 输入内容与请求数量

| 路径 | 默认值/取值 | 含义与选择 |
|---|---|---|
| `traffic.source.type` | `synthetic`、`synthetic-session`、`trace` | 固定长度请求、多轮会话、读取已有流量；入门选 `synthetic` |
| `traffic.source.input_tokens` | `1024` | 每个合成请求的输入 token 数；按业务实际长度设置 |
| `traffic.source.output_tokens` | `128` | 每个合成请求的输出 token 数；越长，decode 工作和 KV 占用通常越多 |
| `traffic.stop.requests` | 完全省略 `traffic` 时为 `100` | 合成请求总数；自己写 `traffic` 时显式填写停止条件 |
| `traffic.stop.requests_per_load_unit` | 未设置 | 按负载大小计算请求数；例如并发 16、此值为 10，则总数为 160。与固定 `requests` 二选一 |

完整省略 `traffic` 会使用：输入 1024、输出 128、并发 10、总请求数 100。
显式合成流量必须包含 `source`、`load` 和有效的 `stop`，不会自动补成这份完整默认流量。

### 2.2 压力怎样到达

以下路径以 `traffic.load.` 开头。

| 字段 | 默认值/条件 | 含义与选择 |
|---|---|---|
| `type` | `concurrency` | 还可选 `constant_rate`、`poisson`、`trace_timestamps`；`kv_capacity_fraction` 仅用于搜索 |
| `concurrency` | `10`；仅 `concurrency` | 在途请求上限；多轮会话时表示活跃会话数 |
| `requests_per_second` | 无；单请求速率模式必填 | 每秒到达的请求数，用于 `constant_rate` / `poisson` |
| `seed` | `42`；仅 `poisson` | 固定流量随机种子；与搜索器的 `optimizer.seed` 是两回事 |
| `sessions_per_second` | 无；会话速率模式必填 | 每秒到达的新会话数；后续轮次按会话依赖执行 |
| `speedup` | `1`；仅 `trace_timestamps` | `2` 表示 trace 时间压缩为一半；不能与固定并发同时使用 |
| `fraction` | 无；仅 `kv_capacity_fraction` | 根据候选的 KV 容量换算并发数；`1` 约为容量对应的工作集，允许大于 1，由排队承接超额负载 |

### 2.3 多轮对话与 trace：按需使用

| 路径 | 默认值/条件 | 含义 |
|---|---|---|
| `traffic.source.new_input_tokens_per_turn` | `1024`；会话 | 每轮新增的输入长度 |
| `traffic.source.output_tokens_per_turn` | `128`；会话 | 每轮生成长度 |
| `traffic.source.session.turns` | `4`，至少 `2` | 每个会话的轮数；使用会话时提供 `session` 对象 |
| `traffic.source.session.shared_prefix_ratio` | `0`，范围 `[0,1]` | 共享前缀比例，用于构造可复用的输入 |
| `traffic.source.session.prefix_groups` | `0` | 共享前缀的分组数；共享比例大于 0 时必须为正 |
| `traffic.source.session.inter_turn_delay_ms` | `0` | 一轮完成到下一轮发起的间隔，ms |
| `traffic.stop.sessions` | 会话必需的停止方式之一 | 总会话数；4 轮 × 10 会话会产生 40 个请求。也可用 `sessions_per_load_unit` |
| `traffic.source.paths` | trace 必填 | 流量文件路径列表；除 `dynamo` 格式外只接受一个文件 |
| `traffic.source.format` | `mooncake` | 还支持 `mooncake-delta`、`agentic_mooncake`、`applied_compute_agentic`、`dynamo`、`weka`，约束各不相同 |
| `traffic.source.block_size` | 普通 trace 为 `512` | trace 中前缀块的 token 数；`dynamo` / `weka` 使用文件内信息。不要与引擎的 KV 块大小混淆 |
| `traffic.stop.max_virtual_time_seconds` | 未设置；部分 trace 支持 | 仿真时间上限，秒；不用于合成请求。trace 不设停止条件则运行到流量结束 |

trace 必须符合所选格式，不能直接塞入任意访问日志。Agentic 格式还有执行栈、调度与 `agentic_lanes` 等限制，见[格式对照](user-guide.md#trace-format-compatibility)。

<a id="engine"></a>

## 3. 模型与部署 `engine`

**本节默认值指 `predict`。** `recommend` 的部分省略字段会展开为搜索范围，见[第 5 节](#domains)。

### 3.1 全局配置

| 路径 | 默认值 | 含义与选择 |
|---|---|---|
| `engine.model` | 必填 | Hugging Face 模型 ID 或本地模型配置目录；模型结构和量化信息参与估算 |
| `engine.hardware` | 必填 | 被模拟的硬件，如 `h200_sxm`；名称来自内置或注册的系统配置 |
| `engine.backend` | `vllm` | `vllm`、`sglang`、`trtllm`；选择实际计划使用的框架 |
| `engine.backend_version` | 自动解析 | 选择性能数据版本，不会安装对应框架。比较实验时显式固定，且需有对应数据 |
| `engine.mode` | `aggregated` | `aggregated` 合并服务；`disaggregated` 将 prefill/decode 分开；`afd` 用于 Attention/FFN 分离分析 |
| `engine.context_length` | `max` | 从模型配置解析上下文上限；也可设正整数。它不改变请求实际输入长度 |
| `engine.workers` | 按模式填写 | 合并模式填 `aggregated`；P/D 分离填 `prefill` 和 `decode`；不能只改 `mode` 而保留冲突角色 |

`hardware: auto` 仅在 `recommend` 中可用，且必须同时提供 `optimization.hardware: h200_sxm` 之类的具体型号。
它从该字段取值，并不自动比较各种 GPU 型号。

### 3.2 并行方式：模型怎样分到 GPU 上

下面是 `engine.workers.<角色>.parallelism` 中的字段。`<角色>` 为 `aggregated`、`prefill` 或 `decode`。

| 字段 | 默认值 | 通俗解释 |
|---|---|---|
| `replicas` | `1` | 部署几份 worker 副本，分摊不同请求 |
| `tensor` | `1` | TP：把一份模型的张量计算拆到几张卡；大模型装不下时重点考虑，也会增加通信 |
| `pipeline` | `1` | PP：按模型层分阶段；统一 CLI 尚未验证完整的流水线调度与气泡模拟，依赖 PP 的分析参阅兼容 CLI |
| `attention_data` | `1` | Attention DP：注意力部分按数据分组并行；与复制整套服务的 `replicas` 不同 |
| `moe_tensor` | `1` | MoE 张量并行，拆分专家内计算；普通 dense 模型保持 `1` |
| `moe_expert` | `1` | EP：分布存放和执行 MoE 专家；普通 dense 模型保持 `1` |

普通语言 worker 的 GPU 数：

```text
单个角色的 GPU 数 = replicas × tensor × pipeline × attention_data
P/D 分离的总 GPU 数 = prefill 角色 GPU 数 + decode 角色 GPU 数
```

`moe_tensor` 和 `moe_expert` 描述已有 GPU 内的划分，不再乘一次。
例如其他值均为 1 时，`replicas: 2, tensor: 4` 使用 8 张卡。
并发数由 `traffic.load` 控制，batch 上限由 `scheduler` 控制，增加这些值不会直接增加 GPU 数。

上述六项是统一 CLI 的 worker 并行字段，并不覆盖后端的全部并行能力：

| 其他策略 | 当前 AISimulate 接口边界 |
|---|---|
| Prefill CP（上下文并行） | 统一 YAML 和 Sweeper 的并行搜索空间尚无 CP 维度。底层 AIC 的 GLM/DSA 模型支持 SGLang CP 成本建模；兼容 `exp` 有 `agg_cp_candidates` / `prefill_cp_candidates`，但这不等于已接入 trace 回放 |
| P/D 分离 | `engine.mode: disaggregated`，prefill/decode 可分别选择并行布局和副本数；支持 full-prompt Mooncake trace |
| Attention/FFN 分离（AFD） | `engine.mode: afd`；当前统一 CLI 要求固定长度 synthetic 流量，不能直接使用本例 trace |
| Sequence Parallel | 当前没有独立的统一 YAML 搜索字段 |

默认 MoE 并行池包括 TP、Attention TP + EP（TEP）、Attention DP + EP（DEP）和
Attention DP + MoE TP（DTP）等合法模式，并受模型、后端、显存和 GPU 预算过滤。
`preset: default` 固定 PP=1，也不自动加入 CP。因此“默认搜索的最佳候选”不能解释成
“所有后端并行策略中的最优”。

### 3.3 调度、显存和时间模型

以下路径相对于 `engine.workers.<角色>.`。

| 字段 | 默认值 | 含义与选择 |
|---|---|---|
| `scheduler.max_batched_tokens` | `8192` | 一轮调度允许处理的 token 总预算；不是单个请求的长度上限 |
| `scheduler.max_sequences` | 合并/decode 为 `256`；prefill 为 `1` | 可纳入调度批次的序列数上限；实际 batch 由到达请求和调度决定 |
| `scheduler.prefill_schedule_interval` | `1` | 仅 `predict`、vLLM；每隔 N 次 Attention DP 组调度轮转才接纳 prefill |
| `scheduler.prefill_decode_interval` | `0` | 仅 `predict`、SGLang；每次 EXTEND 后暂停新 prefill/分块续算 N 个调度轮次；轮次不等于输出 token 数 |
| `kv_cache.block_size` | vLLM `64`、SGLang `1`、TRT-LLM `32` | 每个 KV 块容纳多少 token；先使用框架默认值 |
| `kv_cache.prefix_caching` | `true` | 是否允许复用相同前缀的 KV；效果取决于请求是否共享前缀 |
| `kv_cache.bytes_per_token` | `auto` | 每 token 的物理 KV 字节数，按模型和并行布局解析；通常无需手工填写 |
| `kv_cache.capacity.type` | `default` | 自动估算 KV 容量；`fixed` 表示直接指定块数 |
| `kv_cache.capacity.memory_fraction` | vLLM/TRT-LLM `0.9`，SGLang `0.88` | 显存预算比例，范围 `(0,1]`；只用于 `default` 容量 |
| `kv_cache.capacity.blocks` | 未设置 | `fixed` 容量必须填写的 KV 块数，不能同时填 `memory_fraction` |
| `kv_cache.capacity.cuda_graph_reserved_bytes` | `0` | 预留的 CUDA graph 显存字节数；仅 `predict`、`default` 容量 |
| `timing.type` | `default` | 正常预测保持默认；`fixed` 自定耗时，`polynomial` 使用简化时间模型 |
| `timing.forward_model` | `op_level` | 按算子组合耗时；`fpm` 使用完整 forward 的采集数据，要求精确匹配的数据单元 |
| `timing.prefill_ms` / `timing.decode_ms` | 未设置 | `fixed` 时间模型要求同时提供，单位 ms；表示固定阶段计算时间，不是直接指定端到端请求延迟 |
| `startup_seconds` | `0` | worker 启动延迟，秒 |

显存比例的口径随框架变化：vLLM 对应 `gpu_memory_utilization`，SGLang 对应 `mem_fraction_static`，
TRT-LLM 对应剩余显存中的 `free_gpu_memory_fraction`。不能统一理解为“总显存的 90% 全给 KV”。
增大调度上限或显存预算可能提高吞吐，也可能增加排队与资源压力，需要比较预测结果。

**SGLang 特例：** 当前统一 CLI 将 `max_batched_tokens` 传给通用引擎字段及容量估算，
但 SGLang 调度器读取独立的 `sglang.chunked_prefill_size` 和 `sglang.max_prefill_tokens`。
这两个字段尚未暴露到统一 `predict/recommend` YAML；不能把 `max_batched_tokens` 的搜索结果
解释为 chunk 大小或 prefill 调度预算的最优值。底层回放接口可配置它们，联合搜索需要调用方枚举。

### 3.4 仅在 P/D 分离时考虑

| 路径 | 默认值 | 含义 |
|---|---|---|
| `engine.workers.prefill.hardware` / `.decode.hardware` | 继承全局硬件 | 两端可使用不同 GPU；仍共享同一模型、框架和框架版本 |
| `engine.kv_transfer.bandwidth_gb_per_second` | 未设置 | P→D 传输带宽，单位 **GB/s**；不设置则不计传输延迟，应填有效带宽 |
| `engine.kv_transfer.bytes_per_token` | `auto` | 每 token 需要传输的 KV 字节数，与各 worker 的物理缓存占用是不同概念 |
| `engine.kv_transfer.timing_mode` | `destination_missing` | 只计算目标端缺少的 KV；`full_prompt` 按完整输入 KV 计算传输量 |

合并模式不接受 `kv_transfer`。主机/G3 卸载、EPD、AFD 的专用参数见[完整 engine 参考](user-guide.md#engine)。

<a id="search"></a>

## 4. SLA 与搜索控制

### 4.1 延迟要求：`evaluation.sla`

| 字段 | 默认值 | 含义 |
|---|---|---|
| `ttft_ms` | 不限制 | 首 token 延迟上限，ms |
| `itl_ms` | 不限制 | SLA 按请求平均 token 间隔（TPOT）判定；严格候选过滤使用平均 TPOT，ms |
| `e2e_ms` | 不限制 | 总请求延迟上限，ms；与 `ttft_ms` / `itl_ms` 互斥 |

交互式流式输出可用 `ttft_ms` + `itl_ms`；只关心多久完成时用 `e2e_ms`。
常规内置引擎的 goodput 按已完成请求是否达标计数；仅设置阈值不代表 p99 达标。

### 4.2 搜索目标：`optimization.target`

| 值 | 优化方向 | 适合什么需求 |
|---|---|---|
| `throughput`（默认） | 总输出 tokens/s 越高越好 | GPU 预算内追求总吞吐 |
| `throughput_per_gpu` | 输出 tokens/s/GPU 越高越好 | 关注每张卡的产出 |
| `throughput_per_user` | 用户输出速率越高越好 | 关注生成速度 |
| `goodput` | 达到 SLA 的输出 tokens/s 越高越好 | 关注合格业务吞吐，必须配置 SLA |
| `goodput_per_gpu` | 每卡合格输出吞吐越高越好 | 同时关注 SLA 和用卡效率，必须配置 SLA |
| `ttft` | 首 token 平均延迟越低越好 | 尽快开始响应 |
| `e2e_latency` | 请求平均总耗时越低越好 | 尽快完成请求 |
| `pareto` | 同时比较每卡吞吐与每用户吞吐 | 保留互有优劣的方案；文件顺序不表示单一“最好” |

### 4.3 GPU 预算与搜索预算

下面各项只用于 `recommend`。

| 路径 | 默认值 | 含义与选择 |
|---|---|---|
| `optimization.constraints.max_candidate_gpus` | `32` | 每个候选部署的总 GPU 上限；入门时显式设定 |
| `optimization.constraints.min_candidate_gpus` | 不限制下限 | 候选 GPU 数下限；想固定总用卡数时把上下限设为同一个值 |
| `optimization.hardware` | 未设置 | `engine.hardware: auto` 所需的具体硬件型号 |
| `optimization.strict_sla` | `false` | 先按平均延迟过滤再排名，必须有至少一项 SLA；不保证所有请求或 p99 达标 |
| `optimizer.algorithm` | `bayesian` | `bayesian` 利用已有试验结果选择后续候选；`random` 随机采样，适合小范围试跑 |
| `optimizer.max_trials` | `320` | 搜索尝试预算；入门先用较小值，确认配置后再扩大 |
| `optimizer.parallelism` | `16` | 本机同时执行的候选仿真数；资源有限先设 `1`。不是模型 TP，也不是目标 GPU 数 |
| `optimizer.candidate_timeout_seconds` | `600` | 单个候选允许的实际运行时间，秒 |
| `optimizer.seed` | `42` | 搜索随机种子；对比实验时保持一致 |

<a id="domains"></a>

## 5. 搜索范围的写法

`predict` 只接受具体值。`recommend` 的可搜索字段还接受以下形式：

| 写法 | 含义 |
|---|---|
| `tensor: 2` | 固定 TP=2 |
| `tensor: {choices: [1, 2, 4]}` | 从列出的合法值中选择；不能直接写 `tensor: [1, 2, 4]` |
| `replicas: {range: {min: 1, max: 4, step: 1}}` | 从 1 到 4；整数线性范围必须给 `step` |
| `parallelism: {preset: default}` | 自动生成可行并行布局；当前默认布局的 PP 固定为 1 |
| `parallelism: {preset: false, ...}` | 关闭预设，自己指定并行字段；见教程的完整例子 |

不能同时开启 `preset: default` 又填写独立的 `tensor` 等字段。
自定义 `preset` 列表中的每个条目都必须给齐六个并行字段，见[预设规则](user-guide.md#parallelism-preset-behavior)。

`preset: false` 会独立组合各并行字段，再检查可行性；它不会自动把无效组合修正为合法布局。
若随机搜索时出现大量 `backend-unsupported` 并触发 `projection stalled`，优先使用
`preset: default` 或合法的完整 `preset` 列表。

**搜索配置里，“省略”未必等于“固定为预测默认值”：**

- `engine.mode` 默认搜索合并和 P/D 分离；`engine.backend` 默认搜索 vLLM、SGLang。
- `parallelism` 默认启用自动预设；关闭预设后，省略的并行字段仍可能采用默认搜索范围。
- 调度参数也有默认搜索范围，如合并模式的 `max_sequences` 搜索 `256/512/1024`。

想只比较几个因素，就像教程那样固定其他字段。`model`、具体硬件、框架版本、上下文上限、输入/输出长度、SLA 阈值不接受搜索范围；比较不同长度应分别运行。
完整默认范围见[英文参数表](user-guide.md#presets-and-default-ranges)。

<a id="legacy"></a>

## 6. `aiconfigurator` 兼容参数

以下属于 `aiconfigurator cli <模式>`，不要加到 `aisimulate predict` 后面。

| 参数 | 用在哪 | 含义/对应概念 |
|---|---|---|
| `--model-path` / `--model` | 多种模式 | 模型 ID 或本地配置目录；对应 `engine.model` |
| `--system` / `--backend` / `--backend-version` | 估算、搜索等 | 硬件、框架、性能数据版本；建议明确填写。兼容 CLI 默认框架为 `trtllm` |
| `--isl` / `--osl` | `default/recommend/estimate` 等 | 输入/输出 token 长度；默认 `4000/1000`，与统一 CLI 不同 |
| `--ttft` / `--tpot` | `default/recommend` 等 | 首 token / 每输出 token 的延迟目标，ms；默认 `2000/30` |
| `--request-latency` | 搜索等 | 端到端延迟目标，ms；使用此约束时不再单独指定 TTFT/TPOT 约束 |
| `--strict-sla` | 搜索 | 对候选提前执行 SLA 过滤；具体排名语义见兼容指南 |
| `--total-gpus` | `default/generate` | 已有 GPU 预算；最低 GPU 数估算用下面的负载目标 |
| `--target-request-rate` | `recommend` | 目标 requests/s；与 `--target-concurrency` 二选一 |
| `--target-concurrency` | `recommend` | 目标并发数 |
| `--estimate-mode` | `estimate` | `agg`、`disagg`；固定 batch 分析用 `static`、`static_ctx`（只 prefill）、`static_gen`（只 decode） |
| `--batch-size` | 合并/静态 `estimate` | 估算的 batch 工作点；不能直接等同于统一 CLI 的业务并发数 |
| `--tp-size` / `--pp-size` | `estimate` | 明确指定 TP/PP；P/D 分离另有各角色对应参数 |
| `--prefix` | 估算/搜索 | 假设已缓存的输入前缀 token 数，默认 `0`；与“开启前缀缓存”不是同一项设置 |
| `--gemm-quant-mode` / `--kvcache-quant-mode` | 估算/搜索 | 显式指定计算/KV 量化；不填时按模型配置推断，支持范围取决于数据 |
| `--database-mode` | 估算/搜索 | `SILICON` 使用实测数据；`HYBRID` 缺数据时退回理论估算加经验修正；`EMPIRICAL` 全程使用理论估算加经验修正；`SOL` 只用理论估算。默认 `SILICON` |
| `--systems-paths` | 估算/搜索/生成 | 逗号分隔的系统和数据查找目录；`default` 表示内置目录 |
| `--free-gpu-memory-fraction` | 估算/搜索 | 用于显存/KV 容量估算的比例；不填时按框架、版本和硬件解析默认值。TRT-LLM 以剩余显存为基数 |
| `--max-seq-len` | 估算/搜索 | TRT-LLM 每条序列预分配 KV 的长度，默认 `isl + osl`；应对齐实际部署 |
| `--enable-chunked-prefill` | 搜索 | 启用更细粒度的 prefill token 搜索，默认关闭 |
| `--forward-model` | `estimate/default/exp` 等 | `op_level` 或 `fpm`；FPM 要求匹配的完整 forward 数据 |
| `--detail` | `estimate` | `summary,memory,time,energy,source` 或 `all`；`source` 可查数据来源 |
| `--save-dir` / `--top-n` | `default/exp/generate/estimate` | 保存目录 / 保留的 top 配置数；top 数默认 `5`，具体含义随模式变化 |
| `--deployment-target` | 部署生成 | `dynamo-j2`、`dynamo-python`、`llm-d-helm`、`llm-d-kustomize`、`fpm` |
| `--yaml-path` | `exp` | AIC 实验文件，格式不同于统一 CLI 的预测 YAML |
| `--log-level` / `--no-color` | 所有兼容模式 | 日志等级 / 关闭终端颜色；排查问题可用 `--log-level DEBUG` |

兼容 CLI 的参数并非每种模式都通用。运行 `aiconfigurator cli estimate --help` 等对应模式的帮助确认；
更细的量化、MoE、推测解码和实验搜索参数见[兼容 CLI 指南](legacy-aic-user-guide.md)与[高级调优](../../python/aisimulate/docs/advanced_tuning.md)。
