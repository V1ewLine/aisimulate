<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AISimulate 的架构与运行流程

AISimulate 把一次推理服务拆成三个相互配合的过程：**调度器决定做什么，性能模型计算需要多久，回放器决定何时完成。** 在这三个过程外面，再加上一层候选配置搜索，就构成了 `predict` 和 `recommend` 的主要工作方式。

本文沿着一次模拟的运行过程展开，配合示意图解释各层之间的数据流和状态变化。主线是内置 `engine` stack 下的普通文本请求回放；P/D、Dynamo、EPD 和 AFD 的差异在相应位置说明。图中的时间和容量均为原理示例，不是硬件实测或项目运行结果。

源码基线：`920453e6a6cfb958eae2b2f725db9b767788b4c7`。目录清单及完整参数扩展分析见[代码结构与参数扩展分析](project-structure-and-parameter-roadmap.zh-CN.md)。

## 1. 把部署描述转换成一次模拟任务

一次推理模拟首先需要回答三个问题：用什么部署处理请求，请求怎样到来，以及怎样评价结果。

AISimulate 将这些输入分别放在 `engine`、`traffic` 和 `evaluation` 中。这样，同一组请求可以用于比较不同部署，同一个部署也可以用不同流量进行压力测试。

```text
                    User Configuration
                            |
             +--------------+--------------+
             |              |              |
             v              v              v
          engine         traffic       evaluation
             |              |              |
     model / hardware   token lengths   latency bounds
     backend / workers  arrival times   SLA definition
     parallelism / KV   session / trace
             |              |              |
             +--------------+--------------+
                            |
                            v
                  Validate and Compile
                            |
                            v
                       ReplaySpec
```

配置校验先检查字段类型和组合是否有效。例如，P/D 分离需要 prefill 和 decode 两类 worker；不同后端的专属调度参数也有各自的使用条件。校验通过后，编译层把面向用户的配置转换成 `ReplaySpec`。

`ReplaySpec` 可以理解为“一次模拟的完整任务描述”：包含具体部署、工作负载、评价条件，以及可选 adapter 的配置。它使上层 CLI 和下层执行器之间有了明确的交接点。

例如，用户字段 `scheduler.max_batched_tokens` 会转换为底层引擎参数 `max_num_batched_tokens`。字段在这里完成语义映射，随后才进入实际运行。

对应代码：[配置定义](../python/aisimulate/src/aisimulate/config/)、[`prediction_to_replay_spec`](../python/aisimulate/src/aisimulate/compiler.py)、[Python ReplaySpec](../python/aisimulate/src/aisimulate/sweeper/replay.py)。

## 2. 把模型结构转换成可重复查询的计算计划

有了模拟任务，还需要知道每轮计算的成本。AISimulate 先根据模型配置构建模型结构，再把结构展开成带参数的算子描述。

例如，一个 Transformer 层可以包含归一化、Attention、GEMM 或 MoE，以及并行计算所需的通信。具体结构由模型家族决定，prefill 和 decode 也有不同的算子组合。

```text
 Model Config       Parallelism       Quantization
      |                  |                  |
      +------------------+------------------+
                         |
                         v
                  Python Model Builder
                         |
              +----------+----------+
              |                     |
              v                     v
         Context Ops           Generation Ops
         prefill work          decode work
              |                     |
              +----------+----------+
                         |
                         v
                    EngineSpec
             serialized operation plan
                         |
                         v
                  Rust perfmodel
```

这里的 `EngineSpec` 保存的是模型的计算描述。运行时可以反复把 batch、序列长度、已缓存前缀等动态输入代入这份计划，查询当前这一轮的耗时。

因此，**模型结构的准备和每轮请求状态的变化是分开的**。对一个已经构建好的性能模型，后续迭代可以复用编译计划，不必每轮重新解析整个模型结构。

这个过程也不需要加载完整模型权重去生成文本；权重规模和算子形状用于计算资源需求与性能成本。Python 负责构造描述，Rust `perfmodel` 负责性能数值计算。

对应代码：[`models/`](../python/aisimulate/src/aiconfigurator_core/sdk/models/)、[`compile_engine` 与计划序列化](../python/aisimulate/src/aiconfigurator_core/sdk/engine.py)、[Rust EngineSpec](../crates/core/src/perfmodel/engine/spec.rs)。

## 3. 用性能数据库回答“这一轮需要多久”

模型计划描述了要做哪些计算，但耗时还取决于硬件、后端、版本、精度和当前计算规模。这些信息会共同决定性能模型使用哪些数据。

Collector 在相应硬件和软件环境中采集性能数据，结果主要以 Parquet 表存入仓库。离线模拟读取这些数据，结合算子的计算规则得到耗时及可用的能耗信息。

```text
       Data Preparation                       Offline Simulation

  GPU + Backend + Test Cases                 Compiled Model Plan
              |                                      |
              v                                      |
          Collector                                  |
              |                                      |
              v                                      v
       Performance Tables --------------------> Rust perfmodel
                                                     ^
                                                     |
                                            Current Pass Shape
                                            batch / tokens / KV
                                                     |
                                                     v
                                            Latency + Evidence
```

这里并不是给一个模型查询一次固定的“每 token 延迟”。同一个部署中，prefill 长度、decode batch 和上下文长度不断变化，性能模型会根据当前这一轮的工作量重新求值。

在 `op_level` 路径下，各算子的结果会按重复次数、通信及适用的重叠规则组合。`fpm` 则使用已支持的整次 forward 数据路径。固定耗时和多项式耗时也是可选的 timing provider，适合对应的模拟用途。

```text
                   Scheduler Pass
                         |
                         v
                  TimingProvider
                         |
        +----------------+----------------+
        |                |                |
        v                v                v
   AIC perfmodel     Fixed timing    Polynomial timing
   op_level / fpm
        |                |                |
        +----------------+----------------+
                         |
                         v
                 Modeled Pass Duration
```

TimingProvider 使调度逻辑可以通过统一接口获得时间。数据覆盖和可用的能耗证据仍由具体 provider 决定。

对应代码：[Collector](../python/aisimulate/collector/)、[性能数据库](../crates/core/src/perfmodel/perf_database/)、[算子实现](../crates/core/src/perfmodel/operators/)、[TimingProvider](../crates/core/src/engine/timing.rs)。

## 4. 把请求负载转换成到达事件

部署和性能模型准备好以后，模拟器开始处理工作负载。请求可以来自合成数据，也可以来自 trace。回放需要知道请求的到达时间、输入规模、目标输出规模，以及适用的前缀或会话关系。

```text
 Synthetic Requests     Synthetic Sessions          Trace
         |                      |                     |
         +----------------------+---------------------+
                                |
                                v
                      Load Generation / Admission
                                |
              +-----------------+-----------------+
              |                 |                 |
              v                 v                 v
          Request A         Request B         Request C
          arrival=0         arrival=2         arrival=7
          input=...         input=...         input=...
          output=...        output=...        output=...
                                |
                                v
                      Placement -> Worker Queue
```

负载模式决定请求什么时候进入系统。两种常用模式有不同的控制方式：

```text
Arrival-rate / timestamp mode:

time ----0--------2--------4--------6--------8---->
         A        B        C        D        E
         |
         +-- arrivals follow the configured schedule
             busy workers cause requests to queue

Concurrency mode, limit = 2:

slot 0:  [------ A ------][----- C -----]
slot 1:  [--- B ---][--------- D --------]
                  ^
                  B finishes; a slot admits D
```

到达率模式按外部流量节奏施加请求。并发模式则在释放在途名额后补充请求。即使两次实验处理相同数量、相同长度的请求，它们产生的排队和 batch 也可能不同。

对于多轮会话，并发名额还可以对应整个 session：会话在轮间等待时仍占用名额，直到会话结束。AgentX 工作负载进一步携带任务依赖和分支关系，需要由负载驱动器决定何时释放后续工作。

对应代码：[traffic 配置与转换](../python/aisimulate/src/aisimulate/traffic.py)、[Loadgen](../crates/core/src/replay/loadgen/)、[Admission](../crates/core/src/replay/components/admission.rs)。

## 5. 让调度器决定“这一轮能做哪些工作”

请求到达 worker 后，不会立即变成一个固定 batch。调度器根据当前队列、token 预算、活跃请求数量和 KV 容量，决定这一轮处理什么。

例如，当前可能同时存在已经进入 decode 的请求、等待 prefill 的新请求，以及上次只完成部分 prefill 的长请求。调度策略需要在这些工作之间分配资源。

```text
 Waiting Requests     Running Decode     Partial Prefill
        |                    |                  |
        +--------------------+------------------+
                             |
                             v
                         Scheduler
                             |
              +--------------+--------------+
              |              |              |
              v              v              v
         Token Budget   Sequence Limit   KV Capacity
              |              |              |
              +--------------+--------------+
                             |
                             v
                      Scheduled Pass
              selected requests + token work
                             |
                             v
                       TimingProvider
```

`max_batched_tokens` 限制一轮可处理的 token 工作量，`max_sequences` 限制可同时运行的序列数；KV 容量又提供额外的资源约束。支持的后端可以通过 chunked prefill 将长输入拆到多轮处理，具体选择及 prefill/decode 的组合由后端策略决定。

所以，请求并发、序列上限和实际 batch 是三个不同层次：

```text
Traffic concurrency  -> how much work is in flight
Scheduler limits     -> how much work may run together
Actual pass          -> what the scheduler selects now
```

AISimulate 为 vLLM、SGLang、TRT-LLM 保留相应的调度语义，通过公共 Engine 接口组织执行。共享接口使上层回放可以复用，后端内部的准入、排序和容量策略仍有区别。

对应代码：[Engine 配置](../crates/core/src/engine/config.rs)、[Scheduler](../crates/core/src/engine/scheduler/)、[Generalized Engine](../crates/core/src/engine/generalized/engine.rs)。

## 6. 用 KV 状态把显存限制反馈给调度器

调度器除了判断计算预算，还需要知道请求能否继续占用 KV 空间。AISimulate 会先根据模型、并行配置、硬件和显存策略估算每个 rank 可用的 KV blocks，也允许用户直接指定 blocks。

```text
 Model Weights       Runtime Reservations       GPU Memory Policy
       |                      |                        |
       +----------------------+------------------------+
                              |
                              v
                     KV Capacity Estimation
                              |
                              v
                   num_gpu_blocks, block_size
                              |
                              v
                       KV Block Manager
                              |
                  allocation / reuse / release
                              |
                              v
                          Scheduler
```

从单位上理解，容量估算最终会把可用的 KV 字节量换算成 blocks：

```text
KV block bytes = bytes required by one block's token storage
KV block count = floor(KV memory budget / KV block bytes)
```

不同后端计算可用显存预算的方式并不完全相同，实际预算由容量估算接口决定，上式只表达从字节量到 blocks 的换算关系。

运行期间，模拟器维护 block 的分配、引用和缓存状态。这里保存的是模拟状态，不需要申请与目标服务器同等大小的真实 GPU KV Tensor。

例如，假设有 10 个 blocks，两个请求共享两个完整的前缀块：

```text
Request A:  [P0][P1][P2][P3][P4]
Request B:  [P0][P1][P5][P6]
             ^   ^
             shared prefix

Block pool:
 [P0][P1][P2][P3][P4][P5][P6][free][free][free]

Logical block references: 5 + 4 = 9
Distinct occupied blocks: 7
```

缓存复用既可能减少需要新计算的 prefill，也会影响新增 block 的需求。请求结束后，对应引用被释放；启用前缀缓存时，可复用内容还可能保留为后续请求服务。因此，请求结束不必然意味着相关缓存状态立即消失。

资源不足时，调度器按照后端策略选择等待、分块处理、抢占或其他已支持行为。这就形成了一个反馈关系：

```text
Schedule a pass -> update KV ownership -> available capacity changes
      ^                                             |
      +---------------------------------------------+
                  constrain the next pass
```

对应代码：[容量物化](../python/aisimulate/src/aisimulate/aic.py)、[KV Manager](../crates/core/src/engine/kv_manager/)、[缓存结构](../crates/core/src/engine/cache/)。

## 7. 用虚拟时间推进计算完成

调度器选好工作、性能模型算出耗时后，Replay 就可以确定这一轮什么时候结束。

假设当前虚拟时间为 `10 ms`，本轮计算耗时为 `4 ms`，回放器会安排一个 `14 ms` 的完成事件。模拟程序继续处理事件，无需真实等待这 4 ms。

```text
                       Replay: now = 10 ms
                                |
                                v
                     Engine.execute_pass()
                                |
                                v
                     modeled duration = 4 ms
                                |
                                v
             enqueue PassCompletion at virtual 14 ms
                                |
                                v
                        Pending Event Queue
                     +----------------------+
                     | arrival at 12 ms     |
                     | completion at 14 ms  |
                     +----------------------+
                                |
                                v
                       next time = 12 ms
```

在 `12 ms`，新请求可以到达并排队，但已经提交的那一轮不会因此提前结束或被重新组成 batch。到了 `14 ms`，Engine 的 `complete_pass()` 才发布这一轮的完成效果，例如输出 token 和相应生命周期变化。

这个实现把“一轮已经开始”与“一轮完成效果可见”分开：开始时可以预留资源并提交工作，完成时才让输出等结果对回放系统可见。

```text
              Start Boundary                 Completion Boundary
                     |                               |
                     v                               v
          reserve / commit work ------------> publish pass-end effects
                     |                               |
                     +--------- in flight -----------+
```

如果一个 worker 内部有多个 attention-DP rank，Generalized Engine 会把这组 rank 的本轮完成时间对齐到最慢的 rank：

```text
start = 10 ms

rank 0:  [------ 3 ms ------]              estimated end = 13
rank 1:  [----------- 5 ms -----------]    estimated end = 15
                                          |
                                          v
                                  group completes at 15 ms
```

这样，同组的完成效果具有共同的时间边界。worker 副本之间仍由 Replay 在全局事件时间线上组织。

对应代码：[事件类型](../crates/core/src/replay/event.rs)、[下一时间点选择](../crates/core/src/replay/runtime_utils.rs)、[worker 执行与完成](../crates/core/src/replay/components/engine.rs)、[DP 完成屏障](../crates/core/src/engine/generalized/engine.rs)。

## 8. 用一个小例子串起整个回放过程

为了观察排队如何产生，假设只有一个 aggregated worker，`max_sequences=1`，无前缀复用，每个请求生成 2 个 token。再人为设定 prefill 耗时为 `6 ms`，后续 decode 耗时为 `2 ms`，并将 prefill 完成视为第一个 token 产生的时刻。

请求 A 在 `0 ms` 到达，请求 B 在 `2 ms` 到达。这个例子仅用于说明时序。

```text
Virtual time: 0    2        6    8            14   16 ms
              |    |        |    |             |    |
A arrives ----+             |    |             |    |
B arrives ---------+        |    |             |    |
                   |        |    |             |    |
Worker:       [ A prefill   ][A D][ B prefill  ][B D]
                            |    |             |    |
A tokens:                   t1   t2            |    |
B tokens:                                      t1   t2

B waiting:         [---------]
                   2 ms      8 ms
```

回放器依次处理这些状态变化：

1. `0 ms`：A 到达，调度器准入 A，开始 prefill，安排 `6 ms` 的完成事件。
2. `2 ms`：B 到达。worker 正在执行 A，B 进入等待队列。
3. `6 ms`：A 的 prefill 完成，发布第一个 token。由于序列上限为 1，A 继续占用运行名额，执行 decode。
4. `8 ms`：A 发布第二个 token 并结束。释放请求资源后，B 可以开始 prefill。
5. `14 ms`：B 发布第一个 token，开始 decode。
6. `16 ms`：B 发布第二个 token，回放结束。

由时间线可直接得到：

```text
A: TTFT = 6 - 0 = 6 ms       E2E = 8 - 0 = 8 ms
B: TTFT = 14 - 2 = 12 ms     E2E = 16 - 2 = 14 ms

B queueing time = 8 - 2 = 6 ms
```

两个请求的单独计算成本相同，B 的端到端延迟却更高，原因就是排队。在 AISimulate 中，这个差异来自 Replay、Scheduler 和 TimingProvider 的组合。

增加 worker、改变调度上限或改变到达时间，都会使请求形成不同的执行序列；性能模型再根据实际形成的 pass 计算成本。

## 9. P/D 分离：把一个请求交给两个 worker 池

在 aggregated 模式中，请求在同一类 worker 内完成 prefill 和 decode。P/D 分离则把两个阶段放到不同的 worker 池，中间增加 KV 交接。

```text
Aggregated:

Request -> [ Worker: Prefill -> Decode ] -> Complete

Disaggregated:

Request -> [ Prefill Pool ] -> [ KV Handoff ] -> [ Decode Pool ] -> Complete
                  |                  |                 |
             prefill queue      transfer state    decode queue
             source KV          destination KV    decode progress
```

KV 交接涉及两个角色的资源状态。prefill 侧持有已经计算的 KV，decode 侧需要为接收后的请求建立容量和所有权，传输完成后才能激活目标端。随后源端按照交接协议释放持有状态。

```text
                 Source KV held       Destination KV reserved
                        |                        |
                        +------------+-----------+
                                     |
                                     v
                              Start Transfer
                                     |
                                     v
                         TransferComplete event
                                     |
                                     v
                         Activate Destination
                                     |
                                     v
                            Release Source
```

上图从两端资源都准备好的位置开始。前面的准备顺序由后端决定：当前 vLLM/TRT-LLM 路径采用 source-first，SGLang 路径采用 destination-first。

因此，P/D 的请求时间线会同时受到 prefill 排队、prefill 计算、目标容量等待、KV 传输以及 decode 调度的影响。不能只用两个单点吞吐值推导完整请求延迟。

传输使用配置的带宽与相应 token 字节量计时；`full_prompt` 与 `destination_missing` 决定传输计时使用的 prompt 范围。当前请求级传输机制不代表已经实现完整的共享网络争用或 rank 间 KV 布局转换。

对应代码：[P/D Runtime](../crates/core/src/replay/disagg.rs)、[Handoff 状态机](../crates/core/src/replay/handoff.rs)、[传输契约](../crates/core/src/engine/handoff.rs)。

## 10. 从生命周期事件生成结果

模拟过程中，报告收集器记录请求何时到达、何时被准入、何时产生 token，以及何时结束。最终指标从这些事件中计算出来。

```text
Arrival -> Admission -> First Token -> More Tokens -> Completion
   |                         |                            |
   +--------- TTFT ----------+                            |
   +-------------------- E2E -----------------------------+

                        token 1      token 2      token 3
                           |------------|------------|
                                ITL          ITL
```

对于正常完成且产生 token 的请求，TTFT 是到达到首 token 的时间，E2E 是到达到末 token 的时间。ITL 描述相邻 token 的间隔，TPOT 则可由请求内首末 token 的跨度和间隔数量计算；单 token 请求没有这样的间隔样本。

所有请求再汇总为均值、分位数、吞吐和资源使用等结果。指标使用模拟时间；CLI 自身实际运行了多久属于另外的墙钟时间。

如果配置了 SLA，报告还可以计算 goodput：只累计满足所配置 SLA 的请求及其输出。当前逐请求 ITL 条件使用请求平均 token 间隔，搜索中的 `strict_sla` 又额外检查整体均值，二者需要区分。

```text
                       Request Records
                              |
              +---------------+----------------+
              |                                |
              v                                v
       Latency Distributions                Throughput
       mean / percentiles             tokens / simulated second
              |                                |
              +---------------+----------------+
                              |
                              v
                         ReplayReport
                              |
                   +----------+----------+
                   |                     |
                   v                     v
             prediction.json       requests.jsonl
                                  optional capture
```

功耗证据也可以沿运行链路进入报告，但其可用性取决于 provider 与数据覆盖。当前每 GPU 活动功率不等于整机或机房功率。

对应代码：[TraceCollector 与报告计算](../crates/core/src/replay/report.rs)、[Python 输出](../python/aisimulate/src/aisimulate/output.py)、[功耗结果契约实现](../python/aisimulate/src/aisimulate/power.py)。

## 11. 在回放外面增加搜索循环

前面的流程解决了“给定一个配置，运行结果是什么”。`recommend` 在这个过程外面增加 Sweeper，反复选择候选、执行模拟、评价结果。

```text
 Search Domains + GPU Budget + Workload + Objective
                         |
                         v
                Build Legal Configurations
                         |
                         v
                  Suggest a Candidate <------------------+
                         |                               |
                         v                               |
                Materialize ReplaySpec                   |
                         |                               |
                         v                               |
                    Run Simulation                       |
                         |                               |
                         v                               |
                Metrics / Failure Reason                 |
                         |                               |
                         v                               |
               Feasibility + Score ----------------------+
                         |
                 search budget ends
                         |
                         v
              Ranking / Pareto Selection
                         |
                         v
            Concrete Recommendation YAML
                         |
                         v
                  Run predict again
```

并行参数不能任意拼接。模型结构、后端能力和 GPU 预算会限制合法组合。默认并行搜索会先形成合法配置集合，再把优化器的建议映射到其中一个完整配置。

每个候选随后带着具体参数进入回放。候选构建失败、运行失败、资源约束不满足或超时，会进入相应的失败/不可行记录。重复候选可以复用运行内的结果缓存。

评分可以选择吞吐、每 GPU 吞吐、goodput、延迟等目标。Pareto 模式则保留不同目标之间互不支配的配置，例如单位 GPU 吞吐与单用户速度之间的取舍。当前 Bayesian/random 搜索受 trial 预算约束，不保证穷举整个空间。

最后输出的候选 YAML 已经固定具体值，可以再次交给 `predict`。这让“搜索发现的配置”和“用户重新运行的配置”形成一个可检查的闭环。

对应代码：[推荐入口](../python/aisimulate/src/aisimulate/recommend.py)、[合法并行映射](../python/aisimulate/src/aisimulate/sweeper/parallel_projection.py)、[搜索循环](../python/aisimulate/src/aisimulate/sweeper/search.py)、[评分](../python/aisimulate/src/aisimulate/sweeper/score.py)。

## 12. 各层的职责边界，以及参数如何真正生效

把前面的流程放回整体架构，可以看到两条相交的数据流：一条管理请求和资源状态，另一条提供计算成本。

```text
                           Python Application
                  config / compiler / Sweeper / output
                                    |
                                    v
                                  Runner
                                    |
                                    v
                             Rust Replay
                 virtual clock / placement / lifecycle
                                    |
                                    v
                             Rust Engine
                        scheduler + KV state
                                    |
                            timing request
                                    |
                                    v
                           Rust perfmodel
                       compiled ops + perf data
                                    |
                            duration / evidence
                                    |
                                    +----> completion event in Replay
```

Replay 决定跨请求、跨 worker 的事件顺序；Engine 决定 worker 内本轮做哪些工作并管理资源；perfmodel 计算所选工作的成本。Runner 则负责把上层任务转换为所选执行环境需要的输入。

内置 engine stack 使用轮转放置且没有自动扩缩容策略。Dynamo 集成通过 Runner/配置 adapter 和中立的 placement/scaling 契约接入 Router、Planner；具体策略由集成方提供。由此可以更换请求放置和扩缩容策略，同时保留明确的模拟核心边界。

AFD 和分析式 EPD 具有专用路径。特别是当前 EPD 的 encoder 是分析式组合，不能把普通文本 Replay 的完整事件时间线直接套用到 encoder 排队和 embedding 传输上。旧 `aiconfigurator` 的分析式估算及 Generator 也属于单独的上层工作流。

从这个架构看，扩展一个参数需要完成一条数据路径：

```text
Public YAML
    |
    v
Validation -> Compile / Search Materialization -> Runner -> Runtime
                                                             |
                                                             v
                                                    Behavior and Metrics
                                                             |
                                                             v
                                                      Saved Candidate
                                                             |
                                                             v
                                                        predict again
```

例如，将 `prefill_schedule_interval` 加入推荐搜索，需要让推荐 schema 接受它、搜索器选择它、候选物化传递它、vLLM 调度器消费它，并让保存的 YAML 保留同一个值。当前预测路径已有该字段，推荐 schema 尚未包含它。

新增 KV 参数主要影响容量与资源状态；新增调度参数主要影响 pass 的形成；新增量化参数会影响模型计划、容量和性能查询；新增 P99 SLA 则需要进入指标汇总及可行性判定。先找到参数改变的状态或决策，再贯通其余路径，就能明确实现范围。

对应代码：[Runner 选择](../python/aisimulate/src/aisimulate/stack.py)、[配置 Adapter 契约](../python/aisimulate/src/aisimulate/config_adapter.py)、[Replay Composition](../crates/core/src/replay/replayer.rs)、[EPD 边界](sweeper/epd.md)、[参数扩展建议](project-structure-and-parameter-roadmap.zh-CN.md)。
