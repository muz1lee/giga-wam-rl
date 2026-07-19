# GigaWorld-Policy-0.5 代码与接口审计

审计日期：2026-07-19。

## 结论

在写数据转换或 RL 代码之前，必须先固定并检查 GigaWorld-Policy-0.5 的真实实现。公开代码确认了我们的核心方向是可行的：模型的因果结构支持把 action chunk 作为条件预测 future，因此可以补一个 counterfactual future sampler，再用 imagined future 的 reward 更新 action policy。

当前还不能直接训练，原因不是 RL 算法。模型侧的 16D/32D 冲突已经通过真实 strict-load 和 GPU forward 解决，但 physical action 语义仍未闭合：

1. 官方 0.5 finetune/inference 的 32D 路径是遗留接口；固定 checkpoint 只接受 16D。我们的第一版 contract 是 14D physical action/state 补零到 16D。
2. 学生数据的 16D action 是双臂 EEF 表示，而官方 Giga 示例实际使用前 14 维和特定 delta mask。两者只是维数接近，语义并不兼容，不能截断、直接相减或套用官方 normalization。

因此，第一阶段应先完成“固定源码与权重 → checkpoint smoke test → 明确 action adapter → 转换少量样本 → future-only counterfactual rollout”，然后再接 RL。

## 固定的上游版本

版本记录的机器可读来源是 `configs/upstreams.toml`。

| 组件 | 上游 | 固定 revision | 当前状态 |
|---|---|---|---|
| GigaWorld-Policy-0.5 代码 | `open-gigaai/giga-world-policy` | `5d55073a6508de7354c83679d9028f4010ff6cb2` | 本地和服务器均已 clone，detached HEAD，未修改 |
| GigaWorld-Policy-0.5 transformer | `open-gigaai/Giga-World-Policy-0.5` | `4b68e90c0833fec96df456426be344bab64e01a3` | 已下载；manifest、strict-load、joint forward 通过 |
| Wan2.2 TI2V 5B Diffusers base | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | `b8fff7315c768468a5333511427288870b2e9635` | 已下载；VAE encode/decode roundtrip 通过 |

上游 checkout 放在我们的 `external/` 下并被 Git 忽略。我们不会修改它；所有 adapter、sampler 和实验配置都写在自己的 `src/`、`configs/` 与 `scripts/` 下。

## 官方 0.5 的真实输入与训练接口

主要证据：

- `external/giga-world-policy/configs/giga_world_policy_0_5_agilex_finetune.py`
- `external/giga-world-policy/world_action_model/transforms/wa_transforms_lerobot_pretrain.py`
- `external/giga-world-policy/world_action_model/trainer/wa_casual_trainer_mot.py`
- `external/giga-world-policy/world_action_model/models/transformer_wa_casual_mot.py`
- `external/giga-world-policy/scripts/inference_openloop.py`

### 数据 contract

| 字段 | 官方 0.5 行为 | 对我们的约束 |
|---|---|---|
| 数据格式 | LeRobot v3，代码固定 `lerobot==0.4.4` | 学生 v2.1 数据不能直接喂入 |
| action horizon | 48 steps | 转换时要保证 action chunk 与 future frame 时间对齐 |
| future frame offsets | `[0, 12, 24, 36, 48]` | 每个样本共 5 个观测时刻 |
| camera keys | `cam_high`、`cam_left_wrist`、`cam_right_wrist` | 需要显式 camera mapping |
| image layout | 三相机拼成 T layout，最终 `320×384` | 不能只复用单路图像或隐式改变顺序 |
| prompt | T5 embedding，padding/truncation 到 `[64, 4096]` | 除 transformer/base 外还要准备匹配的 T5 资产或缓存 |
| state | 当前时刻 state | 必须定义其物理语义、坐标系和 normalization |
| action | 48-step chunk | 必须先从学生表示映射到选定的 Giga action contract |
| transform 输出 | `fps, images, ref_images, ref_masks, prompt_embeds, action, state, embodiment_id` | 最小 adapter 应逐字段验证 shape/dtype/range |

注意：transform 默认输出的 `fps` 是 16，而 inference loader 的数据集默认值是 30。这里不能把“数据采集/控制 fps”和“模型条件中的 fps”混成同一个未经验证的常数。

### Action contract 的冲突

官方 finetune 配置提供 14 项 delta mask：

```text
[T, T, T, T, T, T, F, T, T, T, T, T, T, F]
```

这表示前 14 个物理维度中，除每只手的 gripper 维外，其余维度使用 `action - state`；随后 transform 将 state/action 补零到 `model_action_dim=32`。

但是，同一配置里的 transformer 是：

```text
in_action_channels=16
out_action_channels=16
```

公开 Hugging Face checkpoint 的 config 也是 16。真实验证确认 checkpoint 关键 action 权重为 16D；strict-load 无 missing/unexpected/mismatched keys，16D joint forward 成功，32D 输入失败。因此 converter 和 inference adapter 必须把 14D physical state/action 补零到 16D，不能使用官方脚本遗留的 32D。

学生现有 16D action 则是：

```text
left xyz + quaternion + gripper + right xyz + quaternion + gripper
```

因此不能做以下操作：

- 因为都是“16D”就直接复用；
- 将 quaternion 分量逐维做 `action - state`；
- 截到 14 维后套官方 mask；
- 使用学生的 normalization stats 代替转换后表示的 stats。

我们需要明确选一个研究 contract：优先考虑从原始 HDF5 的 `joint_action/vector[14]` 对齐官方 AgileX 示例；如果坚持 EEF 16D，则必须写自己的几何 delta、normalization、transformer action head 配置，并验证 checkpoint 中哪些参数可复用。

## 模型是否支持 counterfactual future

支持，但公开 inference 脚本没有把这条能力封装出来。

模型的 causal mask 允许：

```text
action tokens <- 当前 observation/state/reference/action prefix
future tokens <- 当前 observation/state/reference + 给定 action chunk
```

所以它对应的目标正是：

$$
p(x_{t+1:t+H} \mid o_t, s_t, a_{t:t+H-1})
$$

同一个当前状态下替换 action chunk，就可得到 counterfactual future：

$$
\hat{x}^{(k)}_{t+1:t+H} \sim
p_\theta\!\left(x_{t+1:t+H}\mid o_t,s_t,a^{(k)}_{t:t+H-1}\right)
$$

不过公开 `inference_openloop.py` 只实现 action denoising，没有完成 future latent 的生成与 VAE decode。我们需要自己加一个薄的 future-only sampler：固定 clean action 条件，使 action flow time 为 0；只把 future latent 从噪声沿 visual flow time 从 1 积分到 0，最后用固定 Wan VAE 解码。

这属于补齐官方已有建模能力，不是重新训练一套独立 world model。

## 失败轨迹如何进入训练

失败轨迹的核心用途是教 world model：给定某个会失败的 action chunk，未来确实会进入失败状态。此时 action 是条件，不是要模仿的 target。

因此对 failure/counterfactual 数据，第一版采用：

$$
\mathcal{L}_{\text{failure}} =
\mathcal{L}_{\text{future-flow}}
\left(x_{t+1:t+H}\mid o_t,s_t,a_{t:t+H-1}\right)
$$

而不是再加 action imitation loss。否则模型会同时被要求“准确预测失败 future”和“模仿导致失败的 action”，目标互相冲突。

公开 trainer 当前默认总是联合计算 visual flow 与 action flow loss。我们要在自己的 trainer wrapper/config 中显式支持两类 batch：

- demonstration batch：action loss + future loss；
- failure rollout batch：future loss，action 只作为 clean condition。

第一版不引入 delayed-failure credit assignment、长树搜索或多层 value head。样本先严格定义为一个 action chunk 与紧随其后的 future window；只有实验表明确认一个 chunk 的观测不足以区分成功/失败时，再扩 horizon。

## 学生资产的可复用性

| 资产 | 已核实内容 | 结论 |
|---|---|---|
| `place_bread_50demo_raw_sft_241f_20260714` | LeRobot v2.1；parquet 中主要是 16D action，缺少可直接满足 Giga v3 contract 的完整图像/state 字段 | 不能直接训练；可用于核对 action 语义 |
| RoboTwin raw HDF5 | 有多路 RGB、双臂 end pose、gripper 与 `joint_action/vector[14]` | demo converter 的首选只读源 |
| `rollout_review` JSONL | 主要是 transition/review metadata | 可做首批索引，不是完整训练样本本身 |
| rollout `obs_data_*.pt` | 4 个观测、三相机 `240×320`、14D state | 可复用 raw observation，但要重建时间对齐 |
| rollout `actions_*.pt` | shape 类似 `[1, 30, 2, 16, 1]`，是 LingBot 模型空间 tensor | 不能当作已执行的物理 action chunk |
| rollout `latents_*.pt` | Wan/LingBot latent，示例 `[1,48,2,24,20]` | VAE revision、缩放和 layout 未证明一致前不复用 |

失败数据最重要的缺口是“实际执行的 physical actions”。优先找执行日志；若不存在，才考虑严格复现 LingBot postprocess，并必须用几条轨迹做数值和回放验证。图像应优先从 raw RGB 用固定的 Wan2.2 VAE 重编码，避免隐藏的 latent 分布偏移。

学生目录始终只读。转换结果只写到：

```text
/mnt/nas/wenqian/giga-wam-rl/datasets/converted
```

## 对 RL 路线的直接影响

当前公开 fast inference 使用 `torch.no_grad()`、detached prefix KV cache，并在开启梯度时主动报错。这意味着：

- 它适合批量采 K 个 action chunks 和 imagined futures；
- 它不能原样拿来做端到端 ReFL/DRaFT；
- 官方仓库没有 PPO/GRPO、sampler log-prob 或 rollout buffer。

因此第一版 RL 推荐保留两条线，但先做可跑通的一条：

1. 用 action sampler 产生 K 个候选 action chunks；
2. 用 future-only sampler 生成各自的 counterfactual future；
3. 对 future 打 reward，得到组内 advantage；
4. 用 reward/advantage-weighted flow matching 更新 action expert；
5. 保留 BC/flow-matching anchor，防止策略快速漂离 world model 可靠区域。

形式上可以写为：

$$
A_k = R\!\left(\hat{x}^{(k)}_{t+1:t+H}\right)
- \frac{1}{K}\sum_{j=1}^{K}R\!\left(\hat{x}^{(j)}_{t+1:t+H}\right)
$$

$$
\mathcal{L}_{\text{policy}}
= \sum_{k=1}^{K} w(A_k)\,
\mathcal{L}^{(k)}_{\text{action-flow}}
+ \lambda_{\text{BC}}\mathcal{L}_{\text{BC}}
$$

我们之前 FastWAM-RL 的 Flow-GRPO 仍然有用，但用途是复用 rollout/reward/advantage/训练组织方式，不是直接复制模型接口。严格 PPO/GRPO 需要明确的 stochastic sampler 与可计算的 log-prob；当前官方 action Euler sampler 并不提供它。若后面要做 PPO，需要先把采样过程改成有定义的随机过程并验证 log-ratio，而不是先套 PPO 名字。

ReFL/DRaFT 可作为第二阶段 baseline：关掉 no-grad fast path，对有限 denoising steps 反传 future reward。它更接近端到端优化，但显存和时间开销明显更大，而且容易利用 world model/reward model 的误差。

## 接下来的最小闭环

1. 已完成：下载固定 revision 的 transformer 与 Wan2.2 base 到 NAS。
2. 已完成：建立独立 Python 3.11 smoke 环境，不使用训练 launcher 或 `until_completion`。
3. 已完成：checkpoint strict-load、16D joint forward、32D negative test 与 VAE roundtrip。
4. 当前：验证 raw HDF5 14D action 的逐维语义、单位和控制频率；不要与学生 EEF 16D 混用。
5. 只转换 3–5 个 episode/sample，检查 camera order、时间戳、48-step chunk、5 个 future frames、state/action 单位和 normalization。
6. 实现 future-only counterfactual sampler，并验证同一 observation 下不同 action 能产生可区分的 future。
7. 先做 reward ranking 与 world-model calibration；确认 imagined ranking 对真实成功率有预测性后，再接 advantage-weighted flow matching。

在第 6 步之前，写 PPO/GRPO trainer 都属于过早。当前真正的最短路径是先证明“action-conditioned counterfactual future 能生成，而且排序可信”。
