# Place Bread Failure-Future Post-training Pilot

验证/设计日期：2026-07-19。

## 结论

停止继续扩大 demo-only checkpoint 的 sampler sweep，直接做监督式 failure-future post-training。第一版不使用 reward、PPO、GRPO 或 outcome-conditioned model；训练信号就是执行 action 后的真实 future RGB。

已有 LingBot rollout 中可恢复 6 条结构完整的失败轨迹，足够做数据管线和 tiny-batch overfit。它们不适合作为论文级的最终 failure 数据，因为采样 cadence 与成功 demonstration 不一致。

## 6 条 failure pilot

来源是 `place_bread_basket_exp016_sft_lora_r8_last4_step1500_16_seed6422_20260717/step1500` 下的 6 条失败 episode：

| seed | worker | source episode | success | abort | planning failures |
|---:|---:|---:|---|---|---:|
| 64230003 | 0 | 3 | false | null | 0 |
| 64240000 | 1 | 0 | false | null | 0 |
| 64240001 | 1 | 1 | false | null | 0 |
| 64240002 | 1 | 2 | false | null | 0 |
| 64260000 | 3 | 0 | false | null | 0 |
| 64260003 | 3 | 3 | false | null | 0 |

每条轨迹都有：

- `obs_data_0.pt, obs_data_2.pt, ..., obs_data_44.pt`，共 23 个 chunk；
- chunk 长度为 `[4] + [8] * 22`，合计 180 个 observation；
- 三相机 RGB，shape 均为 `(240, 320, 3) uint8`；
- `observation.state` 为 `(14,) float64`；
- prompt、seed、failure label 和 rollout 路径来自 sidecar。

6 条轨迹都只有 terminal static suffix：180 帧中最后 6 帧相同。去掉 5 个相邻 terminal duplicate 后，每条为 175 帧，可以产生：

$$
175 - 48 = 127
$$

个完整 48-horizon 起点；6 条合计 762 个窗口。它们高度重叠，因此统计上仍然只有 6 条 trajectory，不能写成 762 条独立 failure。

## Action 与 future 的因果对齐

RoboTwin 的 `get_left_arm_jointState()` / `get_right_arm_jointState()` 读取的是 `joint.get_drive_target()`，不是传感器 real qpos。因此 `obs_data` 里的 14D `observation.state` 是执行到该 keyframe 时的 joint drive target。

转换沿用成功 demo 已验证的 causal convention：

$$
s_t = q^{\mathrm{drive}}_t
$$

$$
a_t = q^{\mathrm{drive}}_{t+1}
$$

对一个训练窗口：

$$
\text{state}=s_t
$$

$$
\text{action chunk}=a_{t:t+47}=q^{\mathrm{drive}}_{t+1:t+48}
$$

$$
\text{visual observations}=x_{t+\{0,12,24,36,48\}}
$$

明确不使用：

- `actions_*.pt`：它是 LingBot 执行前、normalized 的内部 action sample，不是 GWP 14D physical action；
- `latents_*.pt`：它是 LingBot latent，与 GWP/Wan latent contract 不同。

## 最大限制：cadence 不匹配

成功 demo HDF5 来自 dense joint path，每 15 个 simulator low-level control step 保存一次（运动段首尾还会额外保存）。failure `obs_data` 则是每 4 次高层 `take_action(ee)` 保存一次；每个 EE action 内部规划并执行的 low-level step 数量可变，而且没有 timestamp。

所以两者虽然都有同语义的 RGB 和 14D drive target，但不是同一物理时间尺度。现有 failure 数据的合理定位是：

- 可以做 converter、dataloader、loss 和 tiny-batch overfit；
- 可以初步观察模型能否拟合可见的失败后果；
- 不应直接与 demo 做论文级 1:1 混合并声称 dynamics 已严格对齐；
- 正式数据要重新 rollout，并在 simulator low-level control loop 中按与 demo 相同的 cadence 保存 RGB、drive target 和 timestamp。

服务器资产中没有更密的 structured log：planner low-level qpos path 没有落盘，comparison video 也不能恢复三相机 + 14D action 对齐。

## Future-only 训练目标

failure label 只用于选数据和拆分，不输入模型。success/failure 使用同一个 future flow objective。

令 `z_ref` 是当前 observation 的 latent，`z_future` 是真实 future latent；action 与 state 使用冻结的成功 demo normalization。只对 future 加噪：

$$
z_\sigma=(1-\sigma)z_{\mathrm{future}}+\sigma\epsilon
$$

$$
v^\star=\epsilon-z_{\mathrm{future}}
$$

$$
\mathcal L_{\mathrm{future}}
=
\operatorname{MSE}
\left(
v_\theta
\left(
z_\sigma,\sigma
\mid z_{\mathrm{ref}},s_t,a_{t:t+47},c
\right),
v^\star
\right)
$$

token timestep 为：

$$
[0_{\mathrm{state}},0_{\mathrm{ref}},0_{\mathrm{clean\ action}},t_{\mathrm{future}}]
$$

第一版的明确约束：

- action 不加噪；
- action timestep 恒为 0；
- 只对 future latent 加噪；
- 只计算 `future_visual_loss`；
- 不计算 reference reconstruction loss；
- 不计算 action loss；
- action decoder/norm/scale-shift 不在 `future_visual_loss` 的反向路径上，因此没有梯度、不会被 optimizer 更新；它们仍保持 `requires_grad=True`，只为让 DeepSpeed full checkpoint 包含这些参数；
- VAE 仍冻结。

这里的“action 输出 head 参数不更新”不等于原 action policy 的数值行为完全不变。MoT 每层把 action/visual QKV 放在联合 self-attention 中，action token 还能读取 reference token；future loss 更新 visual/shared representations 后，即使 action decoder 参数不变，action output 仍可能漂移。

因此第一版保存两个角色不同的 checkpoint：

- 原始 GWP-0.5 checkpoint：保留为 actor/policy baseline；
- failure post-trained checkpoint：只当 action-conditioned world model 使用，不把它的 action output 宣称为被冻结的原 policy。

如果后续必须在同一个权重对象里保留联合建模，正确接口是 future-only adapter：action policy 推理时关闭 adapter，future imagination 时开启 adapter。仅让 action head 不接 loss，或冻结完整 action expert，都不足以保证 action 输出严格不变。

## 已实现的代码

- `configs/datasets/place_bread_failure_pilot.toml`：固定 6 条 failure 和所有 NAS 输出位置；
- `failure_rollout_probe.py`：以 sidecar `files[].path` 为闭合 provenance，安全、只读加载学生 `.pt`，按 numeric chunk index 拼接并验证 schema；这些 archive path 是指向原 rollout 的绝对 symlink，仍不是自包含副本；
- `failure_lerobot_pilot.py`：转换成 14D state/action + 三相机 LeRobot v3，并逐 episode 校验 parquet 向量和三相机首/末帧；
- `failure_t5_cache.py`：从 Diffusers Wan base 生成每 episode 的 UMT5 embedding；
- `valid_window_dataset.py`：只暴露有完整 48-step future 的起点，避免 episode 尾部 padding 混进训练；
- `deterministic_transform.py`：用确定性的 center crop 代替官方无条件 random crop，使固定窗口 overfit 可解释；
- `future_flow.py`：future-only flow target 与 clean-action timestep；
- `failure_future_trainer.py`：继承官方 MoT trainer，仅覆盖训练目标；
- `place_bread_failure_future_overfit.py`：默认固定第一条 failure 的 8 个显式 start `[0,16,32,48,64,80,96,112]`，使用 16D checkpoint contract、BF16、ZeRO-2、无 EMA，做 200-step overfit。

## 服务器执行顺序

以下命令只读取 `/mnt/data/wjh` 和 `/home/wjh`，所有写入都在 `/mnt/nas/wenqian/giga-wam-rl`。

### 1. Probe

```bash
cd /home/knowin-wenqian/giga-wam-rl

PYTHONPATH=src .venv/bin/python -m giga_wam_rl.failure_rollout_probe \
  --config configs/datasets/place_bread_failure_pilot.toml \
  --registry configs/assets.server.toml
```

### 2. 转换 LeRobot v3

```bash
PYTHONPATH=src .venv-convert/bin/python -m giga_wam_rl.failure_lerobot_pilot \
  --config configs/datasets/place_bread_failure_pilot.toml \
  --registry configs/assets.server.toml
```

### 3. 生成 T5 cache

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src .venv/bin/python \
  -m giga_wam_rl.failure_t5_cache \
  --config configs/datasets/place_bread_failure_pilot.toml \
  --registry configs/assets.server.toml \
  --base-model /mnt/nas/wenqian/giga-wam-rl/models/Wan2.2-TI2V-5B-Diffusers \
  --device cuda:0
```

### 4. 一步训练 smoke

```bash
export GIGA_WAM_RL_ARTIFACT_ROOT=/mnt/nas/wenqian/giga-wam-rl
export HF_HOME="$GIGA_WAM_RL_ARTIFACT_ROOT/cache/huggingface"
export HF_DATASETS_CACHE="$GIGA_WAM_RL_ARTIFACT_ROOT/cache/huggingface/datasets"
export TMPDIR="$GIGA_WAM_RL_ARTIFACT_ROOT/tmp"
export CUDA_VISIBLE_DEVICES=3
export GWP_GPU_IDS=0
export GWP_MAX_STEPS=1
export GWP_NUM_WORKERS=0

PYTHONPATH="src:external/giga-world-policy:external/giga-world-policy/third_party/giga-train:external/giga-world-policy/third_party/giga-datasets" \
  .venv/bin/python external/giga-world-policy/scripts/train.py \
  --config /home/knowin-wenqian/giga-wam-rl/configs/experiments/place_bread_failure_future_overfit.py
```

一步 smoke 必须先确认 `.venv` 同时能 import `diffusers==0.36.0` 与 LeRobot dataset reader；不要为了满足 pip resolver 降级模型环境的 diffusers。

### 5. 200-step overfit

一步 smoke 通过后才清除单卡 `CUDA_VISIBLE_DEVICES`、把 `GWP_GPU_IDS` 改回 `0,1,2,3`、`GWP_MAX_STEPS=200`。config 默认只反复采样 8 个固定窗口，并使用 deterministic center crop；它才是 tiny-batch memorization/correctness test，而不是把 762 个窗口跑约一轮。首次 run 的 `until_completion=False`，确定性 OOM/shape/data 错误会直接停下，避免 launcher 每 10 秒无限重启。此 run 只回答：训练 loss 能否稳定下降、固定训练窗口 decode 后能否更接近 failure GT。它不作为正式 benchmark。

## Overfit 后的最小检查

1. 同一小批窗口上 `future_visual_loss` 明显下降；
2. decode 的 failure future 比原始 checkpoint 更接近该 action 对应的 GT；
3. action 换成另一个窗口的 action 时，prediction 不应完全不变；
4. checkpoint 能 strict reload；
5. 然后再收集 cadence-matched failure，并做 success-only 与 success+failure 两个正式 baseline。

RL 放在 WAM 已能区分 action 对应的 success/failure future 之后。此时 FastWAM-RL 的 PPO/GRPO 基础设施仍可复用，但不是当前 failure-future post-training 的组成部分。
