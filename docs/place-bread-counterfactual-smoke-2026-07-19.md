# Place Bread × GWP-0.5 Counterfactual Future Smoke

验证日期：2026-07-19。

## 结论

真实 Place Bread 初始观测、真实 demonstration action chunk 和 GigaWorld-Policy-0.5 已经跑通 action-conditioned future-only rollout。固定 reference、state、prompt 和初始 future noise，只把 joint 0 的 48-step target 整体增加 `0.5 rad`，模型生成的 future latent 会稳定改变；零扰动 control 则得到逐元素相同的 latent 和 decoded frames。

这证明当前 checkpoint 和 sampler **确实存在 action → future 的计算路径**，不是静态视频生成器。但第一条 10-step rollout 对 demonstration 的真实 future 拟合很差，肉眼也看不出明确的“成功变失败”语义。当前结果只能称为 structural counterfactual smoke，不能作为模型已经会预测失败、也不能直接作为 RL reward 的证据。

## 运行 contract

### 视觉

源 episode 0、起点 0，三路 `240×320` RGB 使用官方 open-loop inference 的 deterministic center-crop T-layout：

- `cam_high`：上半区 `320×192`；对当前源图等价于裁原始 rows `[24:216]`；
- `cam_left_wrist`：左下 `160×192`；先 resize 到 `256×192`，再裁 cols `[48:208]`；
- `cam_right_wrist`：右下 `160×192`，同样处理；
- composite 输出为 `384×320` RGB，VAE 输入归一化到 `[-1,1]`。

真实稀疏视觉时刻是 `[0,12,24,36,48]`。模型只接收 offset 0 作为 reference；另外四帧仅用于 qualitative/diagnostic comparison。

### State/action

```text
state  = raw_joint_target[0]       # [14]
action = raw_joint_target[1:49]    # [48,14]
```

臂关节维度 `[0..5,7..12]` 使用 `action - state`，夹爪维度 6/13 保持 absolute target。物理 14D 按 pilot q01/q99 归一化后补两个 0，得到 checkpoint 所需的 state `[1,16]` 和 action `[48,16]`；不 clamp。

pilot norm stats：

```text
/mnt/nas/wenqian/giga-wam-rl/datasets/manifests/place_bread_gwp05_pilot_norm.json
SHA-256: 26b41ce86cd8ab47b1da4be6660a6fd3bfda0e71bf0d7eb06ab986d4d310fa9c
```

统计覆盖 3 个 episode、617 个 state 和 29,616 个 H48 action rows；action/state 名称 strict alignment 通过。它只是 smoke stats，正式训练必须用明确 train split 的更大数据重算。

生成命令：

```bash
.venv-convert/bin/python \
  external/giga-world-policy/scripts/compute_wam_task_norm.py \
  --data-root /mnt/nas/wenqian/giga-wam-rl/datasets/converted/place_bread_gwp05_pilot_lerobot_v3 \
  --output /mnt/nas/wenqian/giga-wam-rl/datasets/manifests/place_bread_gwp05_pilot_norm.json \
  --task-name place_bread_gwp05_pilot \
  --model-dim 16 \
  --action-horizon 48 \
  --strict-align \
  --include-metadata
```

### Prompt 与 diffusion

- prompt：`Pick up the bread and place it in the basket.`
- 使用 pinned Wan Diffusers `UMT5EncoderModel`，先按 512 tokens 编码，再截断/零填充为 `[1,64,4096]`；
- state、reference 和 clean action 的 token timestep 均为 0；只有 future latent 使用 visual diffusion timestep；
- visual scheduler 为 `FlowMatchEulerDiscreteScheduler(shift=2.0)`；
- 不使用 CFG；
- 10 个 Euler steps；
- paired samples 共享同一个 seed 7 future noise；
- VAE 逐样本 decode，避免 batch decode 带来的 BF16 数值差异。

## Controls 与结果

### 1. 真实五帧 VAE control

把五张真实 composite 直接走 Wan VAE encode→decode：

```text
latent shape: [1,48,2,24,20]
overall MAE:  1.765 / 255
per offset:   [0.620, 1.300, 2.382, 2.598, 1.925] / 255
peak memory:  3.131 GiB
```

因此后续 rollout 的大误差不能归因于 RGB 方向、T-layout 或 VAE 本身。

### 2. 零扰动 control

逐样本 decode、2-step diagnostic，两个 sample 的 state/action/ref/prompt/noise 完全相同：

| 指标 | 结果 |
|---|---:|
| future latent mean/max absolute difference | `0 / 0` |
| future decoded pixel mean absolute difference | `0 / 255` |
| reference decoded max absolute difference | `0 / 255` |

这排除了 paired batch 和 decode 数值噪声。

### 3. `joint 0 += 0.5 rad` counterfactual

逐样本 decode、10-step 正式 smoke：

| 指标 | 结果 |
|---|---:|
| 扰动维度 | `0` |
| raw target offset | `+0.5 rad` |
| 对该维的 normalized offset | `0.55354` |
| 全 `[48,16]` normalized mean absolute difference | `0.03460` |
| future latent mean absolute difference | `0.04531` |
| future latent max absolute difference | `1.96484` |
| future decoded pixel mean absolute difference | `2.837 / 255` |
| offsets `[12,24,36,48]` pixel difference | `[1.921,3.296,3.127,3.005] / 255` |
| demo rollout vs ground-truth future MAE | `82.505 / 255` |
| paired reference max difference | `0 / 255` |
| peak GPU reserved memory | `14.430 GiB` |
| total wall time（含模型加载与 prompt） | `11.300 s` |

artifact：

```text
/mnt/nas/wenqian/giga-wam-rl/artifacts/counterfactual_smoke/
  place_bread_ep0_t0_joint0_plus0p5_decode1_seed7_steps10/
```

其中 `comparison_contact_sheet.png` 依次展示真实 demo、demo-action generation 和 perturbed-action generation；`conditions.npz` 保存 raw/normalized conditions；`manifest.json` 保存完整参数和指标。

复现命令：

```bash
cd /home/knowin-wenqian/giga-wam-rl
export CUDA_VISIBLE_DEVICES=3
export TMPDIR=/mnt/nas/wenqian/giga-wam-rl/tmp
export HF_HOME=/mnt/nas/wenqian/giga-wam-rl/cache/huggingface
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=src

.venv/bin/python -m giga_wam_rl.gwp05_pilot_counterfactual \
  --norm-stats /mnt/nas/wenqian/giga-wam-rl/datasets/manifests/place_bread_gwp05_pilot_norm.json \
  --output-dir /mnt/nas/wenqian/giga-wam-rl/artifacts/counterfactual_smoke/<fresh-run-name> \
  --device cuda:0 \
  --episode-id 0 \
  --start 0 \
  --action-dimension 0 \
  --additive-offset 0.5 \
  --num-inference-steps 10 \
  --seed 7
```

runner 会拒绝覆盖已有 output directory，并要求我们的项目 checkout 和 pinned external checkout 都是 clean 状态。

## 怎么解释

正面结果是：action condition 没有被模型忽略。相同 noise 下，action 改变导致 latent 和图像改变；零扰动时差异严格为 0。我们设想的“固定当前观测，对不同 action chunk imagine future”在工程和模型接口层面成立。

负面结果同样重要：当前生成的 future 虽然保留了三视图场景、面包和篮子，但 demo action 也没有重现真实机器人运动；`82.5/255` 远高于 VAE control 的 `1.76/255`。而 `+0.5 rad` 的影响肉眼很弱，尚不能解释成失败、碰撞或任务结果改变。

这里还有三个不能混淆的变量：

1. 当前只跑了 10-step visual denoising，尚未比较 25/50 steps；
2. norm stats 只来自 3 条成功 demo，perturbed action 对模型仍可能是 OOD；
3. 公开 post-training trainer 没有显式提供 clean-action、visual-only AC-WM 配置，checkpoint 的这部分训练分布无法从公开代码完整复现。

所以当前判断不是“Giga 不行”，而是“值得继续做 calibration probe，但还不能接 RL”。

## 下一阶段的最小实验

1. 先做 `10/25/50` step sweep，并在 3 个 episode 的若干完整窗口、3 个固定 seeds 上复测 demo fidelity 和 action sensitivity。若 25/50 steps 不明显改善，就停止在 sampler steps 上继续调参。
2. 对少量有物理含义的 action 扰动做 `-offset / demo / +offset`，只看变化是否随方向和幅度稳定，不急着设计复杂 reward。
3. 在 RoboTwin 中真实执行相同 perturbed action chunk，得到 ground-truth counterfactual future/success/failure。核心指标是 imagined ranking 是否和 simulator outcome 一致，而不是生成图是否“看着合理”。
4. 若 public checkpoint 排序不校准，用 success + failure rollout 做 clean-action、future visual-only post-training；不需要改 transformer 架构。
5. 只有当模型能在 held-out perturbations 上区分/排序 failure future 后，再接 action-side RL。FastWAM-RL 里已有的 flow/PPO/GRPO 基础设施可以复用算法实现经验，但不能替代这一步 world-model calibration。

复现入口：`src/giga_wam_rl/gwp05_pilot_counterfactual.py`。机器可读 provenance：`manifests/place_bread_gwp05_counterfactual_smoke.json`。
