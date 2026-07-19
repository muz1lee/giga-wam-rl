# GWP-0.5 × RoboTwin Place Bread Rollout Collection

设计日期：2026-07-19。

## 第一阶段范围

- benchmark：RoboTwin 2.0；
- task：`place_bread_basket`；
- embodiment：`aloha-agilex`；
- actor：公开 Giga-World-Policy-0.5 checkpoint；
- action：模型 16D contract，前 14D 为物理 action，末 2D padding；
- simulator：250 Hz；
- model/data cadence：每 15 simulator steps 一个 14D target，即约 16.67 Hz；
- initial paired seeds：与 50 条 clean scripted demonstrations 对应的 `0..49`。

采集时只运行 action inference。Imagined future 不在 simulator loop 中生成；后续从保存的当前 observation、实际 executed action sequence 与 prompt 离线生成。

## 执行语义

GWP 输出 48 个 16.67 Hz joint targets。第一版每次 commit 前 24 个 target。对每个 target，从当前 14D drive target 线性插值 15 个 250 Hz setpoint，逐步设置双臂 joint drive target 与 gripper，并在第 15 步读取三相机 RGB 和新的 drive target。

因此 raw trajectory 满足：

$$
s_t=q_t^{\mathrm{drive}}
$$

$$
a_t=q_{t+1}^{\mathrm{drive}}
$$

$$
\Delta n_{\mathrm{sim}}=15
$$

没有复用 RoboTwin 原始 `take_action(qpos)` 的可变长 TOPP 重采样，否则新 failure 与成功 demo 又会落在不同 cadence。

## Raw episode contract

每条 episode 是一个独立目录：

```text
episode_seed_00000000/
├── metadata.json
└── trajectory.npz
```

`trajectory.npz` 使用 `allow_pickle=False` 可读的 numeric/fixed-byte arrays：

- `observation_state [N,14]`；
- `cam_{high,left_wrist,right_wrist}_jpeg [N]`；
- `executed_action [N-1,14]`；
- observation 的 simulator step 与 wall time；
- executed action 对应的 replan index / proposal offset；
- 每次 replan 的 raw normalized `[48,16]` proposal；
- 每次 replan 的 denormalized `[48,14]` proposal；
- committed prefix length、policy seed 与 inference wall time。

Raw simulator facts 与后续 imagined future 是不同 artifact；不会把模型生成画面混入 trajectory source of truth。

## Safety defaults

checked-in TOML 默认：

- 1 worker；
- 1 episode；
- 最多执行 2 个 action；
- 不启用 `torch.compile`；
- 不保存 eval MP4；
- 拒绝覆盖同名 run/worker/episode。

这只是闭环 smoke。不要直接把默认参数改成 6 小时长跑。

## Server preparation

学生 RoboTwin 和 assets 始终只读。实际运行使用我们自己的 runtime snapshot：

```text
/home/knowin-wenqian/giga-wam-rl/external/robotwin-runtime
```

它从已跑通的学生 RoboTwin 目录只读复制到我们的 namespace。官方 RoboTwin 对照 revision 固定为：

```text
c3ddfa8b97d5519efa828b075999bd0006778e5e
```

迁移过来的学生 runtime 没有有效 Git `HEAD`，所以第一条 smoke 的 manifest 会明确标为 `student_runtime_snapshot_20260719`；正式实验前要生成 runtime tree manifest，并完成与 pinned official checkout 的差异审计。

## One-episode smoke

GWP-0.5 与 RoboTwin 使用两个已有、互不修改的 Python 环境：模型进程在项目
`.venv` 中运行，simulator 进程在 `fastwam_robotwin` 环境中运行。它们只通过
localhost 上的 length-prefixed、`allow_pickle=False` NPZ 消息通信；不开放远程端口。

终端 1 启动模型服务，只允许处理这次 smoke 的一个 replan 请求：

```bash
cd /home/knowin-wenqian/giga-wam-rl

export GIGA_WAM_RL_ARTIFACT_ROOT=/mnt/nas/wenqian/giga-wam-rl
export TMPDIR="$GIGA_WAM_RL_ARTIFACT_ROOT/tmp"
export HF_HOME="$GIGA_WAM_RL_ARTIFACT_ROOT/cache/huggingface"
export CUDA_VISIBLE_DEVICES=3

PYTHONPATH="src:external/giga-world-policy:external/giga-world-policy/third_party/giga-train:external/giga-world-policy/third_party/giga-datasets" \
  .venv/bin/python -m giga_wam_rl.gwp05_policy_server \
  --port 39500 \
  --device cuda:0 \
  --max-requests 1
```

看到 `policy_server_ready` 后，在终端 2 启动 simulator client：

```bash
cd /home/knowin-wenqian/giga-wam-rl

export GIGA_WAM_RL_ARTIFACT_ROOT=/mnt/nas/wenqian/giga-wam-rl
export TMPDIR="$GIGA_WAM_RL_ARTIFACT_ROOT/tmp"
export CUDA_VISIBLE_DEVICES=3

PYTHONPATH=src \
  /mnt/data/miniconda3/envs/fastwam_robotwin/bin/python \
  -m giga_wam_rl.robotwin_collect_cli \
  --run-id smoke_seed0_two_actions_20260719_rpc1 \
  --worker-id 0 \
  --num-workers 1 \
  --episode-count 1 \
  --max-actions 2 \
  --policy-port 39500
```

## Smoke gate

扩到 16 seeds 前必须验证：

1. checkpoint strict load 且 action output 是 `[48,16]`；
2. 每次执行后的 observation drive target 与 executed action 一致；
3. observation simulator step 为 `[0,15,30,...]`；
4. 三相机 JPEG 可解码、颜色/顺序正确；
5. raw normalized proposal、physical proposal、committed prefix、executed action 都可回溯；
6. episode 能 strict reload，`allow_pickle=False`；
7. 记录模型 inference、simulator、写盘 wall time后再估算 6 小时吞吐。

通过后才运行固定 16-seed calibration；再根据 p50/p90 episode wall time决定 4 worker 长跑与 no-progress termination 参数。
