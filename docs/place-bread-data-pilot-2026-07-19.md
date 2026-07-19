# Place Bread Raw HDF5 → GWP-0.5 Data Pilot

验证日期：2026-07-19。

## 结论

Place Bread 的 raw HDF5 可以作为 GigaWorld-Policy-0.5 demo post-training 的首批数据源。三条 pilot episode 已转换成 LeRobot v3，并通过 raw vector、视频颜色和 LeRobot temporal query 三层验证。

这批数据只能回答“成功 demonstration 的联合 action/future 建模能否跑通”。它不包含失败轨迹，不能直接支撑 failure-conditioned future 或 RL reward learning。

## Raw 数据事实

- 路径：`/home/wjh/robotwin_clean50x50_official/workspace/RoboTwin/data/place_bread_basket/demo_clean_fastwam/data`。
- 50 个成功 episode，共 11,920 帧；单 episode 长度 160–332。
- 在不跨 episode 且要求完整 48-step action/future 的条件下，共 9,520 个合法起点。
- 图像是 JPEG bytes；三路 GWP 相机映射为：
  - `head_camera → cam_high`
  - `left_camera → cam_left_wrist`
  - `right_camera → cam_right_wrist`
- OpenCV 解码输出必须直接当作 RGB，不能再做 BGR→RGB。原 recorder 把 SAPIEN RGB 直接交给 `cv2.imencode`，使这个来源的通道语义与常见 OpenCV 数据相反。
- `joint_action/vector` 是 14D joint drive target：左臂 6D、左夹爪、右臂 6D、右夹爪。它与四个 component dataset 拼接逐元素一致，误差为 0。

## 因果对齐

raw recorder 保存的是采样时刻的 drive target，不是独立的 next-action 字段。为了让官方 dataloader 的 `delta_info={"action": 48}` 得到真正的未来动作，LeRobot row 使用：

```text
observation.state[t] = raw_joint_target[t]
action[t]            = raw_joint_target[min(t + 1, T - 1)]
images[t]            = raw_images[t]
```

因此从 LeRobot row `t` 查询得到：

```text
state:   raw[t]
actions: raw[t+1 : t+49]
images:  raw[t + {0, 12, 24, 36, 48}]
```

末行 action 重复最后一个 target，只是 episode boundary 的 clamp 值；完整 window 不会使用这个 padding。

## Pilot 输出

源 episode 选择 `[0, 25, 49]`，覆盖数据集前、中、后位置。转换结果：

| 源 episode | LeRobot episode | 帧数 |
|---:|---:|---:|
| 0 | 0 | 228 |
| 25 | 1 | 164 |
| 49 | 2 | 225 |

输出路径：

```text
/mnt/nas/wenqian/giga-wam-rl/datasets/converted/place_bread_gwp05_pilot_lerobot_v3
```

详细生成 manifest：

```text
/mnt/nas/wenqian/giga-wam-rl/datasets/manifests/place_bread_gwp05_pilot_conversion.json
```

LeRobot metadata 为 v3.0、`robot_type=agilex`、`fps=16`；state/action 都保留物理 14D，进入 GWP transform 后才补两个零到 checkpoint 的 16D，不在数据层伪造 16D 物理 action。

## 验证结果

- 三条 episode 的 parquet state/action 与预期 raw arrays 逐元素一致：最大绝对误差均为 0。
- 每条 episode 的末 action 都等于最后一个 raw target，符合显式 boundary contract。
- 9 个“每 episode × 每 camera”的首帧检查中，输出视频相对 source RGB 的 MAE 为 `0.908–1.352`；与 R/B 交换后的 source 比较，MAE 为 `5.389–11.779`。颜色约定通过。
- LeRobot 0.4.4 在三个 episode 起点 `[0, 228, 392]` 均成功返回：
  - state：`(14,)`
  - action：`(48, 14)`
  - 每个 camera：`(5, 3, 240, 320)`

小型 Git provenance manifest 位于 `manifests/place_bread_gwp05_pilot.json`，记录了 NAS manifest 的 SHA-256 和转换代码 revision。

## 仍然存在的限制

1. 这不是严格的 250 Hz action log。simulation 是 250 Hz，raw HDF5 大约每 15 个 simulation step 保存一次；中间 drive setpoints 没有进入该文件。若论文需要“实际 issued action”的严格定义，应读取 `_traj_data/*.pkl` 或重新录制。
2. `fps=16` 是与 GWP 条件匹配的第一版整数近似；名义采样率约为 16.67 Hz，motion segment 边界还可能有额外 capture。当前 pilot 适合接口验证，但在对时间误差敏感的实验中要保留/重建真实 timestamp。
3. 相邻 drive target 或 JPEG 的重复是 recorder 行为，不能全局 deduplicate，否则会改变时间轴。
4. 数据全部是成功 demonstration。下一阶段 counterfactual/failure future 必须从 simulator rollout、受控 action perturbation 或已有 failure rollout 中补充，不能从这 50 条 demo 凭空得到。

## 下一步

future-only sampler 和第一条真实 action perturbation 已完成：模型会随 clean action 改变 imagined future，但 10-step demo prediction 尚未匹配真实 future。下一步仍不要立刻全量转换或接 RL；先按 `docs/place-bread-counterfactual-smoke-2026-07-19.md` 做 sampler-step、窗口、seed 和 simulator ground-truth calibration。
