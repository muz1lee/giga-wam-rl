# Giga-WAM-RL Workspace Status

盘点时间：2026-07-19 01:42:24 +08:00。

这是一份时间点快照。GPU 进程、显存和磁盘容量会变化，启动训练前需要重新检查。

## 我们的 workspace

| 类型 | 路径 | 状态 |
|---|---|---|
| 本地 Git 仓库 | `01_wam_rl_research/giga-wam-rl` | `main`；精确版本用 `git rev-parse HEAD` 查询 |
| 服务器代码 | `/home/knowin-wenqian/giga-wam-rl` | 已同步，Git 状态干净 |
| 持久化产物 | `/mnt/nas/wenqian/giga-wam-rl` | 已创建 |

NAS 下使用以下项目目录：

```text
artifacts/
cache/
datasets/converted/
datasets/manifests/
models/
runs/
tmp/
```

## GPU 与运行进程

| GPU | 型号 | 总显存 MiB | 已用 MiB | 可用 MiB | GPU 利用率 |
|---:|---|---:|---:|---:|---:|
| 0 | NVIDIA L20X | 143771 | 25531 | 117637 | 0% |
| 1 | NVIDIA L20X | 143771 | 17988 | 125179 | 0% |
| 2 | NVIDIA L20X | 143771 | 17988 | 125179 | 0% |
| 3 | NVIDIA L20X | 143771 | 17988 | 125179 | 0% |

当前仍有 4 个学生的 `wan_va_server.py`，端口为 39020–39023，分别持有约 18 GiB 显存。GPU 0 还有一个约 7.5 GiB 的宿主机进程；由于容器 PID namespace，无法直接映射到 pod 内 PID。另有一个长期运行的 `make_toast` eval client。

结论：采样瞬间计算利用率为 0%，但显存并非空闲。不要未经确认停止这些服务；正式训练前重新采样并与学生协调。

## 存储

| 挂载点 | 总量 | 已用 | 可用 | 判断 |
|---|---:|---:|---:|---|
| `/` | 2.4T | 397G | 1.9T | 可放代码和可重建环境；代码必须进 Git |
| `/tmp` | 30G | 29G | 0 | 100%，不能使用 |
| `/mnt/data` | 60T | 58T | 3.0T | 96%，不放我们的新产物 |
| `/mnt/nas` | 1.0P | 159T | 866T | 16%，我们的持久化根目录 |

项目 shell/job 应设置：

```bash
export GIGA_WAM_RL_ARTIFACT_ROOT=/mnt/nas/wenqian/giga-wam-rl
export TMPDIR="$GIGA_WAM_RL_ARTIFACT_ROOT/tmp"
export HF_HOME="$GIGA_WAM_RL_ARTIFACT_ROOT/cache/huggingface"
export TORCH_HOME="$GIGA_WAM_RL_ARTIFACT_ROOT/cache/torch"
export WANDB_DIR="$GIGA_WAM_RL_ARTIFACT_ROOT/runs/wandb"
```

## 已确认可读的学生资产

| 名称 | 路径 | 当前用途/限制 |
|---|---|---|
| Place-bread 50 demos | `/mnt/data/wjh/lingbot_datasets/place_bread_50demo_raw_sft_241f_20260714` | 16D action 很有价值；LeRobot v2.1/Wan latent 兼容性待验证 |
| RobotWin eval rollouts | `/mnt/data/wjh/robotwin_eval` | 同时有成功与失败；对齐关系待验证 |
| 96-transition review | `/mnt/data/wjh/rollout_review` | 第一阶段 adapter/sanity check 首选 |
| RoboTwin 50×50 | `/home/wjh/robotwin_clean50x50_official` | 最终 episode schema 和完整性待核验 |
| Place-bread raw HDF5 | `/home/wjh/robotwin_clean50x50_official/workspace/RoboTwin/data/place_bread_basket/demo_clean_fastwam/data` | 首选 demo 转换源；已观察到多相机 RGB 与 14D `joint_action/vector` |
| LingBot RobotWin checkpoint | `/home/wjh/lingbot-va/checkpoints/lingbot-va-posttrain-robotwin` | baseline/数据生成；不是 Giga 0.5 权重 |
| RECAP SigLIP2 | `/mnt/data/wjh/models/recap/siglip2-so400m-patch14-224` | reward/representation 需要时复用 |
| LingBot source | `/home/wjh/lingbot-va` | dirty 且正在服务；只读参考或外部服务 |

服务器检查器确认上述 8 个路径当前均存在。完整登记见 `configs/assets.server.toml`。

## 当前缺失、需要后续补齐

- 当前服务器上的 FastWAM-RL checkout；
- 学生 LingBot latent 与固定 Wan2.2 VAE 的兼容性证明；默认策略仍是从 raw RGB 重编码；
- raw HDF5 14D action 的逐维语义、单位、坐标系和控制频率；
- 针对 96 transitions 的 schema、action frame、camera order、chunk length 验证结果；
- Git 远端。当前本地和服务器 checkout 都有完整 commit 历史，但尚未配置持久化远端。

## GigaWorld-Policy-0.5 后续审计更新

官方代码已 clone 到以下只读参考位置，并固定在 commit `5d55073a6508de7354c83679d9028f4010ff6cb2`：

```text
本地：   01_wam_rl_research/giga-wam-rl/external/giga-world-policy
服务器： /home/knowin-wenqian/giga-wam-rl/external/giga-world-policy
```

两个 checkout 均为 detached HEAD，Git 状态干净；`external/` 不进入我们的仓库。模型 revision 固定在 `configs/upstreams.toml`，权重已下载到：

```text
/mnt/nas/wenqian/giga-wam-rl/models/Giga-World-Policy-0.5
/mnt/nas/wenqian/giga-wam-rl/models/Wan2.2-TI2V-5B-Diffusers
```

独立 smoke 环境位于 `/home/knowin-wenqian/giga-wam-rl/.venv`。Transformer strict-load/joint forward 和 Wan VAE roundtrip 均已通过；验证结果见 `docs/gwp05-smoke-results-2026-07-19.md`，机器可读接口见 `configs/gwp05_contract.toml`。

## 学生目录未受影响的证据

Bootstrap 前后：

- LingBot HEAD 均为 `f98886064082e28387695e6146cebe69e3fc4e25`；
- `git status --porcelain=v1 -uall` 均为 131 行；
- dirty-status SHA-256 均为 `b1fcd57b70acc2180453b2c981a5f009c103c5ef9b06c529465a15ea80afacb1`；
- `/home/wjh/lingbot-va` 目录 mtime 均为 `1784201099`；
- `/mnt/data/wjh` 目录 mtime 均为 `1784215570`。

所有新文件和目录只出现在我们的本地仓库、`/home/knowin-wenqian/giga-wam-rl` 和 `/mnt/nas/wenqian/giga-wam-rl`。
