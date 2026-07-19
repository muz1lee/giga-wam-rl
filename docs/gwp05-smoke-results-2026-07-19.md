# GigaWorld-Policy-0.5 Checkpoint/VAE Smoke Results

验证日期：2026-07-19。

## 结论

固定 revision 的 GigaWorld-Policy-0.5 transformer 与 Wan2.2 VAE 均已在服务器完成真实加载和 GPU forward。现在可以确定：

- checkpoint 的 model action contract 是 16D，不是 32D；
- 第一版应采用 14D physical action/state，尾部补零到 16D；
- 五个稀疏时刻的 `384×320` RGB 经 Wan2.2 VAE 编码为两个 latent frames；
- joint transformer 输出包含 reference 和 future 两个 latent frames；第二个 latent 经过时间压缩，承载 `[t+12,t+24,t+36,t+48]` 四个 future observations；
- 当前剩余 blocker 是 14D physical action 的逐维语义、单位和控制频率，而不是模型能否加载。

机器可读 contract 见 `configs/gwp05_contract.toml`。

## 模型与环境

| 项目 | 路径/版本 |
|---|---|
| Transformer | `/mnt/nas/wenqian/giga-wam-rl/models/Giga-World-Policy-0.5` |
| Transformer revision | `4b68e90c0833fec96df456426be344bab64e01a3` |
| Wan2.2 base | `/mnt/nas/wenqian/giga-wam-rl/models/Wan2.2-TI2V-5B-Diffusers` |
| Wan revision | `b8fff7315c768468a5333511427288870b2e9635` |
| Python environment | `/home/knowin-wenqian/giga-wam-rl/.venv` |
| Runtime | Python 3.11.15, torch 2.7.1+cu126, diffusers 0.36.0, transformers 4.54.1 |
| GPU | Physical GPU 3, NVIDIA L20X |

Hugging Face CLI 对两个固定 revision 重新执行 dry-run，结果均为 `0 files`、`0 bytes` 待下载。

## Transformer 验证

CPU-only manifest/header 检查：

```text
shards:             3
tensor count:       1664
tensor bytes:       24086243712
in_action_channels: 16
out_action_channels:16
```

关键权重 shape：

```text
state_encoder.in_proj.weight  [2, 16, 128]
action_encoder.in_proj.weight [2, 16, 128]
action_decoder.out_proj.weight[2, 128, 16]
patch_embedding.weight        [3072, 48, 1, 2, 2]
proj_out.weight               [192, 3072]
```

strict `from_pretrained` 要求 `missing_keys`、`unexpected_keys`、`mismatched_keys` 和 `error_msgs` 全为空，验证通过。

Synthetic joint forward 结果：

```text
state:         [1, 1, 16]
action:        [1, 48, 16]
ref_latents:   [1, 48, 1, 24, 20]
future_noise:  [1, 48, 1, 24, 20]
prompt_embeds: [1, 64, 4096]

visual output: [1, 48, 2, 24, 20]  # reference + future
future slice:  [1, 48, 1, 24, 20]
action output: [1, 48, 16]
```

所有输出 finite；32D state/action 在 action input projection 处按预期触发 `RuntimeError`。峰值 GPU reserved memory 为 11.512 GiB。

这说明官方 finetune/inference 中遗留的 `model_action_dim=32`、`action_dim=32` 不能用于固定的 0.5 checkpoint。我们的代码不修改 external，而是在自己的 adapter 中执行：

$$
a^{14D}_{\text{physical}}
\xrightarrow{\text{pad zeros}}
a^{16D}_{\text{model}}
$$

## Wan VAE 验证

固定 VAE config：

```text
z_dim:                 48
scale_factor_spatial:  16
scale_factor_temporal: 4
latents_mean/std:      48 values each
```

真实 encode/decode roundtrip：

```text
RGB input:       [1, 3, 5, 384, 320]
raw latent:      [1, 48, 2, 24, 20]
single-frame:    [1, 48, 1, 24, 20]
reference:       [1, 48, 1, 24, 20]
future:          [1, 48, 1, 24, 20]
reconstruction:  [1, 3, 5, 384, 320]
postprocessed:   [1, 5, 3, 384, 320]
```

VAE 输入与 decode 输出范围为 `[-1,1]`；postprocess 后范围为 `[0,1]`。注意这里的五帧是 offsets `[0,12,24,36,48]`，不是连续视频帧。future sampler 只需要生成第二个 latent time position，不应错误地生成四个或五个 future latent positions。

归一化及其逆变换为：

$$
z_{\text{model}} = \frac{z_{\text{raw}}-\mu}{\sigma}
$$

$$
z_{\text{raw}} = z_{\text{model}}\sigma + \mu
$$

roundtrip 数值可逆且输出 finite；峰值 GPU reserved memory 为 3.131 GiB。Diffusers 0.36 提示 Wan config 的额外字段 `clip_output=false` 会被忽略，但不影响权重加载、shape 或 encode/decode。

## 复现命令

```bash
cd /home/knowin-wenqian/giga-wam-rl
export TMPDIR=/mnt/nas/wenqian/giga-wam-rl/tmp
export HF_HOME=/mnt/nas/wenqian/giga-wam-rl/cache/huggingface
export CUDA_VISIBLE_DEVICES=3

PYTHONPATH=src .venv/bin/python -m giga_wam_rl.gwp05_smoke \
  --checkpoint /mnt/nas/wenqian/giga-wam-rl/models/Giga-World-Policy-0.5 \
  --device cuda:0

PYTHONPATH=src .venv/bin/python -m giga_wam_rl.gwp05_vae_smoke \
  --base-model /mnt/nas/wenqian/giga-wam-rl/models/Wan2.2-TI2V-5B-Diffusers \
  --device cuda:0
```

这些 smoke tests 不加载数据、不训练、不使用 `torch.compile`，也不启动或停止学生服务。
