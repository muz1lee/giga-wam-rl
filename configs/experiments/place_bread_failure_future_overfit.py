import os


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value.strip() == "" else value


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


artifact_root = _env("GIGA_WAM_RL_ARTIFACT_ROOT", "/mnt/nas/wenqian/giga-wam-rl")
data_path = _env(
    "GWP_FAILURE_DATA_PATH",
    os.path.join(
        artifact_root,
        "datasets/converted/place_bread_failure_sparse_pilot_lerobot_v3",
    ),
)
transformer_pretrained = _env(
    "GWP05_TRANSFORMER_PRETRAINED",
    os.path.join(artifact_root, "models/Giga-World-Policy-0.5"),
)
wan_pretrained = _env(
    "GWP_WAN_PRETRAINED",
    os.path.join(artifact_root, "models/Wan2.2-TI2V-5B-Diffusers"),
)
norm_stats_path = _env(
    "GWP_NORM_STATS_PATH",
    os.path.join(
        artifact_root,
        "datasets/manifests/place_bread_gwp05_pilot_norm.json",
    ),
)
t5_embedding_dir = _env("GWP_T5_EMBEDDING_DIR", os.path.join(data_path, "t5_embedding"))
num_frames = 48
view_keys = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]
image_frame_offsets = [0, 12, 24, 36, 48]
selected_raw_indices = [
    int(value)
    for value in _env("GWP_SELECTED_RAW_INDICES", "0,16,32,48,64,80,96,112").split(",")
    if value.strip()
]


config = dict(
    runners=["giga_wam_rl.failure_future_trainer.FailureFutureTrainerMoT"],
    project_dir=_env(
        "GWP_PROJECT_DIR",
        os.path.join(artifact_root, "runs/place_bread_failure_future_overfit"),
    ),
    launch=dict(
        gpu_ids=[
            int(value)
            for value in _env("GWP_GPU_IDS", "0,1,2,3").split(",")
            if value.strip()
        ],
        distributed_type="DEEPSPEED",
        deepspeed_config=dict(deepspeed_config_file="accelerate_configs/zero2.json"),
        until_completion=False,
    ),
    dataloaders=dict(
        train=dict(
            data_or_config=[
                dict(
                    _class_name="ValidWindowWAMLeRobotDataset",
                    data_path=data_path,
                    data_size=None,
                    valid_horizon=num_frames,
                    selected_raw_indices=selected_raw_indices,
                    delta_info={"action": num_frames},
                    delta_frames={key: image_frame_offsets for key in view_keys},
                    video_backend="pyav",
                    t5_load_from="dir",
                    t5_cfg=dict(
                        t5_embedding_dir=t5_embedding_dir,
                        t5_embedding_pattern="episode_{episode_index:06d}.pt",
                        t5_embedding_key="t5_embedding",
                    ),
                    t5_cache_size=64,
                )
            ],
            batch_size_per_gpu=_env_int("GWP_BATCH_SIZE_PER_GPU", 1),
            num_workers=_env_int("GWP_NUM_WORKERS", 2),
            pin_memory=True,
            transform=dict(
                type="DeterministicWALeRobotTransforms",
                dst_size=(320, 384),
                num_frames=num_frames,
                is_train=True,
                norm_path=[{"path": norm_stats_path, "data_paths": [data_path]}],
                robotype_to_embodiment_id={
                    "agilex": 0,
                    "agilex_mobile": 0,
                    "agilex_cobot_magic": 0,
                },
                robotype_default_embodiment_id=0,
                model_action_dim=16,
                delta_mask_by_embodiment_id={
                    "0": [
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        False,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        False,
                    ]
                },
                norm_use_quantiles=True,
                norm_enable_clamp=False,
                num_views=len(view_keys),
                view_keys=view_keys,
                image_cfg=dict(
                    mask_generator=dict(max_ref_frames=1, start=1, factor=4)
                ),
                max_prompt_len=64,
                subtask_prob=0,
            ),
        ),
        test=dict(),
    ),
    models=dict(
        pretrained=wan_pretrained,
        transformer_pretrained=transformer_pretrained,
        strict_load=True,
        transformer=dict(
            added_kv_proj_dim=None,
            attention_head_dim=128,
            cross_attn_norm=True,
            eps=1e-6,
            ffn_dim=14336,
            freq_dim=256,
            image_dim=None,
            in_channels=48,
            num_attention_heads=24,
            num_layers=30,
            out_channels=48,
            patch_size=[1, 2, 2],
            pos_embed_seq_len=None,
            qk_norm="rms_norm_across_heads",
            rope_max_seq_len=1024,
            text_dim=4096,
            action_expert_dim=1024,
            action_ffn_dim=4096,
            in_action_channels=16,
            out_action_channels=16,
            num_embodiments=2,
        ),
        visual_flow_shift=2.0,
        action_flow_shift=5.0,
        expand_timesteps=True,
        enable_gradient_checkpointing=True,
        skip_action_expert=False,
    ),
    optimizers=dict(
        type="CAME8Bit",
        lr=float(_env("GWP_LEARNING_RATE", "5e-6")),
        weight_decay=1e-2,
    ),
    schedulers=dict(type="ConstantScheduler"),
    train=dict(
        resume=False,
        max_epochs=0,
        max_steps=_env_int("GWP_MAX_STEPS", 200),
        gradient_accumulation_steps=_env_int("GWP_GRAD_ACCUM_STEPS", 1),
        mixed_precision="bf16",
        checkpoint_interval=_env_int("GWP_CHECKPOINT_INTERVAL", 100),
        checkpoint_total_limit=2,
        checkpoint_safe_serialization=False,
        checkpoint_strict=False,
        log_with="tensorboard",
        log_interval=1,
        with_ema=False,
        activation_checkpointing=False,
        activation_class_names=["WanAttention"],
    ),
    test=dict(),
)
