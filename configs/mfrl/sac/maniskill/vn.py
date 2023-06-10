agent_cfg = dict(
    type="SAC",
    batch_size=256,
    gamma=0.95,
    alpha=0.1,
    automatic_alpha_tuning=True,
    ignore_dones=False,
    update_coeff={
        "default": 0.01,
        "(.*?)visual_nn(.*?)": 0.05,
    },
    target_update_interval=2,
    actor_update_interval=2,
    alpha_optim_cfg=dict(type="Adam", lr=1e-3, betas=(0.5, 0.999)),
    shared_backbone=True,
    detach_actor_feature=True,
    actor_cfg=dict(
        type="ContinuousActor",
        head_cfg=dict(
            type="TanhGaussianHead",
            log_std_bound=[-10, 2],
        ),
        nn_cfg=dict(
            type="Visuomotor",
            visual_nn_cfg=dict(
                type="VNPointNet",
                feat_dim="pcd_all_channel",
                mlp_spec=[128, 128, 256],
                out_channels=128,
                n_knn=4,
            ),
            mlp_cfg=dict(
                type="LinearMLP",
                norm_cfg=None,
                mlp_spec=["128 + agent_shape", 1024, 1024, "action_shape * 2"],
                inactivated_output=True,
                # zero_out_indices=slice("action_shape", None, None),
            ),
        ),
        optim_cfg=dict(type="Adam", lr=1e-3, param_cfg={"(.*?)visual_nn(.*?)": None}),
    ),
    critic_cfg=dict(
        type="ContinuousCritic",
        num_heads=2,
        nn_cfg=dict(
            type="Visuomotor",
            visual_nn_cfg=None,
            mlp_cfg=dict(
                type="LinearMLP",
                norm_cfg=None,
                mlp_spec=["128 + agent_shape + action_shape", 1024, 1024, 1],
                inactivated_output=True,
            ),
        ),
        optim_cfg=dict(type="Adam", lr=1e-3),
    ),
)

env_cfg = dict(
    type="gym",
    env_name="OpenCabinetDrawer_1000-v0",
    obs_mode="pointcloud",
    ego_mode=True,
    no_early_stop=True,
)

train_cfg = dict(
    on_policy=False,
    total_steps=500000,
    warm_steps=1000,
    n_steps=2,
    n_updates=1,
    n_eval=100000,
    n_checkpoint=100000,
    exp_logger_cfg=dict(type="aim", log_dir="./"),
)

replay_cfg = dict(
    type="ReplayMemory",
    capacity=100000,
    sampling_cfg=dict(type="OneStepTransition"),
)

rollout_cfg = dict(type="Rollout", num_procs=2)

eval_cfg = dict(
    type="Evaluation",
    num_procs=1,
    num=1,
    use_hidden_state=False,
    save_traj=False,
    save_video=True,
    log_every_step=False,
)
