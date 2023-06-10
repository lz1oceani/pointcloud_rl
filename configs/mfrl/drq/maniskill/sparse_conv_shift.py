_base_ = ["./base/sparse_conv_base.py"]

agent_cfg = dict(
    obs_aug=dict(
        type="GlobalRotScaleTrans",
        main_key="xyz",
        req_keys=["xyz"],
        rot_range=None,
        scale_ratio_range=None,
        translation_range=[0.1, 0.1, 0.1],
        shift_height=True,
    ),
)

env_cfg = dict(env_name="OpenCabinetDrawer_1000-v0")
