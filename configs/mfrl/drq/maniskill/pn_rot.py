_base_ = ["./base/pn_base.py"]

agent_cfg = dict(
    obs_aug=dict(
        type="GlobalRotScaleTrans",
        main_key="xyz",
        req_keys=["xyz"],
        rot_range=[-0.15, 0.15],
        scale_ratio_range=None,
        translation_range=None,
        shift_height=False,
    ),
)

