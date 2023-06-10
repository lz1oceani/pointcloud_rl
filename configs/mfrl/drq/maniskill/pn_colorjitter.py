_base_ = ["./base/pn_base.py"]

agent_cfg = dict(
    obs_aug=dict(
        type="ColorJitterPoints",
        main_key="rgb",
        req_keys=["rgb"],
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.5,
    ),
)

env_cfg = dict(env_name="PushChair_3001-v0")
