_base_ = ["./base/pn_base.py"]

agent_cfg = dict(
    obs_aug=dict(
        type="RandomDownSample",
        main_key="xyz",
        req_keys=["xyz", "rgb", "seg"],
        drop_ratio=0.3, 
        fixed_ratio=False
    ),
)

env_cfg = dict(env_name="OpenCabinetDrawer_1000-v0")
