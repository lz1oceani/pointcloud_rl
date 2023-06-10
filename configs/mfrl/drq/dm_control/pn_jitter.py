_base_ = ["./base/pn_base.py"]

agent_cfg = dict(
    obs_aug=dict(
        type="RandomJitterPoints",
        main_key="xyz",
        req_keys=["xyz"],
        jitter_range=[-0.01,0.01],
    ),
)

env_cfg = dict(env_name="dmc_cheetah_run-v0")
