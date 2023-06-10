from wandb.apis.public import Runs, Api
import wandb, re, numpy as np


def wandb_get_runs(project, entity=None, key=None, per_page=50):
    if key is not None:
        wandb.login(key=key)
    api = Api()
    if entity is None:
        entity = api.default_entity
    runs = Runs(api.client, project=project, entity=entity, per_page=per_page)
    return runs


def wandb_filter_runs(runs, group_name=None, name=None):
    ret = []
    for run in runs:
        if group_name is not None and re.match(group_name.lower().replace("-", "_"), run.group.lower().replace("-", "_")) is None:
            continue
        if name is not None and re.match(name.lower().replace("-", "_"), run.name.lower().replace("-", "_")) is None:
            continue
        ret.append(run)
    return ret


def get_project_logs(entity, project, save_name=None, run_name_process=lambda _: _):
    runs = wandb_get_runs(entity=entity, project=project, per_page=1000)
    history = {}
    for run_i in runs:
        run_name = run_name_process(run_i.name)
        run_name_split = run_name.split("-")
        run_state = str(run_i._state)

        run_info = run_i.history(keys=["env/rewards_mean", "_step"])

        if "_step" not in run_info and run_state != "running":
            continue
        step = np.array(run_info["_step"])
        reward = run_info["env/rewards_mean"]
        if run_name in history:
            if run_state == "running" or (np.max(history[run_name][0]) < np.max(step) and history[run_name][2] != "running"):
                history[run_name] = [step, reward, run_state, run_name_split[-1]]
        else:
            history[run_name] = [step, reward, run_state, run_name_split[-1]]
    if save_name is not None:
        from pyrl.utils.file import load, dump

        dump(history, save_name)
    return history


if __name__ == "__main__":
    runs = wandb_get_runs("ICML_2023_Main", entity="icml_2023_3d_svea")
    runs = wandb_filter_runs(runs, group_name="DMC_dmc_ball_in_cup_catch-v0_SAC_CNN")
    for run in runs:
        print(run.name, run.group)
