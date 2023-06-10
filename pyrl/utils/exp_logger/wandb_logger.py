import wandb, numpy as np, os.path as osp
import matplotlib.pyplot as plt
from copy import deepcopy

from .base_logger import BaseLogger
from ..data import is_num
from .builder import EXP_LOGGER


@EXP_LOGGER.register_module(name="wandb")
@EXP_LOGGER.register_module()
class WandbLogger(BaseLogger):
    """
    id: uid for resume
    """

    def __init__(self, work_dir: str, log_dir: str = None, timestamp=None, clean_up=False, **kwargs):
        super().__init__(work_dir, timestamp)
        if "key" in kwargs:
            wandb.login(key=kwargs.pop("key"))

        if clean_up and "name" in kwargs:
            from wandb.apis.public import Runs, Api

            kwargs_for_api = deepcopy(kwargs)

            exp_id = kwargs_for_api.pop("id", None)
            name = kwargs_for_api.pop("name")
            assert exp_id is None or name is None, "You can only provide one of id and name!"

            if name is not None:
                api = Api(kwargs_for_api)
                if "entity" not in kwargs_for_api:
                    kwargs_for_api["entity"] = api.default_entity
                for key in list(kwargs_for_api.keys()):
                    if key not in ["entity", "project"]:
                        kwargs_for_api.pop(key)
                runs = Runs(client=api._client, **kwargs_for_api)

                for run in runs:
                    if run.name == name:
                        run.delete()
                        print("Delete duplicate run", run)
        wandb.init(dir=work_dir, **kwargs)

    def log(self, tags, n_iter, tag_name="train"):
        ret_dict = dict()
        tags = self.get_loggable_tags(tags, tag_name=tag_name)
        self.log_to_csv(tags, n_iter)
        for tag, val in tags.items():
            if is_num(val) or val.size == 1 or isinstance(val, str):
                ret_dict[tag] = val
            elif isinstance(val, np.ndarray) and val.ndim == 1:
                ret_dict[tag] = np.linalg.norm(val)
            else:
                if val.ndim == 2:
                    cmap = plt.get_cmap("jet")
                    val = cmap(val)[..., :3]
                assert val.ndim == 3, f"Image should have two dimension! You provide: {tag, val.shape}!"
                ret_dict[tag] = wandb.Image(val)
        wandb.log(ret_dict, step=n_iter, commit=True)
