import gym
import mani_skill.env
import time
import numpy as np

import pathlib, json, os

_this_file = pathlib.Path(__file__).resolve()

# from mani_skill.utils.misc import get_model_ids_from_yaml
from collections import defaultdict

def find_level_indices(
    task_name,
    n_total_levels=125,
    suffix=None,
    first_half=False,
    second_half=False,
):
    d = defaultdict(list)
    
    env = gym.make(task_name)
    all_model_ids = env.unwrapped.get_all_model_ids()
    all_model_ids = [int(x) for x in all_model_ids]
    if first_half:
        all_model_ids = all_model_ids[:len(all_model_ids)//2]
    elif second_half:
        all_model_ids = all_model_ids[len(all_model_ids)//2:]
    else:
        all_model_ids = all_model_ids

    n_obj = len(all_model_ids)
    n_levels_per_obj = int(np.ceil(n_total_levels / n_obj))
    
    level_cnt = 0
    while True:
        level_idx = np.random.randint(0, int(1e9))
        env.reset(level=level_idx)
        obj_id = env.level_config['layout']['articulations'][0]['partnet_mobility_id']
        if obj_id in all_model_ids and len(d[obj_id]) < n_levels_per_obj:
            d[obj_id].append(level_idx)
            level_cnt += 1
            if level_cnt >= n_levels_per_obj * n_obj:
                break

    n_to_pop = n_levels_per_obj * n_obj - n_total_levels
    # import pdb; pdb.set_trace()
    for obj_id in all_model_ids[:n_to_pop]:
        d[obj_id].pop()
    
    output_dir = str(_this_file.parent.joinpath('../assets/config_files/test_level_indices'))
    if suffix is not None:
        output_dir += '_' + suffix
    os.makedirs(output_dir, exist_ok=True)

    path = os.path.join(output_dir, '{:s}.json'.format(task_name))
    with open(path, 'w') as f:
        # json.dump(d, f, sort_keys=True, indent=4)
        json.dump(d, f, sort_keys=True)

if __name__ == '__main__':
    # find_level_indices('MoveBucket_test-v0')
    # find_level_indices('PushChair_test-v0')
    # find_level_indices('OpenCabinetDoor_test-v0')
    # find_level_indices('OpenCabinetDrawer_test-v0')

    # find_level_indices('MoveBucket-v0')
    # find_level_indices('PushChair-v0')
    # find_level_indices('OpenCabinetDoor-v0')
    # find_level_indices('OpenCabinetDrawer-v0')

    # find_level_indices('MoveBucket_test-v0', suffix='stage_1', first_half=True)
    # find_level_indices('PushChair_test-v0', suffix='stage_1', first_half=True)
    # find_level_indices('OpenCabinetDoor_test-v0', suffix='stage_1', first_half=True)
    # find_level_indices('OpenCabinetDrawer_test-v0', suffix='stage_1', first_half=True)

    # find_level_indices('MoveBucket-v0', suffix='stage_1')
    # find_level_indices('PushChair-v0', suffix='stage_1')
    # find_level_indices('OpenCabinetDoor-v0', suffix='stage_1')
    # find_level_indices('OpenCabinetDrawer-v0', suffix='stage_1')

    # find_level_indices('MoveBucket_test-v0', suffix='stage_2', second_half=True)
    # find_level_indices('PushChair_test-v0', suffix='stage_2', second_half=True)
    # find_level_indices('OpenCabinetDoor_test-v0', suffix='stage_2', second_half=True)
    # find_level_indices('OpenCabinetDrawer_test-v0', suffix='stage_2', second_half=True)

    find_level_indices('MoveBucket-v0', suffix='stage_2')
    find_level_indices('PushChair-v0', suffix='stage_2')
    find_level_indices('OpenCabinetDoor-v0', suffix='stage_2')
    find_level_indices('OpenCabinetDrawer-v0', suffix='stage_2')