import json
import os.path as osp

def split_and_write(path, output_dir_1, output_dir_2):
    with open(path, 'r') as f:
        d = json.load(f)
    
    keys = list(d.keys())
    keys.sort()
    m = int(len(keys)/2)
    
    d1 = {k: d[k][:25] for k in keys[:m]}
    d2 = {k: d[k][:25] for k in keys[m:]}

    base = osp.basename(path)

    output_path = osp.join(output_dir_1, base)
    with open(output_path, 'w') as f:
        json.dump(d1, f, sort_keys=True)
    output_path = osp.join(output_dir_2, base)
    with open(output_path, 'w') as f:
        json.dump(d2, f, sort_keys=True)

if __name__ == '__main__':
    path_list = [
        'mani_skill/assets/config_files/test_level_indices_arxiv_paper/MoveBucket_test-v0.json',
        'mani_skill/assets/config_files/test_level_indices_arxiv_paper/OpenCabinetDoor_test-v0.json',
        'mani_skill/assets/config_files/test_level_indices_arxiv_paper/OpenCabinetDrawer_test-v0.json',
        'mani_skill/assets/config_files/test_level_indices_arxiv_paper/PushChair_test-v0.json',
    ]
    output_dir_1 = 'mani_skill/assets/config_files/test_level_indices_stage_1_v2'
    output_dir_2 = 'mani_skill/assets/config_files/test_level_indices_stage_2_v2'

    for path in path_list:
        split_and_write(path, output_dir_1, output_dir_2)