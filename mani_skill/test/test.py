import gym
from sapien_rl.env.open_cabinet import OpenCabinetEnv
from sapien_rl.env.sapien_env import _renderer, _engine
import time

total_step_time = 0
total_step_count = 0


def step(env, action):
    global total_step_time
    global total_step_count
    start = time.time()
    env.step(action)
    total_step_time += time.time() - start
    total_step_count += 1
    print("average FPS:", total_step_count / total_step_time)


env = OpenCabinetEnv(variant_config={"partnet_mobility_id": "44817"})
for i in range(1000):
    action = env.action_space.sample()
    step(env, action)
    env.render("human")
