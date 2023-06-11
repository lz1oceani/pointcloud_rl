import sapien_rl.env
import matplotlib.pyplot as plt
import gym, numpy as np, time
np.random.seed(0)


env = gym.make('OpenCabinet_state_medium-v1')
for i in range(99):
    env.reset(level=1)

