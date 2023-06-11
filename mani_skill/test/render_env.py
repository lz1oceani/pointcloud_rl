import sapien_rl.env
import matplotlib.pyplot as plt
import gym, numpy as np

env = gym.make('OpenCabinet_state_medium-v1')
env.reset(level=2)


while True:
    env.step(env.action_space.sample() * 0)
    env.render('human')

