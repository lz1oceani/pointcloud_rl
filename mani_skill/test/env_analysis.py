import sapien_rl.env
import gym, numpy as np, time, pandas as pd


env = gym.make('OpenCabinet_state_medium-v1', skip_reward=True)
df = pd.DataFrame(index=np.arange(100), columns=['FPS', 'Set state error'])


def test_env(index):
    env.reset(level=index)
    start_time = time.time()
    states = []
    actions = []
    next_states = []
    num = 100

    for i in range(num):
        states.append(env.get_state())
        action = env.action_space.sample()
        actions.append(action)
        env.step(action)
        next_states.append(env.get_state())
    total_time = time.time() - start_time

    max_set_state_error = []
    for i in range(num):
        env.set_state(states[i])
        env.step(actions[i])
        max_set_state_error.append(np.linalg.norm(env.get_state() - next_states[i]))
    df.loc[index, ['FPS', 'Set state error']] = num / total_time, np.max(max_set_state_error)


for j in range(100):
    test_env(j)

df.to_csv('env_analysis.csv')

