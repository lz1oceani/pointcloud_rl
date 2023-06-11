import argparse

from gym.spaces import Box

from sapien_rl.eval import Evaluator, BasePolicy, UserPolicy


def parse_args():
    parser = argparse.ArgumentParser(description='SAPIEN RL Evaluation')
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--n-episodes', type=int, default=2)
    parser.add_argument('--level-range', type=str, default=None)  # format is like 100-200
    parser.add_argument('--result-path', type=str, default='./eval_results.csv')
    parser.add_argument('--vis', action='store_true')
    parser.add_argument(
        'opts',
        help='Parameters passed to user policy',
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args()


class RandomPolicy(BasePolicy):  # just for testing purpose
    def __init__(self, env_name):
        # env = gym.make(env_name)
        # self.action_space = copy.copy(env.action_space)
        # env.close()
        # del env
        self.action_space = Box(-1.0, 1.0, shape=(8,))

    def act(self, state):
        return self.action_space.sample()


def parse_level_range(s):
    # s is like 100-200
    if s is None:
        return None
    if '-' not in s:
        raise Exception('Incorrect level range format, it should be like 100-200.')
    a, b = s.split('-')
    return (int(a), int(b))


def evaluate_random_policy():
    args = parse_args()
    policy = RandomPolicy(args.env)
    e = Evaluator(args.env, policy)
    e.run(args.n_episodes, parse_level_range(args.level_range), args.vis)
    e.export_to_csv(args.result_path)


def evaluate_user_policy():
    args = parse_args()
    policy = UserPolicy(args.opts)
    e = Evaluator(args.env, policy)
    e.run(args.n_episodes, parse_level_range(args.level_range), args.vis)
    e.export_to_csv(args.result_path)


if __name__ == '__main__':
    # evaluate_random_policy()
    evaluate_user_policy()
