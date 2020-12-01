import os
import gym
import numpy as np
import HumanoidRL
from torch.nn import Tanh
from stable_baselines3 import TD3
from stable_baselines3.td3.policies import MlpPolicy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_runs', type=int, default=5)
    parser.add_argument('--env_name', type=str, default="HumanoidRL-v0")
    parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--data_dir', type=str, default='results')
    parser.add_argument('--save_freq', type=int, default=20)
    parser.add_argument('--render', type=bool, default=False)
    parser.add_argument('--steps', type=int, default=int(1e7))
    parser.add_argument('--debug', type=bool, default=False)
    args = parser.parse_args()
    env = gym.make(args.env_name, render=args.render)
    # The noise objects for TD3
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(
        n_actions), sigma=0.1 * np.ones(n_actions))
    for i in range(args.num_runs):
        save_dir = os.path.join(args.data_dir, args.exp_name+f"/{i}/")
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        env = gym.make(args.env_name, render=args.render)
        model = TD3(MlpPolicy, env, action_noise=action_noise, verbose=2 if args.debug else 1, seed=i,
                    learning_starts=1000, optimize_memory_usage=True, tensorboard_log=save_dir, 
                    create_eval_env=True, policy_kwargs={'activation_fn': Tanh})
        model.learn(total_timesteps=args.steps, log_interval=10)
        model.save(save_dir)
