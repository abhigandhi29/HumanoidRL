import HumanoidRL
from stable_baselines3 import TD3

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    model = TD3.load(path)
    env = gym.make(args.env_name, render=args.render)
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, _, _, _ = env.step(action)