
# Script to check the working of other modules

# import humanoidRL as env
import Utility as ut
import pickle
import argparse
import numpy as np

"""File containing the sample walk i.e. the joint angles per frame"""
path = "walk_positions.pckl"
action_lows = np.array([-1.14529, -0.379435, -1.53589, -0.0923279, -1.18944,
                        -0.397761, -1.14529, -0.79046, -1.53589, -0.0923279,
                        -1.1863, -0.768992, -2.08567, -0.314159, -2.08567,
                        -1.54462, -2.08567, -1.32645, -2.08567, 0.0349066])
action_highs = np.array([0.740718, 0.79046, 0.48398, 2.11255, 0.922581, 0.768992,
                         0.740718, 0.379435, 0.48398, 2.11255, 0.932006, 0.397761,
                         2.08567, 1.32645, 2.08567, -0.0349066, 2.08567, 0.314159,
                         2.08567, 1.54462])
action_mean = (action_highs+action_lows)/2
action_range = (action_highs-action_lows)/2


def read_from_pickle(path):
    """Reads from pickel file"""
    poses = []
    with open(path, 'rb') as file:
        while True:
            try:
                poses.append(pickle.load(file))
            except EOFError:
                break
    return poses


def test_env(render):
    """To test the HumanoidEnv (Environment) Class"""
    poses = read_from_pickle(path)[0]
    import gym
    env = gym.make("HumanoidRL-v0", render=render)
    obs = env.reset()
    for configs in poses:
        action = np.array[configs[6], configs[1], configs[10], configs[2], configs[18],
                  configs[12], configs[8], configs[4], configs[5], configs[14],
                  configs[0], configs[11], configs[19], configs[13],
                  configs[9], configs[15], configs[3], configs[7],
                  configs[16], configs[17]]
        action = (action-action_mean)/action_range
        obs, rew, done, w = env.step(action)


def test_utility(render):
    """To test the Utility Class"""
    poses = read_from_pickle(path)[0]
    Bot = ut.Utility()
    Bot.init_bot(240, render=render)
    Bot.get_observation()
    print("Observation:\n", Bot.observation)
    for configs in poses:
        action = [configs[6], configs[1], configs[10], configs[2], configs[18],
                  configs[12], configs[8], configs[4], configs[5], configs[14],
                  configs[0], configs[11], configs[19], configs[13],
                  configs[9], configs[15], configs[3], configs[7],
                  configs[16], configs[17]]
        Bot.execute_frame(action)
        Bot.update_joints()
        Bot.get_observation()
        print("Observation:", Bot.observation)


def main():
    """use python test_script --env=True to test environment
    use python test_script --util=True to test utilities
    """
    parser = argparse.ArgumentParser(description="Test framework")
    parser.add_argument(
        "--env",
        type=bool,
        default=False,
        help="set True to test the Gym Environment"
    )
    parser.add_argument(
        "--util",
        type=bool,
        default=False,
        help="set True to test the Utility Class"
    )
    parser.add_argument(
        "--render",
        type=bool,
        default=False,
        help="set True to run render humanoid"
    )
    args = parser.parse_args()
    if (args.env and args.util):
        print("Cannot test both envionment and "
              "Utility at once")
    if args.env:
        test_env(args.render)
    elif args.util:
        test_utility(args.render)
    else:
        print("No flag given try -h for help")


main()
