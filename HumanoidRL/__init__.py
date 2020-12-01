from gym.envs.registration import register

register(
        id='HumanoidRL-v0',
        entry_point='HumanoidRL.envs:HumanoidEnv',
        max_episode_steps=1000
        )
