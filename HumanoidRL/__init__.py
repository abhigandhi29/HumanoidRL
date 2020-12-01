from gym.envs.registration import register
from HumanoidRL.envs import HumanoidEnv

register(
        id='HumanoidRL-v0',
        entry_point=HumanoidEnv,
        max_episode_steps=1000
        )
