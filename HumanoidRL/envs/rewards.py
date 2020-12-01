import numpy as np


def not_fall(env, prev_ob):
    cost = np.sum(np.abs(env.Nao.bodyPos[0:2]-env.Nao.startPos[0:2]))
    vel_cost = np.sum(np.abs(env.Nao.bodyVel))+np.sum(np.abs(env.Nao.bodyAng))
    reward = (1./cost * 0.1)+(1./vel_cost * 0.01)
    return reward

def walk(env, prev_ob):
    pass


def get_reward(env=None, prev_ob=None, type=''):
    switcher = {
        'not_fall': not_fall,
        'walk': walk
    }
    reward = switcher.get(type)
    return reward(env, prev_ob)
