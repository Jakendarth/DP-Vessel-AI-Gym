from gym.envs.registration import register


register(
    id='Lincoln-v0',
    entry_point='gym_lincoln.envs:LincolnEnv',
)

