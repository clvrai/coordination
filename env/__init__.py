import gym
from gym.envs.registration import register


def register_environments():
    # register ant environments
    register(
        id='ant-forward-v0',
        entry_point='env.ant.primitives.ant_forward:AntForwardEnv',
        kwargs={},
    )
    register(
        id='ant-push-v0',
        entry_point='env.ant.composite.ant_push:AntPushEnv',
        kwargs={},
    )

    # register jaco environments
    register(
        id='two-jaco-pick-v0',
        entry_point='env.jaco.primitives.two_jaco_pick:TwoJacoPickEnv',
        kwargs={},
    )
    register(
        id='two-jaco-push-v0',
        entry_point='env.jaco.primitives.two_jaco_push:TwoJacoPushEnv',
        kwargs={},
    )
    register(
        id='two-jaco-place-v0',
        entry_point='env.jaco.primitives.two_jaco_place:TwoJacoPlaceEnv',
        kwargs={},
    )
    register(
        id='two-jaco-pick-push-place-v0',
        entry_point='env.jaco.composite.two_jaco_ppp:TwoJacoPickPushPlaceEnv',
        kwargs={},
    )
    register(
        id='two-jaco-bar-moving-v0',
        entry_point='env.jaco.composite.two_jaco_pmp:TwoJacoPickMovePlaceEnv',
        kwargs={},
    )


register_environments()


def str2dict(args_str):
    args_dict = {}
    args_list = args_str.split("/")
    for args in args_list:
        k, v = args.split('-')
        v = v.strip()
        if v in ['True', 'False']:
            v_val = True if v == 'True' else False
        elif v.startswith('[') and v.endswith(']'):
            v_val = [float(val.strip()) for val in v[1:-1].split(',')]
        else:
            try:
                v_val = float(v)
            except:
                v_val = v
        args_dict.update({k: v_val})
    return args_dict


def make_env(name, config):
    env_config = {}
    if config.env_args is not None:
        env_config = str2dict(config.env_args)
    if config.init_qpos_dir is not None:
        env_config['init_qpos_dir'] = config.init_qpos_dir
    return gym.make(config.env, **env_config)

