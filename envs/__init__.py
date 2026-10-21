from functools import partial
import sys
import os

from .grf.grf import GRF
from .multiagentenv import MultiAgentEnv

from .starcraft import StarCraft2Env
from .mpe.push_box import PushBox
from .gymma import GymmaWrapper


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["grf"] = partial(env_fn, env=GRF)
REGISTRY["push_box"] = partial(env_fn, env=PushBox)
REGISTRY["gymma"] = partial(env_fn, GymmaWrapper)

try:
    from .starcraft.smacv2_wrapper import SMACv2Wrapper
    REGISTRY["sc2v2"] = partial(env_fn, env=SMACv2Wrapper)
except ImportError:
    pass



if sys.platform == "linux":
    os.environ.setdefault("SC2PATH", "~/StarCraftII")

#
# def __check_and_prepare_smac_kwargs(kwargs):
#     assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
#     assert kwargs[
#         "common_reward"
#     ], "SMAC only supports common reward. Please set `common_reward=True` or choose a different environment that supports general sum rewards."
#     del kwargs["common_reward"]
#     del kwargs["reward_scalarisation"]
#     assert "map_name" in kwargs, "Please specify the map_name in the env_args"
#     return kwargs
#
# def smacv2_fn(**kwargs) -> MultiAgentEnv:
#     kwargs = __check_and_prepare_smac_kwargs(kwargs)
#     return SMACv2Wrapper(**kwargs)
#
# REGISTRY["sc2v2"] = smacv2_fn