import datetime
import math
import os
import pickle
import pprint
import random
import subprocess
import time
import threading
import torch as th
import wandb
from types import SimpleNamespace as SN

from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
try:
    from smac.env import StarCraft2Env
except ImportError:
    from smacv2.env import StarCraft2Env

def get_agent_own_state_size(env_args):
    sc_env = StarCraft2Env(**env_args)
    # qatten parameter setting (only use in qatten)
    return  4 + sc_env.shield_bits_ally + sc_env.unit_type_bits

def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}__{}".format(args.label, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(dirname(abspath(__file__)))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        print('args.test_nepisode',args.test_nepisode)
        batch, stats =runner.run(test_mode=True)
        print('batch:',batch)
        print('stats:',stats)
    if args.save_replay:
        runner.save_replay()


    runner.close_env()
    os.makedirs('replay', exist_ok=True)
    subprocess.run(f"cp -r {os.path.expanduser(os.environ.get('SC2PATH'))}/Replays/* replay", shell=True, env=os.environ)

    return batch

def run_sequential(args, logger):
    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.native_state_size = env_info['native_state_size']
    args.native_state_summary = env_info['native_state_summary']
    args.env_args['time_limit']=env_info["episode_limit"]
    args.n_enemies=env_info["n_enemies"]
    args.accumulated_episodes = getattr(args, "accumulated_episodes", None)

    if getattr(args, 'agent_own_state_size', False):
        if args.env=='gfootball':
            args.agent_own_state_size=13
        else:
            args.agent_own_state_size = get_agent_own_state_size(args.env_args)

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "alive_state": {"vshape": env_info["alive_state_size"]},
        "native_state": {"vshape": env_info['native_state_size']},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "probs": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.float},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)
    random_buffer = ReplayBuffer(scheme, groups, args.random_buffer_size, env_info["episode_limit"] + 1,
                                 preprocess=preprocess,
                                 device="cpu" if args.buffer_cpu_only else args.device)

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
    random_mac = mac_REGISTRY['random_mac']()

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Set up initial MI, stats for each substate
    if args.env == "gymma":
        map_name = args.env_args['key']
    else:
        map_name = args.env_args['map_name']
    substate_entropy_path = f"substate_entropy/{map_name}.pkl"
    if os.path.exists(substate_entropy_path):
        with open(substate_entropy_path, 'rb') as file:
            substate_entropy = pickle.load(file)
    else:
        random_samples = get_random_samples(runner, random_mac, random_buffer)
        substate_scale = get_substate_scale(random_samples, args.device)
        substate_entropy = get_substate_entropy(args, runner, random_mac, random_buffer, substate_scale)
        with open(substate_entropy_path, 'wb') as file:
            pickle.dump(substate_entropy, file)
    logger.console_logger.info(f'at {runner.t_env}, substate_entropy: {substate_entropy}')

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args, substate_entropy)

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":


        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))
                print('timesteps',timesteps)

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner)
            return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    last_under_explored_update_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    if args.under_explored_update:
        random_samples = get_random_samples(runner, random_mac, random_buffer)
        substate_scale = get_substate_scale(random_samples, args.device)
        diff_count = {}
        for i in range(args.native_state_size):
            diff_count[i] = {}
        diff_count['total'] = 0

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    while runner.t_env <= args.t_max:

        # Run for a whole episode at a time

        with th.no_grad():
            episode_batch, _ = runner.run(test_mode=False)
            buffer.insert_episode_batch(episode_batch)

        if buffer.can_sample(args.batch_size):
            episode_sample = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            learner.train(episode_sample, runner.t_env, episode)
            del episode_sample

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env

            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", map_name, args.label, str(runner.t_env))
            #"results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            # logger.print_recent_stats()
            last_log_T = runner.t_env

        if args.under_explored_update:
            update_count(args, substate_scale, diff_count, episode_batch)
            if (runner.t_env - last_under_explored_update_T) >= args.under_explored_update_interval:
                H_s = th.zeros(args.native_state_size, device=args.device)
                for i in range(args.native_state_size):
                    for s in diff_count[i]:
                        p_s = diff_count[i][s] / diff_count['total']
                        H_s[i] -= p_s * math.log(p_s)
                learner.update_substate_weights(H_s, runner.t_env)
                diff_count = {}
                for i in range(args.native_state_size):
                    diff_count[i] = {}
                diff_count['total'] = 0
                last_under_explored_update_T = runner.t_env


    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config

def get_random_samples(runner, random_policy, buffer):
    original_policy = runner.mac
    runner.mac = random_policy
    while not buffer.can_sample(buffer.buffer_size):
        episode_batch, _ = runner.run(test_mode=True)
        buffer.insert_episode_batch(episode_batch)
    runner.mac = original_policy
    runner.reset()
    return buffer.sample(buffer.buffer_size)


def get_average_diff(samples, device):
    terminated = samples["terminated"][:, :-1].float() # s_{t} terminated
    mask = samples["filled"][:, :-1].float() # s_{t},s_{t+1} filled
    mask *= 1 - terminated # mask t-th sample  (s_{t}, s_{t+1})
    mask[:, 1:] *= 1 - terminated[:, :-1] # mask t+1-th sample  (s_{t+1}, s_{t+2})
    mask[:, 0] = 0 # mask first sample
    diff = th.abs(samples['native_state'][:, :-1, :] - samples['native_state'][:, 1:, :])
    diff *= mask
    diff = diff.sum(dim=(0,1)) / mask.sum()
    return th.where(diff == 0, 1, diff).to(device)

def get_substate_scale(samples, device):
    return 1 / get_average_diff(samples, device)

def get_substate_entropy(args, runner, random_policy, buffer, substate_scale):
    original_policy = runner.mac
    runner.mac = random_policy
    n_episode = buffer.episodes_in_buffer
    n_updated_episode = buffer.episodes_in_buffer
    count = {}
    H_s = th.zeros(args.native_state_size, device=args.device)
    total_steps = 0
    start_time = time.time()
    for i in range(args.native_state_size):
        count[i] = {}

    while n_episode <= args.n_episodes_substate_entropy:
        if random.random() < 0.5:
            elapsed = time.time() - start_time
            ep_time = n_episode / elapsed
            remain = (args.n_episodes_substate_entropy - n_episode) / ep_time
            print(f"Random traj sampling: {n_episode} / {args.n_episodes_substate_entropy}. {remain:.2f} seconds remains.")

        while n_updated_episode / buffer.episodes_in_buffer < 1:
            episode_batch, _ = runner.run(test_mode=True)
            buffer.insert_episode_batch(episode_batch)
            n_updated_episode += episode_batch.batch_size
        samples = buffer.sample(buffer.buffer_size)
        samples.to(args.device)
        n_updated_episode = 0
        n_episode += buffer.buffer_size

        b, t, _ = samples['native_state'].shape
        diff = ((samples['native_state'][:, 1:] - samples['native_state'][:, :-1]) * substate_scale * (
                    10 ** args.round_decimal)).int()

        for i in range(b):
            for j in range(t):
                if samples['terminated'][i][j + 1].item() == 1:
                    break

                total_steps += 1
                for k in range(args.native_state_size):
                    s = diff[i][j][k].item()
                    count[k][s] = count[k].get(s, 0) + 1

    for i in range(args.native_state_size):
        for s in count[i]:
            p_s = count[i][s] / total_steps
            H_s[i] -= p_s * math.log(p_s)

    runner.mac = original_policy
    runner.reset()
    return H_s

def update_count(args, substate_scale, count, samples):
    samples.to(args.device)

    b, t, _ = samples['native_state'].shape
    diff = ((samples['native_state'][:, 1:] - samples['native_state'][:, :-1]) * substate_scale * (
            10 ** args.round_decimal)).int()

    for i in range(b):
        for j in range(t):
            if j + 1 < t and samples['terminated'][i][j + 1].item() == 1:
                break

            count['total'] += 1
            for k in range(args.native_state_size):
                s = diff[i][j][k].item()
                count[k][s] = count[k].get(s, 0) + 1
