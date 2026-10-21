from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from multiprocessing import Pipe, Process
import numpy as np
import os, wandb, csv
import copy



# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class ParallelRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run

        # Make subprocesses for the envs
        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])
        env_fn = env_REGISTRY[self.args.env]
        self.ps = []
        for i, worker_conn in enumerate(self.worker_conns):
            ps = Process(target=env_worker,
                         args=(worker_conn, CloudpickleWrapper(partial(env_fn, **self.args.env_args))))
            self.ps.append(ps)

        for p in self.ps:
            p.daemon = True
            p.start()

        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]

        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        self.n_native_state = None

        self.log_train_stats_t = -100000
        if self.args.env == "gymma":
            map_name = self.args.env_args['key']
        else:
            map_name = self.args.env_args['map_name']
        seed = self.args.env_args['seed']
        self.csv_dir = f'./csv_files/{map_name}/{args.mixer}/{args.label}/'
        self.csv_path = f'{self.csv_dir}seed_{seed}_{args.label}.csv'
        if not os.path.exists(self.csv_dir):
            os.makedirs(self.csv_dir)
        if args.wandb:
            job_type = 'training'
            wandb_name = f'{map_name}_{args.label}'
            wandb.login(key=args.key, relogin=True)
            wandb.init(project=args.project, entity=args.entity, name=wandb_name, group=map_name, job_type=job_type, config=args)

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess
        self.n_native_state = scheme['native_state']['vshape']

    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        pass

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self):
        self.batch = self.new_batch()

        # Reset the envs
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))

        pre_transition_data = {
            "state": [],
            "avail_actions": [],
            "obs": [],
            "native_state": [],
            "alive_state": [],
        }
        # Get the obs, state and avail_actions back
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])
            pre_transition_data["native_state"].append(data["native_state"])
            pre_transition_data["alive_state"].append(data["alive_state"])

        self.batch.update(pre_transition_data, ts=0)

        self.t = 0
        self.env_steps_this_run = 0

    def run(self, test_mode=False):
        self.reset()

        all_terminated = False
        episode_returns = [0] * self.batch_size
        episode_lengths = [0] * self.batch_size
        self.mac.init_hidden(batch_size=self.batch_size)
        terminated = [False] * self.batch_size
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
        final_env_infos = []  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION

        save_probs = getattr(self.args, "save_probs", False)
        while True:

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch for each un-terminated env
            if save_probs:
                actions, probs = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env,
                                                         bs=envs_not_terminated, test_mode=test_mode)
            else:
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated,
                                                  test_mode=test_mode)

            cpu_actions = actions.to("cpu").numpy()

            # Update the actions taken
            actions_chosen = {
                "actions": actions.unsqueeze(1).to("cpu"),
            }
            if save_probs:
                actions_chosen["probs"] = probs.unsqueeze(1).to("cpu")

            self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Send actions to each env
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated:  # We produced actions for this env
                    if not terminated[idx]:  # Only send the actions to the env if it hasn't terminated
                        parent_conn.send(("step", cpu_actions[action_idx]))
                    action_idx += 1  # actions is not a list over every env

            # Update envs_not_terminated
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            all_terminated = all(terminated)
            if all_terminated:
                break

            # Post step data we will insert for the current timestep
            post_transition_data = {
                "reward": [],
                "terminated": [],
            }
            # Data for the next step we will insert in order to select an action
            pre_transition_data = {
                "state": [],
                "avail_actions": [],
                "native_state": [],
                "alive_state": [],
                "obs": []
            }

            # Receive data back for each unterminated env
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv()
                    # Remaining data for this current timestep
                    post_transition_data["reward"].append((data["reward"],))

                    episode_returns[idx] += data["reward"]
                    episode_lengths[idx] += 1
                    if not test_mode:
                        self.env_steps_this_run += 1

                    env_terminated = False
                    if data["terminated"]:
                        final_env_infos.append(data["info"])
                    if data["terminated"] and not data["info"].get("episode_limit", False):
                        env_terminated = True
                    terminated[idx] = data["terminated"]
                    post_transition_data["terminated"].append((env_terminated,))

                    # Data for the next timestep needed to select an action
                    pre_transition_data["state"].append(data["state"])
                    pre_transition_data["avail_actions"].append(data["avail_actions"])
                    pre_transition_data["native_state"].append(data["native_state"])
                    pre_transition_data["alive_state"].append(data["alive_state"])
                    pre_transition_data["obs"].append(data["obs"])

            # Add post_transiton data into the batch
            self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)

        if not test_mode:
            self.t_env += self.env_steps_this_run

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        infos = [cur_stats] + final_env_infos

        cur_stats.update({k: sum(d.get(k, 0) for d in infos) for k in set.union(*[set(d) for d in infos])})
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)
        cur_stats['step'] = self.t_env

        if test_mode:
            if 'battle_won' in cur_stats.keys():
                win_rate = cur_stats['battle_won'] / cur_stats['n_episodes']
            else:
                win_rate = ((np.array(cur_returns) > 8).astype('int')).sum() / cur_stats['n_episodes']
            cur_stats['win_rate'] = win_rate

        stats = copy.deepcopy(cur_stats)
        cur_returns.extend(episode_returns)

        n_test_runs = max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        if test_mode and (len(self.test_returns) == n_test_runs):
            reward = np.mean(cur_returns)
            mean_steps = cur_stats["ep_length"] / cur_stats['n_episodes']
            if win_rate > 0.3 and self.args.anneal_enable:
                self.args.anneal_start = True
                self.args.anneal_start_t = self.t_env
            self.writereward(self.csv_path, reward, win_rate, mean_steps, self.t_env)
            if self.args.wandb:
                wandb.log({'Test_win_rate': win_rate, log_prefix + "return_mean": reward,
                           log_prefix + "return_std": np.std(cur_returns), 'mean_steps:': mean_steps}, step=self.t_env)
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch, stats

    def writereward(self, path, reward, win_rate, mean_steps, step):
        if os.path.isfile(path):
            with open(path, 'a+') as f:
                csv_write = csv.writer(f)
                csv_write.writerow([step, reward, win_rate, mean_steps])
        else:
            with open(path, 'w') as f:
                csv_write = csv.writer(f)
                csv_write.writerow(['step', 'reward', 'win_rate', 'mean_steps'])
                csv_write.writerow([step, reward, win_rate, mean_steps])

    def _log(self, returns, stats, prefix):

        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean", v / stats["n_episodes"], self.t_env)
        stats.clear()

    def save_plot(self, label, t):
        self.parent_conns[0].send(("save_plot", (label, t)))
        self.parent_conns[0].recv()


def env_worker(remote, env_fn):
    # Make environment
    env = env_fn.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data
            # Take a step in the environment
            reward, terminated, env_info = env.step(actions)
            # Return the observations, avail_actions and state to make the next action
            state = env.get_state()
            avail_actions = env.get_avail_actions()
            obs = env.get_obs()
            native_state = env.get_native_state()
            alive_state = env.get_alive_state()
            remote.send({
                # Data for the next timestep needed to pick an action
                "state": state,
                "avail_actions": avail_actions,
                "obs": obs,
                "native_state": native_state,
                "alive_state":alive_state,
                # Rest of the data for the current timestep
                "reward": reward,
                "terminated": terminated,
                "info": env_info
            })
        elif cmd == "reset":
            env.reset()
            remote.send({
                "state": env.get_state(),
                "avail_actions": env.get_avail_actions(),
                "native_state": env.get_native_state(),
                "alive_state": env.get_alive_state(),
                "obs": env.get_obs()
            })
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            env_info = env.get_env_info()
            env_info["native_state_size"] = env.get_native_state_size()
            env_info["alive_state_size"] = env.get_alive_state_size()
            env_info['n_enemies'] = env.get_n_enemies()
            env_info['native_state_summary'] = env.get_native_state_summary()
            remote.send(env_info)
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        else:
            raise NotImplementedError


class CloudpickleWrapper():
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)
