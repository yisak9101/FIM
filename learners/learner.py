import copy
import json

import torch, os, random, wandb, csv

from modules.mixers.nmix import Mixer
from modules.mixers.vdn import VDNMixer
from modules.mixers.qatten import QattenMixer
from modules.pred.pred_net import PredNetwork
from utils.rl_utils import build_td_lambda_targets, build_q_lambda_targets
import torch as th
from torch.optim import RMSprop, Adam
import numpy as np
from utils.th_utils import get_parameters_num, min_max_normalize
from torch.nn import functional as F
from components.episode_buffer import EpisodeBatch


class Learner:
    def __init__(self, mac, scheme, logger, args, entropy):
        self.args = args
        self.mac = mac
        self.inf_mac = copy.deepcopy(mac)
        self.logger = logger

        self.last_target_update_episode = 0
        self.device = th.device('cuda' if args.use_cuda else 'cpu')
        self.params = list(mac.parameters())
        np.random.seed(args.seed)
        th.manual_seed(args.seed)
        if args.mixer == "qatten":
            self.mixer = QattenMixer(args)
        elif args.mixer == "vdn":
            self.mixer = VDNMixer()
        elif args.mixer == "qmix":
            self.mixer = Mixer(args)
        else:
            raise "mixer error"
        self.target_mixer = copy.deepcopy(self.mixer)
        self.params += list(self.mixer.parameters())

        print('Mixer Size: ')
        print(get_parameters_num(self.mixer.parameters()))

        if self.args.optimizer == 'adam':
            self.optimiser = Adam(params=self.params, lr=args.lr, weight_decay=getattr(args, "weight_decay", 0))
        else:
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)
        self.target_inf_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1
        self.train_t = 0
        if args.env == "gymma":
            map_name = self.args.env_args['key']
        else:
            map_name = self.args.env_args['map_name']
        self.csv_dir = f'./csv_file/{map_name}'
        self.csv_path = f'{self.csv_dir}/seed_{args.seed}_{args.label}.csv'

        if not os.path.exists(self.csv_dir):
            os.makedirs(self.csv_dir)
        # priority replay
        self.use_per = getattr(self.args, 'use_per', False)
        self.return_priority = getattr(self.args, "return_priority", False)
        if self.use_per:
            self.priority_max = float('-inf')
            self.priority_min = float('inf')

        self.n_agents = args.n_agents
        self.n_enemies = args.n_enemies
        self.n_actions = args.n_actions
        self.n_native_states = args.native_state_size
        self.n_states = args.state_shape
        self.n_flatten_actions = self.n_agents * self.n_actions
        self.native_state_summary = args.native_state_summary

        logs = {}

        changeable = entropy > 1e-3
        logs['raw_entropy'] = entropy.cpu().numpy()
        entropy[changeable] = min_max_normalize(entropy[changeable])
        entropy[~changeable] = 1e10
        logs['normalized_entropy'] = entropy.cpu().numpy()
        logs['under_explored'] = entropy.cpu().numpy()
        self.substate_weight = F.softmax(-entropy / self.args.softmax_temperature, dim=0)
        self.substate_weight = th.where(self.substate_weight > 1e-2, self.substate_weight, th.zeros_like(self.substate_weight))
        logs['substate_weight'] = self.substate_weight.cpu().numpy()

        if self.args.wandb:
            save_dir = os.path.join(wandb.run.dir, "substate_logs")
            os.makedirs(save_dir, exist_ok=True)

            for k, v in logs.items():
                file_path = os.path.join(save_dir, f"{k}.log")
                value_dict = {int(i): float(v_i) for i, v_i in enumerate(v)}
                entry = {
                    "timestep": int(0),
                    "values": value_dict
                }
                with open(file_path, "a") as f:
                    json.dump(entry, f, indent=2)
                    f.write("\n\n")

        self.pred_net = PredNetwork(args, self.n_native_states + args.n_actions * args.n_agents + 1, args.rnn_hidden_dim, self.n_native_states, self.device)
        self.target_pred_net = PredNetwork(args, self.n_native_states + args.n_actions * args.n_agents + 1, args.rnn_hidden_dim, self.n_native_states, self.device)

        self.target_pred_net.load_state_dict(self.pred_net.state_dict())
        self.Target_update = False

        if args.use_cuda:
            self.pred_net.cuda()
            self.target_pred_net.cuda()

    # ------------------------------------------------------------------
    # Annealing coefficient: 1.0 → 0.0 as t_env → anneal_t_max
    # ------------------------------------------------------------------
    def _intrinsic_anneal_coef(self, t_env: int) -> float:
        ratio = min((t_env - self.args.anneal_start_t) / max(self.anneal_t_max, 1), 1.0)
        return float(1.0 - ratio)

    def update_substate_weights(self, entropy, t):
        logs = {}
        changeable = entropy > 1e-3
        logs['raw_entropy'] = entropy.cpu().numpy()
        entropy[changeable] = min_max_normalize(entropy[changeable])
        entropy[~changeable] = 1e10
        logs['normalized_entropy'] = entropy.cpu().numpy()
        logs['under_explored'] = entropy.cpu().numpy()
        substate_weight = F.softmax(-entropy / self.args.softmax_temperature, dim=0)
        substate_weight = th.where(substate_weight > 1e-2, substate_weight, 0)
        self.substate_weight = (1 - self.args.weight_update_rate) * self.substate_weight + self.args.weight_update_rate * substate_weight
        logs['substate_weight'] = self.substate_weight.cpu().numpy()

        if self.args.wandb:
            save_dir = os.path.join(wandb.run.dir, "substate_logs")
            for k, v in logs.items():
                file_path = os.path.join(save_dir, f"{k}.log")
                value_dict = {int(i): float(v_i) for i, v_i in enumerate(v)}
                entry = {
                    "timestep": int(t),
                    "values": value_dict
                }
                with open(file_path, "a") as f:
                    json.dump(entry, f, indent=2)
                    f.write("\n\n")

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, per_weight=None):
        # Get the relevant quantities
        ext_rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        state = batch["state"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        agent_alive = batch['alive_state'][:, 1:]
        ally_alive = agent_alive[..., :self.n_agents]
        enemy_alive = agent_alive[..., self.n_agents:]
        b, t, a, _ = batch["obs"][:, :-1].shape
        actions_onehot = (batch["actions_onehot"][:, :-1])
        native_state = batch['native_state'][:, :-1]
        substate = batch['native_state'][:, :-1]
        next_state = batch['native_state'][:, 1:]
        timesteps = (torch.tensor(range(0, t), device=self.device) / self.args.env_args['time_limit']).reshape(1, t, 1).expand(b, t, 1)
        pred_input = th.cat((native_state, actions_onehot.reshape(b, t, -1), timesteps), dim=-1)
        intrinsic_mask = mask.clone()
        alive_mask = (ally_alive * mask).unsqueeze(-1)
        inf_mask = alive_mask.bool().permute(0, 2, 1, 3).expand(-1, -1, -1, self.n_native_states)

        pred_loss = []
        for _ in range(self.args.predict_epoch):
            loss = (self.pred_net.update(pred_input, next_state, mask))
            pred_loss.append(loss)

        # Calculate estimated Q-Values
        self.mac.agent.train()
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs, traj = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        with (th.no_grad()):
            self.target_mac.agent.train()
            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs, _ = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)

            # We don't need the first timesteps Q-Value estimate for calculating targets
            target_mac_out = th.stack(target_mac_out, dim=1)  # Concat across time

            # Max over target Q-Values/ Double q learning
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)

            # Calculate n-step Q-Learning targets
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"])

            mask_env = mask.clone().roll(dims=-2, shifts=-1)
            mask_env[:, -1, :] = 0

            ## Calculate influence reward
            diff = (self.target_pred_net(pred_input) - substate).abs()
            diff = diff.unsqueeze(1).expand(b, self.n_agents, t, -1)

            countfact_sa = th.zeros(b, self.n_agents, self.n_actions, t, self.n_native_states + self.n_flatten_actions + 1, device=self.device)
            countfact_sa[..., :self.n_native_states] = native_state.reshape(b, 1, 1, t, -1)
            countfact_sa[..., self.n_native_states:self.n_native_states+self.n_flatten_actions] = actions_onehot.reshape(b, 1, 1, t, -1)
            countfact_sa[..., -1] = (torch.tensor(range(0, t), device=self.device) / self.args.env_args['time_limit']).reshape(1, 1, 1, t)
            action = th.eye(self.n_actions, device=self.device)
            for i in range(self.n_agents):
                for j in range(self.n_actions):
                    range_start = self.n_native_states + self.n_actions * i
                    idx = range(range_start, range_start + self.n_actions)
                    countfact_sa[:, i, j, :, idx] = action[j]
            countfact_s_prime = th.zeros(b, self.n_agents, self.n_actions, t, self.n_native_states, device=self.device)
            countfact_s = substate.reshape(b, 1, 1, t, -1).expand(-1, self.n_agents, self.n_actions, -1, -1) # (b, n_agents, n_actions, t, -1)
            for _t in range(t):
                countfact_s_prime[..., _t:_t + 1, :] = self.target_pred_net(countfact_sa[..., _t:_t+1, :])
            countfact_avail_actions = (avail_actions[:, :-1]).permute(0,2,3,1).unsqueeze(-1)
            countfact_diff = (countfact_s_prime - countfact_s).abs()
            countfact_diff = (countfact_diff * countfact_avail_actions).sum(2)  / (countfact_avail_actions.sum(2) + 1e-6)

            inf_tdi = (diff - countfact_diff) * inf_mask
            inf_td = inf_tdi.sum(1)
            trace_td = th.zeros((b, t, self.n_native_states), device=self.device)
            trace_td[:,0] = inf_td[:,0]
            for _t in range(1, t):
                trace_td[:, _t] = self.args.gamma * trace_td[:,_t-1] + inf_td[:,_t]
            trace_td = trace_td.clamp(min=1)
            trace_inf_td = th.zeros_like(inf_td)
            trace_inf_td[:,0] = inf_td[:,0]
            trace_inf_td[:,1:] = trace_td[:,:-1] * inf_td[:,1:]
            inf_t = (trace_inf_td * self.substate_weight).sum(-1, keepdim=True)
            inf_t = inf_t * self.args.alpha
            intrinsic = inf_t

            tot_reward = ext_rewards + intrinsic
            if getattr(self.args, 'q_lambda', False):
                qvals = th.gather(target_mac_out, 3, batch["actions"]).squeeze(3)
                qvals = self.target_mixer(qvals, batch["state"])

                targets = build_q_lambda_targets(tot_reward, terminated, mask, target_max_qvals, qvals,
                                                 self.args.gamma, self.args.td_lambda)
            else:
                targets = build_td_lambda_targets(tot_reward, terminated, mask, target_max_qvals,
                                                  self.args.n_agents, self.args.gamma, self.args.td_lambda)

        # Mixer
        chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])

        td_error = (chosen_action_qvals - targets.detach())
        td_error2 = 0.5 * td_error.pow(2)

        mask = mask.expand_as(td_error2)
        masked_td_error = td_error2 * mask

        # important sampling for PER
        if self.use_per:
            per_weight = th.from_numpy(per_weight).unsqueeze(-1).to(device=self.device)
            masked_td_error = masked_td_error.sum(1) * per_weight

        loss = L_td = masked_td_error.sum() / mask.sum()

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if random.random() < 0.1 and self.Target_update:
            pred_diff = th.abs(self.target_pred_net(pred_input)[0] - next_state)
            pred_diff = (pred_diff * mask).sum(dim=(0, 1))

            agent_alive = batch['alive_state'][:, 1:] * terminated
            mean_alive = (agent_alive[..., :self.n_agents]).sum(dim=(1,2)).mean()
            enemy_mean_alive = (agent_alive[..., self.n_agents:]).sum(dim=(1,2)).mean()

            self.writereward(
                actions, tot_reward, ext_rewards, inf_t, trace_td,
                intrinsic, L_td, np.array(pred_loss).mean(),
                pred_diff, mean_alive, enemy_mean_alive, intrinsic_mask, alive_mask, t_env)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_td", L_td.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean",
                                 (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                                 t_env)
            self.log_stats_t = t_env


        # return info
        info = {}
        # calculate priority
        if self.use_per:
            if self.return_priority:
                info["td_errors_abs"] = ext_rewards.sum(1).detach().to('cpu')
                # normalize to [0, 1]
                self.priority_max = max(th.max(info["td_errors_abs"]).item(), self.priority_max)
                self.priority_min = min(th.min(info["td_errors_abs"]).item(), self.priority_min)
                info["td_errors_abs"] = (info["td_errors_abs"] - self.priority_min) \
                                        / (self.priority_max - self.priority_min + 1e-5)
            else:
                info["td_errors_abs"] = ((td_error.abs() * mask).sum(1)
                                         / th.sqrt(mask.sum(1))).detach().to('cpu')
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return info

    def _update_targets(self):
        self.Target_update = True
        self.target_mac.load_state(self.mac)
        self.target_inf_mac.load_state(self.inf_mac)
        self.target_pred_net.load_state_dict(self.pred_net.state_dict())

        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        self.inf_mac.cuda()
        self.target_inf_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.pred_net.state_dict(), "{}/pred_net.th".format(path))
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def writereward(self, actions, total_reward, ext_reward, inf_t, cons_trace_td, int_reward,
                    td_loss, pred_loss, pred_diff, ally_alive, enemy_alive, intrinsic_mask, alive_mask, step):
        episodes = ext_reward.shape[0]
        timesteps = intrinsic_mask.sum()
        win_rate = (ext_reward.sum(dim=1) > 0).float().mean()
        total_return = total_reward.sum() / episodes
        ext_return = ext_reward.sum() / episodes
        int_return = int_reward.sum() / episodes
        inf_return = inf_t.sum() / episodes

        pred_diff_avg = pred_diff.mean() / timesteps

        cons_avg = (cons_trace_td * intrinsic_mask).sum() / episodes

        if self.args.wandb:
            data = {'t_win_rate': win_rate, " TD_Loss": td_loss, "total_return": total_return,
                    "pred_loss": pred_loss, "extrinsic_return": ext_return, 'intrinsic_return': int_return,
                    'inf_return': inf_return, "cons_avg": cons_avg,
                    'ally_alive': ally_alive, 'enemy_alive': enemy_alive, 'pred_diff': pred_diff_avg}

            if self.args.env == 'sc2':
                n_attack = (actions >= 6).sum() / episodes
                enemy_killed = self.args.n_enemies - enemy_alive
                data['n_attack'] = n_attack
                data['enemy_killed'] = enemy_killed
                data['focus_degree'] = enemy_killed / n_attack

            wandb.log(data,step=step)

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        self.pred_net.load_state_dict(th.load("{}/pred_net.th".format(path), map_location=lambda storage, loc: storage))
        self.target_pred_net.load_state_dict(th.load("{}/pred_net.th".format(path), map_location=lambda storage, loc: storage))

        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))