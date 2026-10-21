import pickle
import random
import numpy as np
import copy
import gym
import seaborn as sns
import matplotlib.pyplot as plt
import os
import wandb
from envs import MultiAgentEnv

TOP = 0
BOT = 1
LEFT = 2
RIGHT = 3
TOP_RIGHT = 4
RIGHT_BOT = 5
BOT_LEFT = 6
LEFT_TOP = 7

class Entity(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.force = np.zeros(4,dtype=np.int8)

class Box(Entity):
    def __init__(self, x, y):
        super(Box, self).__init__(x, y)
        self.radius = 1

    def top(self):
        return self.y - self.radius

    def bot(self):
        return self.y + self.radius
    
    def left(self):
        return self.x - self.radius
    
    def right(self):
        return self.x + self.radius


class PushBox(MultiAgentEnv):
    def __init__(self, episode_limit=300, n_boxes=2, grid_size=15, spawn_range=10, write_wide_trace=False, *args, **kwargs):
        self.n_actions = 8
        self.n_boxes = n_boxes
        self.n_agents = 2
        self.spawn_range = spawn_range
        self.write_wide_trace = write_wide_trace
        self.observation_space = gym.spaces.MultiDiscrete([grid_size for _ in range(2 * (self.n_agents + self.n_boxes))])
        # each agent can choose one branch at each timestep
        self.action_space = gym.spaces.MultiDiscrete([self.n_actions] * self.n_agents)
        offset = np.array([i * 4 for i in range(self.n_boxes)])
        offset = offset - offset.mean()
        self.init_box = [Box(grid_size // 2 + i, grid_size // 2 + i) for i in offset.astype(int)]
        self.wall_map = np.zeros((grid_size, grid_size))
        self.grid_size = grid_size
        self.episode_limit = episode_limit
        self.step_count = 0
        self.done = False
        self.agents, self.boxes = None, None
        self.success_rew = 100
        self.box_trace_narrow = np.ones((self.grid_size, self.grid_size), dtype=np.int64)
        self.box_trace_wide = np.ones((self.grid_size, self.grid_size), dtype=np.int64)
        self.agent_trace = np.ones((self.grid_size, self.grid_size), dtype=np.int64)
        self.wall = np.zeros((self.grid_size, self.grid_size))
        wall_width = 3
        self.wall[:wall_width, :] = 1
        self.wall[-wall_width:, :] = 1
        self.wall[:, :wall_width] = 1
        self.wall[:, -wall_width:] = 1

    def step(self, action):
        assert not self.done, "error: Trying to call step() after an episode is done"
        self._compute_force(action)
        self._update_box_location()
        for agent_id, agent in enumerate(self.agents):
            self._update_agent_location(agent_id)
        self.step_count += 1
        rew = self._reward()
        self.done = True if self.step_count == self.episode_limit or rew >= 1 else False

        box_radius=2
        agent_radius=1

        for box in self.boxes:
            self.box_trace_narrow[box.x, box.y] += 1
            self.box_trace_wide[max(0, box.y-box_radius):min(self.grid_size-1, box.y+box_radius), max(0,box.x-box_radius):min(self.grid_size-1,box.x+box_radius)] += 1
        for agent in self.agents:
            self.agent_trace[max(0, agent.y-agent_radius):min(self.grid_size-1, agent.y+agent_radius), max(0,agent.x-agent_radius):min(self.grid_size-1,agent.x+agent_radius)] += 1

        return rew, self.done, {}

    def get_obs(self):
        return [self.get_state() for _ in range(self.n_agents)]

    def get_obs_size(self):
        return self.get_state_size()

    def get_state(self):
        obs = []
        for agent in self.agents:
            obs.extend([agent.x, agent.y])
        for box in self.boxes:
            obs.extend([box.x, box.y])
        return np.array(obs)

    def get_state_size(self):
        return (self.n_agents + self.n_boxes) * 2

    def get_avail_actions(self):
        return [[1 for _ in range(self.n_actions)] for _ in range(self.n_agents)]

    def get_total_actions(self):
        return self.n_actions

    def get_native_state(self):
        return self.get_state()

    def get_native_state_size(self):
        return self.get_state_size()

    def get_alive_state(self):
        return np.ones(self.n_agents)

    def get_alive_state_size(self):
        return self.n_agents

    def get_n_enemies(self):
        return 0

    def get_native_state_summary(self):
        return {}

    def reset(self):
        self.boxes = copy.deepcopy(self.init_box)
        self._update_wall()

        agents = []
        for i in range(self.n_agents):
            x = random.randint(self.grid_size // 2 - self.spawn_range, self.grid_size // 2 + self.spawn_range)
            y = random.randint(self.grid_size // 2 - self.spawn_range, self.grid_size // 2 + self.spawn_range)
            while self.wall_map[y][x] == 1:
                x = random.randint(self.grid_size // 2 - self.spawn_range, self.grid_size // 2 + self.spawn_range)
                y = random.randint(self.grid_size // 2 - self.spawn_range, self.grid_size // 2 + self.spawn_range)
            agents.append(Entity(x, y))

        self.agents = agents
        self.step_count = 0
        self.done = False

        if self.write_wide_trace:
            plt.figure(figsize=(10, 10))

            box_trace = plt.cm.Reds(np.log(self.box_trace_wide) / np.log(self.box_trace_wide).max())
            box_trace[..., 3] = 0.6
            box_trace[self.box_trace_wide == 1, 3] = 0
            agent_trace = plt.cm.Blues(np.log(self.agent_trace) / np.log(self.agent_trace).max())
            agent_trace[..., 3] = 0.6
            agent_trace[self.agent_trace == 1, 3] = 0
            wall = plt.cm.Greys(self.wall)
            wall[self.wall==0, 3] = 0

            plt.imshow(agent_trace)
            plt.imshow(box_trace)
            plt.imshow(wall)
            plt.axis('off')
            plt.savefig(f"push_box.png", transparent=True, bbox_inches='tight', pad_inches=0)
            plt.close()
        return self.get_state()

    def reset_trace(self):
        self.box_trace_narrow[...] = 1
        self.box_trace_wide[...] = 1
        self.agent_trace[...] = 1

    def _compute_force(self, actions):
        # compute force on the box
        for box in self.boxes:
            box.force[:] = 0

            for i, agent in enumerate(self.agents):
                self.agents[i].force[:] = 0

                if actions[i] == TOP:
                    self.agents[i].force[TOP] = 2
                    if box.left() <= agent.x <= box.right() and agent.y == box.bot() + 1:
                        box.force[TOP] += 1
                elif actions[i] == BOT:
                    self.agents[i].force[BOT] = 2
                    if box.left() <= agent.x <= box.right() and agent.y == box.top() - 1:
                        box.force[BOT] += 1
                elif actions[i] == LEFT:
                    self.agents[i].force[LEFT] = 2
                    if box.top() <= agent.y <= box.bot() and agent.x == box.right() + 1:
                        box.force[LEFT] += 1
                elif actions[i] == RIGHT:
                    self.agents[i].force[RIGHT] = 2
                    if box.top() <= agent.y <= box.bot() and agent.x == box.left() - 1:
                        box.force[RIGHT] += 1
                elif actions[i] == TOP_RIGHT:
                    self.agents[i].force[[TOP,RIGHT]] = 1
                elif actions[i] == RIGHT_BOT:
                    self.agents[i].force[[BOT,RIGHT]] = 1
                elif actions[i] == BOT_LEFT:
                    self.agents[i].force[[BOT,LEFT]] = 1
                elif actions[i] == LEFT_TOP:
                    self.agents[i].force[[TOP,LEFT]] = 1
                else:
                    raise NotImplementedError

    def _update_box_location(self):
        for box in self.boxes:
            for idx, f in enumerate(box.force):
                if f == 0:
                    continue

                if idx == TOP:
                    box.y -= min(box.top(), f)
                elif idx == BOT:
                    box.y += min((self.grid_size - 1) - box.bot(), f)
                elif idx == LEFT:
                    box.x -= min(box.left(), f)
                elif idx == RIGHT:
                    box.x += min((self.grid_size - 1) - box.right(), f)
        self._update_wall()

    def _update_wall(self):
        self.wall_map[:] = 0
        for box in self.boxes:
            self.wall_map[box.top() : box.bot() + 1, box.left() : box.right() + 1] = 1

    def _update_agent_location(self, agent_id):
        x, y = self.agents[agent_id].x, self.agents[agent_id].y

        for idx, f in enumerate(self.agents[agent_id].force):
            if f == 0:
                continue
            move = 0

            if idx == TOP:
                for i in range(1,f+1):
                    if y-i >= 0 and self.wall_map[y-i,x] == 0:
                        move += 1
                    else:
                        break
                self.agents[agent_id].y -= move
            elif idx == BOT:
                for i in range(1,f+1):
                    if y+i < self.grid_size and self.wall_map[y+i,x] == 0:
                        move += 1
                    else:
                        break
                self.agents[agent_id].y += move
            elif idx == LEFT:
                for i in range(1,f+1):
                    if x-i >= 0 and self.wall_map[y,x-i] == 0:
                        move += 1
                    else:
                        break
                self.agents[agent_id].x -= move
            elif idx == RIGHT:
                for i in range(1,f+1):
                    if x+i < self.grid_size and self.wall_map[y,x+i] == 0:
                        move += 1
                    else:
                        break
                self.agents[agent_id].x += move

    def _reward(self):
        rew = 0
        for box in self.boxes:
            if box.left() == 0 or \
               box.top() == 0 or \
               box.right() == self.grid_size - 1 or \
               box.bot() == self.grid_size - 1:
                rew += self.success_rew
        return rew

    def save_plot(self, label, t):
        path = f"plot/push_box/{label}"
        os.makedirs(path, exist_ok=True)

        fig = plt.figure(figsize=(10, 10))

        box_trace = plt.cm.Reds(np.log(self.box_trace_wide) / np.log(self.box_trace_wide).max())
        box_trace[..., 3] = 0.6
        box_trace[self.box_trace_wide == 1, 3] = 0
        agent_trace = plt.cm.Blues(np.log(self.agent_trace) / np.log(self.agent_trace).max())
        agent_trace[..., 3] = 0.6
        agent_trace[self.agent_trace == 1, 3] = 0
        wall = plt.cm.Greys(self.wall)
        wall[self.wall == 0, 3] = 0

        plt.imshow(agent_trace)
        plt.imshow(box_trace)
        plt.imshow(wall)
        plt.axis('off')
        plt.savefig(f"push_box.png", transparent=True, bbox_inches='tight', pad_inches=0)

        wandb.log({
            "box_trace": wandb.Image(fig, caption=f"{t}"),
            "timestep": t
        })

        plt.close(fig)  # Good practice to close the figure

        with open(f"{path}/{t}.pkl", "wb") as f:
            pickle.dump(self.box_trace_narrow, f)

    def render(self):
        plt.figure(figsize=(10, 10))
        loc = np.ones((self.grid_size, self.grid_size), dtype=np.int64)
        for box in self.boxes:
            loc[box.top():box.bot()+1, box.left():box.right()+1] = 10
        for agent in self.agents:
            loc[agent.y, agent.x] = 5
        ax = sns.heatmap(loc, xticklabels=False, yticklabels=False, cbar=False)
        ax.set_xlabel('')
        ax.set_ylabel('')

        plt.show()

    def close(self):
        pass

    def seed(self):
        pass

    def save_replay(self):
        pass
