
from components.action_selectors import REGISTRY as action_REGISTRY

class RandomMAC:
    def __init__(self):
        self.action_selector = action_REGISTRY['random']()

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        chosen_actions = self.action_selector.select_action(avail_actions[bs])
        return chosen_actions

    def init_hidden(self, batch_size):
        pass

    def parameters(self):
        pass

    def load_state(self, other_mac):
        pass

    def cuda(self):
        pass

    def save_models(self, path):
        pass

    def load_models(self, path):
        pass