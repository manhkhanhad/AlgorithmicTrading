from elegantrl_.agents.net import (
    ActorSAC,
    CriticTwin,
)
import os
import torch
from elegantrl_.agents.AgentBase import AgentBase

class AgentRandom(AgentBase):
    def __init__(self, net_dim, state_dim, action_dim, gpu_id=0, args=None):
        self.if_off_policy = True
        self.act_class = getattr(self, "act_class", ActorSAC)
        self.cri_class = getattr(self, "cri_class", CriticTwin)
        super().__init__(net_dim, state_dim, action_dim, gpu_id, args)
        self.act.explore_noise = getattr(
            args, "explore_noise", 0.1
        )  # set for `get_action()`

        
    # def __init__(
    #     self, net_dim: int, state_dim: int, action_dim: int, gpu_id=0, args=None
    # ):
    #     """initialize

    #     replace by different DRL algorithms
    #     explict call self.init() for multiprocessing.

    #     :param net_dim: the dimension of networks (the width of neural networks)
    #     :param state_dim: the dimension of state (the number of state vector)
    #     :param action_dim: the dimension of action (the number of discrete action)
    #     :param reward_scale: scale the reward to get a appropriate scale Q value
    #     :param gamma: the discount factor of Reinforcement Learning

    #     :param learning_rate: learning rate of optimizer
    #     :param if_per_or_gae: PER (off-policy) or GAE (on-policy) for sparse reward
    #     :param env_num: the env number of VectorEnv. env_num == 1 means don't use VectorEnv
    #     :param gpu_id: the gpu_id of the training device. Use CPU when cuda is not available.
    #     """
    #     self.gamma = getattr(args, "gamma", 0.99)
    #     self.env_num = getattr(args, "env_num", 1)
    #     self.batch_size = getattr(args, "batch_size", 128)
    #     self.repeat_times = getattr(args, "repeat_times", 1.0)
    #     self.reward_scale = getattr(args, "reward_scale", 1.0)
    #     self.lambda_gae_adv = getattr(args, "lambda_entropy", 0.98)
    #     self.if_use_old_traj = getattr(args, "if_use_old_traj", False)
    #     self.soft_update_tau = getattr(args, "soft_update_tau", 2**-8)
    #     self.states = None
    #     self.device = torch.device(
    #         f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu"
    #     )
        
    #     self.act = ActorSAC(net_dim, state_dim, action_dim).to(self.device)
    #     self.cri = (
    #         CriticTwin(net_dim, state_dim, action_dim).to(self.device)
    #     )

    #     self.save_or_load_agent(cwd=config["TRAINED_MODEL_FOLDER"] + scenario + '/' + agent_name, if_save=True)
        

    # def save_or_load_agent(self, cwd, if_save):
    #     """save or load training files for Agent

    #     :param cwd: Current Working Directory. ElegantRL save training files in CWD.
    #     :param if_save: True: save files. False: load files.
    #     """

    #     def load_torch_file(model_or_optim, _path):
    #         state_dict = torch.load(_path, map_location=lambda storage, loc: storage)
    #         model_or_optim.load_state_dict(state_dict)

    #     name_obj_list = [
    #         ("actor", self.act),
    #         ("critic", self.cri),
    #     ]
    #     name_obj_list = [(name, obj) for name, obj in name_obj_list if obj is not None]
    #     if if_save:
    #         for name, obj in name_obj_list:
    #             save_path = f"{cwd}/{name}.pth"
    #             torch.save(obj.state_dict(), save_path)
    #     else:
    #         for name, obj in name_obj_list:
    #             save_path = f"{cwd}/{name}.pth"
    #             load_torch_file(obj, save_path) if os.path.isfile(save_path) else None