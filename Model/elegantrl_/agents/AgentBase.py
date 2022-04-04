import os
import torch
from copy import deepcopy


class AgentBase:
    def __init__(
        self, net_dim: int, state_dim: int, action_dim: int, gpu_id=0, args=None
    ):
        """initialize

        replace by different DRL algorithms
        explict call self.init() for multiprocessing.

        :param net_dim: the dimension of networks (the width of neural networks)
        :param state_dim: the dimension of state (the number of state vector)
        :param action_dim: the dimension of action (the number of discrete action)
        :param reward_scale: scale the reward to get a appropriate scale Q value
        :param gamma: the discount factor of Reinforcement Learning

        :param learning_rate: learning rate of optimizer
        :param if_per_or_gae: PER (off-policy) or GAE (on-policy) for sparse reward
        :param env_num: the env number of VectorEnv. env_num == 1 means don't use VectorEnv
        :param gpu_id: the gpu_id of the training device. Use CPU when cuda is not available.
        """
        self.gamma = getattr(args, "gamma", 0.99)
        self.env_num = getattr(args, "env_num", 1)
        self.batch_size = getattr(args, "batch_size", 128)
        self.repeat_times = getattr(args, "repeat_times", 1.0)
        self.reward_scale = getattr(args, "reward_scale", 1.0)
        self.lambda_gae_adv = getattr(args, "lambda_entropy", 0.98)
        self.if_use_old_traj = getattr(args, "if_use_old_traj", False)
        self.soft_update_tau = getattr(args, "soft_update_tau", 2**-8)

        if_act_target = getattr(args, "if_act_target", False)
        if_cri_target = getattr(args, "if_cri_target", False)
        if_off_policy = getattr(args, "if_off_policy", True)
        learning_rate = getattr(args, "learning_rate", 2**-12)

        self.states = None
        self.device = torch.device(
            f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu"
        )
        self.traj_list = [
            [list() for _ in range(4 if if_off_policy else 5)]
            for _ in range(self.env_num)
        ]  # for `self.explore_vec_env()`

        act_class = getattr(self, "act_class", None)
        cri_class = getattr(self, "cri_class", None)
        self.act = act_class(net_dim, state_dim, action_dim).to(self.device)
        self.cri = (
            cri_class(net_dim, state_dim, action_dim).to(self.device)
            if cri_class
            else self.act
        )
        self.act_target = deepcopy(self.act) if if_act_target else self.act
        self.cri_target = deepcopy(self.cri) if if_cri_target else self.cri

        self.act_optimizer = torch.optim.Adam(self.act.parameters(), learning_rate)
        self.cri_optimizer = (
            torch.optim.Adam(self.cri.parameters(), learning_rate)
            if cri_class
            else self.act_optimizer
        )

        """function"""
        self.criterion = torch.nn.SmoothL1Loss()

        if self.env_num == 1:
            self.explore_env = self.explore_one_env
        else:
            self.explore_env = self.explore_vec_env

        if getattr(
            args, "if_use_per", False
        ):  # PER (Prioritized Experience Replay) for sparse reward
            self.criterion = torch.nn.SmoothL1Loss(reduction="none")
            self.get_obj_critic = self.get_obj_critic_per
            print("Using get_obj_critic_per")
        else:
            self.criterion = torch.nn.SmoothL1Loss(reduction="mean")
            self.get_obj_critic = self.get_obj_critic_raw

    def explore_one_env(self, env, target_step) -> list:
        """actor explores in single Env, then returns the trajectory (env transitions) for ReplayBuffer

        :param env: RL training environment. env.reset() env.step()
        :param target_step: explored target_step number of step in env
        :return: `[traj_env_0, ]`
        `traj_env_0 = [(state, reward, mask, action, noise), ...]` for on-policy
        `traj_env_0 = [(state, other), ...]` for off-policy
        """
        traj_list = list()
        last_done = [
            0,
        ]
        state = self.states[0]

        step_i = 0
        done = False
        while step_i < target_step or not done:
            ten_s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            ten_a = (
                self.act.get_action(ten_s.to(self.device)).detach().cpu()
            )  # different
            next_s, reward, done, _ = env.step(ten_a[0].numpy())  # different

            traj_list.append((ten_s, reward, done, ten_a))  # different

            step_i += 1
            state = env.reset() if done else next_s

        self.states[0] = state
        last_done[0] = step_i
        return self.convert_trajectory(traj_list, last_done)  # traj_list

    def explore_vec_env(self, env, target_step) -> list:
        """actor explores in VectorEnv, then returns the trajectory (env transitions) for ReplayBuffer

        :param env: RL training environment. env.reset() env.step(). It should be a vector env.
        :param target_step: explored target_step number of step in env
        :return: `[traj_env_0, ]`
        `traj_env_0 = [(state, reward, mask, action, noise), ...]` for on-policy
        `traj_env_0 = [(state, other), ...]` for off-policy
        """
        traj_list = list()
        last_done = torch.zeros(self.env_num, dtype=torch.int, device=self.device)
        ten_s = self.states

        step_i = 0
        ten_dones = torch.zeros(self.env_num, dtype=torch.int, device=self.device)
        while step_i < target_step or not any(ten_dones):
            ten_a = self.act.get_action(ten_s).detach()  # different
            ten_s_next, ten_rewards, ten_dones, _ = env.step(ten_a)  # different

            traj_list.append(
                (ten_s.clone(), ten_rewards.clone(), ten_dones.clone(), ten_a)
            )  # different

            step_i += 1
            last_done[torch.where(ten_dones)[0]] = step_i  # behind `step_i+=1`
            ten_s = ten_s_next

        self.states = ten_s
        return self.convert_trajectory(traj_list, last_done)  # traj_list

    def convert_trajectory(self, buf_items, last_done):  # [ElegantRL.2022.01.01]
        """convert trajectory (env exploration type) to trajectory (replay buffer type)

        convert `other = concat((      reward, done, ...))`
        to      `other = concat((scale_reward, mask, ...))`

        :param traj_list: `traj_list = [(tensor_state, other_state), ...]`
        :return: `traj_list = [(tensor_state, other_state), ...]`
        """
        # assert len(buf_items) == step_i
        # assert len(buf_items[0]) in {4, 5}
        # assert len(buf_items[0][0]) == self.env_num
        buf_items = list(
            map(list, zip(*buf_items))
        )  # state, reward, done, action, noise
        # assert len(buf_items) == {4, 5}
        # assert len(buf_items[0]) == step
        # assert len(buf_items[0][0]) == self.env_num

        """stack items"""
        buf_items[0] = torch.stack(buf_items[0])
        buf_items[3:] = [torch.stack(item) for item in buf_items[3:]]

        if len(buf_items[3].shape) == 2:
            buf_items[3] = buf_items[3].unsqueeze(2)

        if self.env_num > 1:
            buf_items[1] = (torch.stack(buf_items[1]) * self.reward_scale).unsqueeze(2)
            buf_items[2] = ((~torch.stack(buf_items[2])) * self.gamma).unsqueeze(2)
        else:
            buf_items[1] = (
                (torch.tensor(buf_items[1], dtype=torch.float32) * self.reward_scale)
                .unsqueeze(1)
                .unsqueeze(2)
            )
            buf_items[2] = (
                ((1 - torch.tensor(buf_items[2], dtype=torch.float32)) * self.gamma)
                .unsqueeze(1)
                .unsqueeze(2)
            )
        # assert all([buf_item.shape[:2] == (step, self.env_num) for buf_item in buf_items])

        """splice items"""
        for j in range(len(buf_items)):
            cur_item = list()
            buf_item = buf_items[j]

            for env_i in range(self.env_num):
                last_step = last_done[env_i]

                pre_item = self.traj_list[env_i][j]
                if len(pre_item):
                    cur_item.append(pre_item)

                cur_item.append(buf_item[:last_step, env_i])

                if self.if_use_old_traj:
                    self.traj_list[env_i][j] = buf_item[last_step:, env_i]

            buf_items[j] = torch.vstack(cur_item)

        # on-policy:  buf_item = [states, rewards, dones, actions, noises]
        # off-policy: buf_item = [states, rewards, dones, actions]
        # buf_items = [buf_item, ...]
        return buf_items

    def get_obj_critic_raw(self, buffer, batch_size):
        """
        Calculate the loss of networks with **uniform sampling**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and states.
        """
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_a = self.act_target(next_s)
            critic_targets: torch.Tensor = self.cri_target(next_s, next_a)
            (next_q, min_indices) = torch.min(critic_targets, dim=1, keepdim=True)
            q_label = reward + mask * next_q

        q = self.cri(state, action)
        #print("q_label:", q_label)
        #print("q value:", q)
        obj_critic = self.criterion(q, q_label)
        #print("obj_critic:",obj_critic)

        return obj_critic, state

    def get_obj_critic_per(self, buffer, batch_size):
        """
        Calculate the loss of the network with **Prioritized Experience Replay (PER)**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and states.
        """
        with torch.no_grad():
            reward, mask, action, state, next_s, is_weights = buffer.sample_batch(
                batch_size
            )
            next_a = self.act_target(next_s)
            critic_targets: torch.Tensor = self.cri_target(next_s, next_a)
            # taking a minimum while preserving the dimension for possible twin critics
            (next_q, min_indices) = torch.min(critic_targets, dim=1, keepdim=True)
            q_label = reward + mask * next_q

        q = self.cri(state, action)
        td_error = self.criterion(q, q_label)
        obj_critic = (td_error * is_weights).mean()

        buffer.td_error_update(td_error.detach())
        return obj_critic, state

    @staticmethod
    def optimizer_update(optimizer, objective):
        """minimize the optimization objective via update the network parameters

        :param optimizer: `optimizer = torch.optim.SGD(net.parameters(), learning_rate)`
        :param objective: `objective = net(...)` the optimization objective, sometimes is a loss function.
        """
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

    @staticmethod
    def soft_update(target_net, current_net, tau):
        """soft update target network via current network

        :param target_net: update target network via current network to make training more stable.
        :param current_net: current network update via an optimizer
        :param tau: tau of soft target update: `target_net = target_net * (1-tau) + current_net * tau`
        """
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))

    def save_or_load_agent(self, cwd, if_save):
        """save or load training files for Agent

        :param cwd: Current Working Directory. ElegantRL save training files in CWD.
        :param if_save: True: save files. False: load files.
        """

        def load_torch_file(model_or_optim, _path):
            state_dict = torch.load(_path, map_location=lambda storage, loc: storage)
            model_or_optim.load_state_dict(state_dict)

        name_obj_list = [
            ("actor", self.act),
            ("act_target", self.act_target),
            ("act_optim", self.act_optimizer),
            ("critic", self.cri),
            ("cri_target", self.cri_target),
            ("cri_optim", self.cri_optimizer),
        ]
        name_obj_list = [(name, obj) for name, obj in name_obj_list if obj is not None]
        if if_save:
            for name, obj in name_obj_list:
                save_path = f"{cwd}/{name}.pth"
                torch.save(obj.state_dict(), save_path)
        else:
            for name, obj in name_obj_list:
                save_path = f"{cwd}/{name}.pth"
                print("Load from:", save_path)
                load_torch_file(obj, save_path) if os.path.isfile(save_path) else None
