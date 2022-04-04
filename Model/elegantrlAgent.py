# DRL models from ElegantRL: https://github.com/AI4Finance-Foundation/ElegantRL
from statistics import mode
from venv import create
import torch
import sys

sys.path.append('./Model')
from elegantrl_.agents.AgentDDPG import AgentDDPG
from elegantrl_.agents.AgentPPO import AgentPPO
from elegantrl_.agents.AgentSAC import AgentSAC
from elegantrl_.agents.AgentTD3 import AgentTD3
from elegantrl_.agents.AgentRandom import AgentRandom

# from elegantrl.agents.agent import AgentDDPG
# from elegantrl.agents.agent import AgentPPO
# from elegantrl.agents.agent import AgentSAC
# from elegantrl.agents.agent import AgentTD3

from elegantrl.train.config import Arguments
# from elegantrl.agents.agent import AgentA2C
from Model.elegantrl_.train_.run import train_and_evaluate, init_agent
import numpy as np
MODELS = {"ddpg": AgentDDPG, "td3": AgentTD3, "sac": AgentSAC, "ppo": AgentPPO, "random": AgentRandom}
OFF_POLICY_MODELS = ["ddpg", "td3", "sac","random"]
ON_POLICY_MODELS = ["ppo"]
"""MODEL_KWARGS = {x: config.__dict__[f"{x.upper()}_PARAMS"] for x in MODELS.keys()}
NOISE = {
    "normal": NormalActionNoise,
    "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise,
}"""


class DRLAgent:
    """Implementations of DRL algorithms
    Attributes
    ----------
        env: gym environment class
            user-defined class
    Methods
    -------
        get_model()
            setup DRL algorithms
        train_model()
            train DRL algorithms in a train dataset
            and output the trained model
        DRL_prediction()
            make a prediction in a test dataset and get results
    """

    def __init__(self, env, price_array, tech_array, turbulence_array, config):
        self.env = env
        self.price_array = price_array
        self.tech_array = tech_array
        self.turbulence_array = turbulence_array
        self.config = config

    def get_model(self, model_name, model_kwargs):
        env_config = {
            "price_array": self.price_array,
            "tech_array": self.tech_array,
            "turbulence_array": self.turbulence_array,
            "if_train": True,
        }
        env = self.env(config=env_config, initial_capital=self.config['INITIAL_CAPITAL'])
        env.env_num = 1
        agent = MODELS[model_name]
        self.model_name = model_name
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")
        model = Arguments(agent=agent, env=env)
        model.if_off_policy = model_name in OFF_POLICY_MODELS
        if model_kwargs is not None:
            try:
                model.learning_rate = model_kwargs["learning_rate"]
                model.batch_size = model_kwargs["batch_size"]
                model.gamma = model_kwargs["gamma"]
                model.seed = model_kwargs["seed"]
                model.net_dim = model_kwargs["net_dimension"]
                model.target_step = model_kwargs["target_step"]
                model.eval_gap = model_kwargs["eval_gap"]
                model.eval_times = model_kwargs["eval_times"]
                model.learner_gpus = model_kwargs["learner_gpus"]
                model.if_use_per = model_kwargs["if_use_per"]
                #self.devices = model_kwargs["learner_gpus"]
            except BaseException:
                raise ValueError(
                    "Fail to read arguments, please check 'model_kwargs' input."
                )
        return model

    def train_model(self, model, cwd, total_timesteps=5000):
        model.cwd = cwd
        model.break_step = total_timesteps
        train_and_evaluate(model)

    @staticmethod
    def DRL_prediction(model_name, cwd, net_dimension, environment, devices):
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")
        agent = MODELS[model_name]
        environment.env_num = 1
        args = Arguments(agent=agent, env=environment)
        args.cwd = cwd
        args.net_dim = net_dimension
        # load agent
        try:
            agent = init_agent(args, gpu_id=devices)
            act = agent.act
            cri = agent.cri 
            device = agent.device
        except BaseException:
            raise ValueError("Fail to load agent!")

        # test on the testing env
        _torch = torch
        state = environment.reset()
        episode_returns = []  # the cumulative_return / initial_account
        episode_total_assets = [environment.initial_total_asset]
        episode_sell_buy = np.zeros((1,environment.action_dim))
        rewards = []  #Just for logging
        
        action_values = [[0,0,0,0,0]]

        with _torch.no_grad():
            for i in range(environment.max_step):
                s_tensor = _torch.as_tensor((state,), device=device)
                a_tensor = act(s_tensor)  # action_tanh = act.forward()
                
                print("test action value:",cri(s_tensor,a_tensor).mean())
                ####################################################
                #This code block is used for evalutating the action of model
                #print("action:",a_tensor)
                if model_name in OFF_POLICY_MODELS:
                    #print("action value", cri(s_tensor, a_tensor))
                    action_value = []
                    for action in [[[-1]],[[-0.5]],[[0]],[[0.5]],[[1]]]:
                        action =  _torch.as_tensor((action), device=device)
                        action_value.append(cri(s_tensor, action)[0][0])
                    action_values.append(action_value)
                        
                elif model_name in ON_POLICY_MODELS:
                    print("state value", cri(s_tensor))
                #####################################################
                
                action = (
                    a_tensor.detach().cpu().numpy()[0]
                )  # not need detach(), because with torch.no_grad() outside
                state, reward, done, buy_sell_actions, _ = environment.step(action)

                total_asset = (
                        environment.amount
                        + (
                                environment.price_ary[environment.day] * environment.stocks
                        ).sum()
                )
                episode_total_assets.append(total_asset)
                episode_return = total_asset / environment.initial_total_asset
                episode_returns.append(episode_return)
                rewards.append(reward)
                buy_sell_actions = np.expand_dims(np.array(buy_sell_actions),axis=0)
                episode_sell_buy = np.concatenate([episode_sell_buy, buy_sell_actions], axis = 0)
                if done:
                    break
        

        #episode_sell_buy = np.array(episode_sell_buy)
        rewards.append(0)
        print("episode_sell_buy",episode_sell_buy.shape)
        print("Test Finished!")
        # return episode total_assets on testing data
        print("episode_return", episode_return)
        return episode_total_assets, episode_sell_buy, rewards, np.array(action_values)