def assetReward(enviroment):
    total_asset = enviroment.amount + (enviroment.stocks * price).sum()
    reward = (total_asset - self.total_asset) * self.reward_scaling