from gym_anytrading.envs import StocksEnv
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C, DQN, PPO2
import pandas as pd

df = pd.read_csv('data/Google_data_5_extra_features.csv')


def add_signals(env):
    start = env.frame_bound[0] - env.window_size
    end = env.frame_bound[1]
    prices = env.df.loc[:, 'Low'].to_numpy()[start:end]
    signal_features = env.df.loc[:, ['Low', 'Volume', 'SMA', 'RSI', 'OBV', 'ADX', 'STOCH']].to_numpy()[
        start:end]
    return prices, signal_features


class MyCustomEnv(StocksEnv):
    _process_data = add_signals


env2 = MyCustomEnv(df=df, window_size=12, frame_bound=(12, 80))


def env_maker(): return env2


env = DummyVecEnv([env_maker])

# A2C, DQN, PPO2
algorithm = A2C

# MlpLstmPolicy, MlpPolicy
policy = 'MlpLstmPolicy'
save_name = 'models/A2C_Mlp.zip'

model = algorithm(policy, env, verbose=1,
                  tensorboard_log="./training_logs/")
model.learn(total_timesteps=400000)

model.save(save_name)

# To watch logs use:
# `tensorboard --logdir ./training_logs/`
