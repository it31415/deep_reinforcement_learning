import numpy as np
import gym
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

# 環境の生成と行動の数の取得
ENV_NAME = 'CartPole-v0'
env = gym.make(ENV_NAME)
nb_actions = env.action_space.n

# 多層ニューラルネットワークの構築
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# DQNエージェントの作成
memory = SequentialMemory(limit=5000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, memory=memory, policy=policy, nb_actions=nb_actions, nb_steps_warmup=10)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# 学習の実施
dqn.fit(env, nb_steps=50, visualize=False, verbose=2)

#ここに解答を書いてください
dqn.test(env, nb_episodes=5, visualize=False)
