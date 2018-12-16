import numpy as np
import gym
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


ENV_NAME = 'CartPole-v0'


# 環境の生成と行動の数の取得
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
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
model.add(Dense(nb_actions, activation='linear'))
print(model.summary())

# DQNエージェントの作成
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()

# ここに解答を書いてください
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10, enable_dueling_network=True, dueling_type='avg', target_model_update=1e-2, policy=policy)

dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# 学習の実施
dqn.fit(env, nb_steps=100, visualize=False, verbose=2)

# 最終学習結果を保存する場合には、以下のコメントアウトを実行してください。
#dqn.save_weights('duel_dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# テストの実施
dqn.test(env, nb_episodes=2, visualize=False)
