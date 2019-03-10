import gym
import numpy as np
import pylab
import random
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential

EPISODES = 500

class DQNAgent:
    def __init__(self, input_size, output_size, load_model=False):
        self.load_model = load_model

        self.input_size = input_size
        self.output_size = output_size

        self.learning_rate = 0.001
        self.discount_rate = 0.99
        self.epsilon = 1.
        self.epsilon_decay = .9999
        self.epsilon_min = 0.01
        self.target_update_frequency = 5
        self.batch_size = 64
        self.replay_memory = 500000
        self.buffer = deque(maxlen=self.replay_memory)

        self.main_model = self.build_model()
        self.target_model = self.build_model()

        if load_model:
            self.model.load_weights('./save_model/cartpole_deep_q_learning_trained.h5')
            self.epsilon = 0.1

    def build_model(self):
        model = Sequential()
        model.add(Dense(512,input_dim=self.input_size, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.output_size, activation='linear'))
        model.compile(Adam(lr=self.learning_rate), loss='mse')
        return model

    def get_action(self, state):
        if random.rand() <= self.epsilon:
            return random.randrange(self.output_size)
        else:
            return np.argmax(self.model.predict(state)[0])

    def append_memory(self, state, action, reward, next_state, done):
        self.buffer.append(state, action, reward, next_state, done)

    def replay_train(self, mini_batch):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        