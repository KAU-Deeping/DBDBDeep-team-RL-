import sys
import gym
import pylab
import random
import numpy as np
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential

EPISODES = 500

class SARSAAgent:
    def __init__(self, state_size, action_size, load_model):
        self.load_model = False
        self.state_size = state_size
        self.action_size = action_size

        self.discount_factor = 0.99
        self.learning_rate = 0.01
        self.epsilon = 1.
        self.epsilon_decay = .9999
        self.epsilon_min = 0.01
        self.model = self.build_model()

        if self.load_model:
            self.model.load_weights('./save_model/deep_sarsa_trained.h5')

    def build_model(self):
        model = Sequential()
        model.add(Dense(30, input_dim=self.state_size, activation='relu'))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(Adam(lr=self.learning_rate), loss='mse')
        return model

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = np.float32(state)
            state = np.reshape(state, (1, self.state_size))
            q_val = self.model.predict(state)
            return np.argmax(q_val[0])

    def train_model(self, state, action, reward, next_state, next_action, done):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        state = np.float32(state)
        state = np.reshape(state, (1, self.state_size))
        next_state = np.float32(next_state)
        next_state = np.reshape(next_state, (1, self.state_size))
        target = self.model.predict(state)[0]

        if done:
            target[action] = reward
        else:
            target[action] = reward + self.discount_factor * self.model.predict(next_state)[0][next_action]

        target = np.reshape(target, [1, self.action_size])
        self.model.fit(state, target, verbose=0)

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    agent = SARSAAgent(env.observation_space.shape[0], env.action_space.n, False)

    global_step = 0
    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()

        while not done:
            global_step += 1
            action = agent.get_action(state)

            next_state, reward, done, _ = env.step(action)

            next_action = agent.get_action(next_state)
            agent.train_model(state, action, reward, next_state, next_action, done)

            state = next_state
            score += reward

            if done:
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./save_graph/deep-sarsa.png")
                print("Episode : ", e, "Score : ", score, "Global Step : ", global_step)

            if e % 100 == 0:
                agent.model.save_weights('./save_model/deep_sarsa_trained.h5')