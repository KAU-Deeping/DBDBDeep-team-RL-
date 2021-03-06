import gym
import numpy as np
import random
import threading
import time
import tensorflow as tf
from skimage.transform import resize
from skimage.color import rgb2gray
from keras.layers.convolutional import Conv2D
from keras.layers import Dense, Flatten, Input
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from keras import backend as K

episode = 0
EPISODES = 8000000

class A3CAgent:
    def __init__(self, action_size):
        self.state_size = (84, 84, 4)
        self.action_size = action_size

        self.actor_learning_rate = 0.00025
        self.critic_learning_rate = 0.00025
        self.disc_rate = 0.99
        self.rmsprop_decay = 0.99
        self.no_op_step = 20

        self.actor, self.critic = self.build_model()

    def build_model(self):
        actor = Sequential()
        critic = Sequential()

        actor.add(Conv2D(16, kernel_size=(8, 8), strides=(4, 4), input_shape=self.state_size))
        actor.add(Conv2D(32, kernel_size=(4, 4), strides=(2, 2)))
        actor.add(Flatten())

        actor.add(Dense(256, activation='relu'))
        actor.add(Dense(self.action_size, activation='softmax'))

        critic.add(Conv2D(16, kernel_size=(8, 8), strides=(4, 4), input_shape=self.state_size))
        critic.add(Conv2D(32, kernel_size=(4, 4), strides=(2, 2)))
        critic.add(Flatten())

        critic.add(Dense(256, activation='relu'))
        critic.add(Dense(1, activation='linear'))

        actor._make_predict_function()
        critic._make_predict_function()

        return actor, critic

    def get_action(self, history):
        history = np.float32(history / 255.)
        policy = self.actor.predict(history)[0]
        return np.argmax(policy)

    def load_model(self):
        self.actor.load_weights('./save_model/breakout_A3C_actor_downloaded.h5')

def pre_processing(next_observe, observe):
    processed_observe = np.maximum(next_observe, observe)
    processed_observe = np.uint8(
        resize(rgb2gray(processed_observe), (84, 84), mode='constant') * 255)
    return processed_observe

if __name__ == '__main__':
    env = gym.make('BreakoutDeterministic-v4')
    env.render()
    agent = A3CAgent(3)
    agent.load_model()

    step = 0

    while episode < EPISODES:
        dead = False
        done = False
        start_life = 5
        score = 0

        observe = env.reset()
        next_observe = observe

        for _ in range(random.randint(1, 20)):
            observe = next_observe
            next_observe, _, _, _ = env.step(1)

        state = pre_processing(next_observe, observe)
        history = np.stack((state, state, state, state), axis=2)
        history = np.reshape([history], (1, 84, 84, 4))

        while not done:
            env.render()
            step += 1
            observe = next_observe

            action = agent.get_action(history)

            if action == 1:
                fake_action = 2
            elif action == 2:
                fake_action = 3
            else:
                fake_action = 1

            if dead:
                fake_action = 1
                dead = False

            next_observe, reward, done, info = env.step(fake_action)
            next_state = pre_processing(next_observe, observe)
            next_state = np.reshape([next_state], (1, 84, 84, 1))
            next_history = np.append(next_state, history[:, :, :, :3], axis=3)

            if start_life > info['ale.lives']:
                dead = True
                reward = -1
                start_life = info['ale.lives']

            score += reward

            # if agent is dead, then reset the history
            if dead:
                history = np.stack(
                    (next_state, next_state, next_state, next_state), axis=2)
                history = np.reshape([history], (1, 84, 84, 4))
            else:
                history = next_history

            # if done, plot the score over episodes
            if done:
                episode += 1
                print("episode:", episode, "  score:", score, "  step:", step)
                step = 0