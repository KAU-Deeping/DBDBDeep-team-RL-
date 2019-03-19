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

global episode
episode = 0
EPISODES = 8000000

class A3CAgent:
    def __init__(self, action_size, thread_num):
        self.state_size = (84, 84, 4)
        self.action_size = action_size

        self.actor_learning_rate = 0.00025
        self.critic_learning_rate = 0.00025
        self.thread_num = thread_num
        self.disc_rate = 0.99
        self.rmsprop_decay = 0.99
        self.no_op_step = 20

        self.actor, self.critic = self.build_model()
        self.optimizer = [self.actor_optimize(), self.critic_optimize()]

        self.tmax = 5

        self.config = tf.ConfigProto(intra_op_parallelism_threads=16,
                                inter_op_parallelism_threads=16,
                                allow_soft_placement=True,
                                device_count={'CPU': 1, 'GPU': 0})

        self.sess = tf.InteractiveSession(config=self.config)
        K.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())

        self.summary_placeholders, self.update_ops, self.summary_op = \
            self.setup_summary()
        self.summary_writer = \
            tf.summary.FileWriter('summary/breakout_a3c', self.sess.graph)

    def build_model(self):
        #with tf.device("/cpu:0"):
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

    def actor_optimize(self):
        action = K.placeholder(shape=(None, self.action_size))
        advantage = K.placeholder(shape=(None,))

        policy = self.actor.output

        action_prob = K.sum(action * policy, axis=1)
        cross_entropy = K.log(action_prob + 1e-10) * advantage
        cross_entropy = -K.sum(cross_entropy)

        entropy = K.sum(policy * K.log(policy + 1e-10), axis=1)
        entropy = K.sum(entropy)

        loss = cross_entropy + 0.01 * entropy

        optimizer = RMSprop(lr=self.actor_learning_rate, rho=0.99, epsilon=0.01)
        update = optimizer.get_updates(self.actor.trainable_weights, [], loss)
        train = K.function((self.actor.input, action, advantage), [loss], updates=update)

        return train

    def critic_optimize(self):
        target = K.placeholder(shape=(None,))

        loss = K.mean(K.square(target - self.critic.output))

        optimizer = RMSprop(lr=self.critic_learning_rate, rho=0.99, epsilon=0.01)
        update = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        train = K.function([self.critic.input, target], [loss], updates=update)

        return train

    def save_model(self, file_path):
        self.actor.save_weights(file_path + '_actor.h5')
        self.critic.save_weights(file_path + '_critic.h5')

    def load_model(self, file_path):
        self.actor.load_weights(file_path + '_actor.h5')
        self.critic.load_weights(file_path + '_critic.h5')

    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        episode_avg_max_q = tf.Variable(0.)
        episode_duration = tf.Variable(0.)

        tf.summary.scalar('Total Reward/Episode', episode_total_reward)
        tf.summary.scalar('Average Max Prob/Episode', episode_avg_max_q)
        tf.summary.scalar('Duration/Episode', episode_duration)

        summary_vars = [episode_total_reward,
                        episode_avg_max_q,
                        episode_duration]

        summary_placeholders = [tf.placeholder(tf.float32)
                                for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i])
                      for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op

    def train(self):
        agents = [Agent(self.action_size, self.state_size,
                        [self.actor, self.critic], self.sess,
                        self.optimizer, self.disc_rate,
                        [self.summary_op, self.summary_placeholders,
                         self.update_ops, self.summary_writer])
                  for _ in range(self.thread_num)]

        for agent in agents:
            time.sleep(1)
            agent.start()

        # 10분(600초)에 한번씩 모델을 저장
        while True:
            time.sleep(60 * 10)
            self.save_model("./save_model/breakout_a3c")

class Agent(threading.Thread):
    def __init__(self, action_size, state_size, model, sess, optimizer, discount_factor, summary_ops):
        threading.Thread.__init__(self)

        # A3CAgent 클래스에서 상속
        self.action_size = action_size
        self.state_size = state_size
        self.actor, self.critic = model
        self.sess = sess
        self.optimizer = optimizer
        self.discount_factor = discount_factor
        [self.summary_op, self.summary_placeholders,
         self.update_ops, self.summary_writer] = summary_ops

        self.states, self.actions, self.rewards = [], [], []

        self.local_actor, self.local_critic = self.build_local_model()

        self.avg_p_max = 0
        self.avg_loss = 0

        self.t_max = 20
        self.t = 0

    def run(self):
        global episode

        step = 0

        env = gym.make('BreakoutDeterministic-v4')

        while episode < EPISODES:
            done = False
            dead = False

            score, start_life = 0, 5
            observe = env.reset()
            next_observe = observe

            for _ in range(random.randint(1, 30)):
                observe = next_observe
                next_observe, _, _, _ = env.step(1)

            state = pre_processing(next_observe, observe)
            history = np.stack((state, state, state, state), axis=2)
            history = np.reshape([history], (1, 84, 84, 4))

            while not done:
                step += 1
                self.t += 1
                observe = next_observe

                action, policy = self.get_action(history)

                if action == 0:
                    real_action = 1
                elif action == 1:
                    real_action = 2
                else:
                    real_action = 3

                if dead:
                    action = 0
                    real_action = 1
                    dead = False

                self.avg_p_max += np.amax(self.actor.predict(
                    np.float32(history / 255.)))

                next_state, reward, done, info = env.step(real_action)
                next_state = pre_processing(next_observe, observe)
                next_state = np.reshape([next_state], (1, 84, 84, 1))
                next_history = np.append(next_state, history[:, :, :, :3], axis=3)

                if start_life > info['ale.lives']:
                    dead = True
                    start_life = info['ale.lives']

                score += reward
                reward = np.clip(reward, -1., 1.)
                self.append_sample(history, action, reward)

                if dead:
                    history = np.stack((next_state, next_state,
                                        next_state, next_state), axis=2)
                    history = np.reshape([history], (1, 84, 84, 4))
                else:
                    history = next_history

                if self.t > self.t_max or done:
                    self.train_model(done)
                    self.update_local_weights()

                if done:
                    episode += 1
                    print("episode:", episode, "  score:", score, "  step:",
                          step)

                    stats = [score, self.avg_p_max / float(step),
                             step]
                    for i in range(len(stats)):
                        self.sess.run(self.update_ops[i], feed_dict={
                            self.summary_placeholders[i]: float(stats[i])
                        })
                    summary_str = self.sess.run(self.summary_op)
                    self.summary_writer.add_summary(summary_str, episode + 1)
                    self.avg_p_max = 0
                    self.avg_loss = 0
                    step = 0

    def train_model(self, done):
            discounted_prediction = self.discounted_prediction(self.rewards, done)

            states = np.zeros((len(self.states), 84, 84, 4))
            for i in range(len(self.states)):
                states[i] = self.states[i]

            states = np.float32(states / 255.)

            values = self.critic.predict(states)
            values = np.reshape(values, len(values))

            advantages = discounted_prediction - values

            self.optimizer[0]([states, self.actions, advantages])
            self.optimizer[1]([states, discounted_prediction])
            self.states, self.actions, self.rewards = [], [], []

    def build_local_model(self):
        #with tf.device("/cpu:0"):
        input = Input(shape=self.state_size)
        conv = Conv2D(16, (8, 8), strides=(4, 4), activation='relu')(input)
        conv = Conv2D(32, (4, 4), strides=(2, 2), activation='relu')(conv)
        conv = Flatten()(conv)
        fc = Dense(256, activation='relu')(conv)
        policy = Dense(self.action_size, activation='softmax')(fc)
        value = Dense(1, activation='linear')(fc)

        local_actor = Model(inputs=input, outputs=policy)
        local_critic = Model(inputs=input, outputs=value)

        local_actor._make_predict_function()
        local_critic._make_predict_function()

        local_actor.set_weights(self.actor.get_weights())
        local_critic.set_weights(self.critic.get_weights())

        local_actor.summary()
        local_critic.summary()

        return local_actor, local_critic

    def update_local_weights(self):
        self.local_actor.set_weights(self.actor.get_weights())
        self.local_critic.set_weights(self.critic.get_weights())

    def get_action(self, history):
        history = np.float32(history / 255.0)
        policy = self.local_actor.predict(history)[0]
        action_index = np.random.choice(self.action_size, 1, p=policy)[0]
        return action_index, policy

    def append_sample(self, history, action, reward):
        self.states.append(history)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)
        self.rewards.append(reward)

    def discounted_prediction(self, rewards, done):
        discounted_prediction = np.zeros_like(rewards)
        running_add = 0

        if not done:
            running_add = self.critic.predict(np.float32(
                self.states[-1] / 255.))[0]

        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_prediction[t] = running_add
        return discounted_prediction

def pre_processing(next_observe, observe):
    processed_observe = np.maximum(next_observe, observe)
    processed_observe = np.uint8(
        resize(rgb2gray(processed_observe), (84, 84), mode='constant') * 255)
    return processed_observe

if __name__ == "__main__":
    # For MacBook
    #global_agent = A3CAgent(action_size=3, thread_num=8)

    # For Desktop
    global_agent = A3CAgent(action_size=3, thread_num=16)
    global_agent.train()