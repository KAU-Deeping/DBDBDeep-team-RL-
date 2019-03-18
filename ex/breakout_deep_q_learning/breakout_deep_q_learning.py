import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import RMSprop
from keras.models import Sequential
from skimage.color import rgb2gray
from skimage.transform import resize
import keras.backend as K

EPISODE = 50000

class DQNAgent:
    def __init__(self, state_size, action_size, load_model=False):
        self.load_model = load_model

        self.state_size = state_size
        self.action_size = action_size

        self.learning_rate = 0.00025
        self.disc_rate = 0.99
        self.target_update_freq = 10000
        self.epsilon = 1
        self.epsilon_start, self.epsilon_end = 1.0, 0.1
        self.exploration_steps = 1000000.
        self.epsilon_decay_step = (self.epsilon_start - self.epsilon_end) / self.exploration_steps
        self.batch_size = 32
        self.train_start = 50000
        self.memory = deque(maxlen=500000)
        self.no_op_steps = 30

        self.main_model = self.build_model()
        self.target_model = self.build_model()
        self.optimizer = self.optimizer()
        self.update_target_model()

        # 텐서보드 설정
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)

        self.avg_q_max, self.avg_loss = 0, 0
        self.summary_placeholders, self.update_ops, self.summary_op = \
            self.setup_summary()
        self.summary_writer = tf.summary.FileWriter(
            'summary/breakout_dqn', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        if self.load_model:
            self.main_model.load_weights("./save_model/breakout_DQN_trained.h5")

    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        episode_avg_max_q = tf.Variable(0.)
        episode_duration = tf.Variable(0.)
        episode_avg_loss = tf.Variable(0.)

        tf.summary.scalar('Total Reward/Episode', episode_total_reward)
        tf.summary.scalar('Average Max Q/Episode', episode_avg_max_q)
        tf.summary.scalar('Duration/Episode', episode_duration)
        tf.summary.scalar('Average Loss/Episode', episode_avg_loss)

        summary_vars = [episode_total_reward, episode_avg_max_q,
                        episode_duration, episode_avg_loss]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in
                                range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in
                      range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op

    def get_action(self, history):
        history = np.float32(history / 255.0)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.main_model.predict(history)[0])

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=self.state_size))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu', input_shape=self.state_size))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.summary()
        return model

    def append_memory(self, history, action, reward, next_history, dead):
        self.memory.append((history, action, reward, next_history, dead))

    def update_target_model(self):
        self.target_model.set_weights(self.main_model.get_weights())

    def optimizer(self):
        a = K.placeholder(shape=(None,), dtype='int32')
        y = K.placeholder(shape=(None,), dtype='float32')

        prediction = self.main_model.output

        a_one_hot = K.one_hot(a, self.action_size)
        q_value = K.sum(prediction * a_one_hot, axis=1)
        error = K.abs(y - q_value)

        quadratic_part = K.clip(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)

        optimizer = RMSprop(lr=self.learning_rate, epsilon=0.01)
        updates = optimizer.get_updates(self.main_model.trainable_weights, [], loss)
        train = K.function([self.main_model.input, a, y], [loss], updates=updates)

        return train

    def train(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step

        mini_batch = random.sample(self.memory, self.batch_size)

        historys = np.zeros((self.batch_size, self.state_size[0], self.state_size[1], self.state_size[2]))
        next_historys = np.zeros((self.batch_size, self.state_size[0], self.state_size[1], self.state_size[2]))
        targets = np.zeros((self.batch_size,))
        actions, rewards, deads = [], [], []

        for i in range(self.batch_size):
            historys[i] = np.float32(mini_batch[i][0] / 255.0)
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_historys[i] = np.float32(mini_batch[i][3] / 255.0)
            deads.append(mini_batch[i][4])

        target_val = self.target_model.predict(next_historys)

        for i in range(self.batch_size):
            if deads[i]:
                targets[i] = rewards[i]
            else:
                targets[i] = rewards[i] + self.disc_rate * np.amax(target_val[i])

        loss = self.optimizer([historys, actions, targets])
        self.avg_loss += loss[0]

def pre_proc(observation):
    observation = rgb2gray(observation)
    observation = resize(observation, (84, 84), mode='constant') * 255
    proc_observe = np.uint8(observation)
    return proc_observe

if __name__ == '__main__':
    env = gym.make('BreakoutDeterministic-v4')
    agent = DQNAgent((84, 84, 4), 3)

    scores, episodes, global_step = [], [], 0

    for e in range(EPISODE):
        done = False
        dead = False

        step, score, start_life = 0, 0, 5
        observe = env.reset()

        for _ in range(random.randint(1, agent.no_op_steps)):
            observe, _, _, _ = env.step(1)

        state = pre_proc(observe)
        history = np.stack((state, state, state, state), axis=2)
        history = np.reshape([history], (1, 84, 84, 4))

        while not done:
            global_step += 1
            step += 1

            action = agent.get_action(history)

            if action == 0:
                real_action = 1
            elif action == 1:
                real_action = 2
            else:
                real_action = 3

            observe, reward, done, info = env.step(action)
            next_state = pre_proc(observe)
            next_state = np.reshape([next_state], (1, 84, 84, 1))
            next_history = np.append(next_state, history[:, :, :, :3], axis=3)

            agent.avg_q_max += np.amax(
                agent.main_model.predict(np.float32(history / 255.))[0])

            if start_life > info['ale.lives']:
                dead = True
                start_life = info['ale.lives']

            reward = np.clip(reward, -1., 1.)
            # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장 후 학습
            agent.append_memory(history, action, reward, next_history, dead)

            if len(agent.memory) >= agent.train_start:
                agent.train()

            # 일정 시간마다 타겟모델을 모델의 가중치로 업데이트
            if global_step % agent.target_update_freq == 0:
                agent.update_target_model()

            score += reward

            if dead:
                dead = False
            else:
                history = next_history

            if done:
                # 각 에피소드 당 학습 정보를 기록
                if global_step > agent.train_start:
                    stats = [score, agent.avg_q_max / float(step), step,
                             agent.avg_loss / float(step)]
                    for i in range(len(stats)):
                        agent.sess.run(agent.update_ops[i], feed_dict={
                            agent.summary_placeholders[i]: float(stats[i])
                        })
                    summary_str = agent.sess.run(agent.summary_op)
                    agent.summary_writer.add_summary(summary_str, e + 1)

                print("episode:", e, "  score:", score, "  memory length:",
                      len(agent.memory), "  epsilon:", agent.epsilon,
                      "  global_step:", global_step, "  average_q:",
                      agent.avg_q_max / float(step), "  average loss:",
                      agent.avg_loss / float(step))

                agent.avg_q_max, agent.avg_loss = 0, 0

            # 1000 에피소드마다 모델 저장
        if e % 1000 == 0:
            agent.main_model.save_weights("./save_model/breakout_DQN_trained.h5")