import gym
import numpy as np
import sys
import pylab
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras import backend as K

EPISODES = 400

class A2CAgent:
    def __init__(self, state_size, action_size, load_model=False):
        self.load_model = load_model
        self.state_size = state_size
        self.action_size = action_size

        self.actor_learning_rate = 0.001
        self.critic_learning_rate = 0.005
        self.discount_rate = 0.99

        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.actor_updater = self.actor_optimize()
        self.critic_updater = self.critic_optimize()

        # For Tensorboard
        self.avg_advantage = 0
        self.avg_critic_loss = 0

        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)

        self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
        self.summary_writer = tf.summary.FileWriter('summary/', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        if self.load_model:
            self.actor.load_weights('./save_model/A2C_trained_actor.h5')
            self.critic.load_weights('./save_model/A2C_trained_critic.h5')

    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        episode_avg_advantage = tf.Variable(0.)
        episode_avg_critic_loss = tf.Variable(0.)

        tf.summary.scalar('Total Reward/Episode', episode_total_reward)
        tf.summary.scalar('Average Advantage/Episode', episode_avg_advantage)
        tf.summary.scalar('Average Critic Loss/Episode', episode_avg_critic_loss)

        summary_vars = [episode_total_reward, episode_avg_advantage, episode_avg_critic_loss]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op

    def build_actor(self):
        actor = Sequential()
        actor.add(Dense(24, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        actor.add(Dense(self.action_size, activation='softmax', kernel_initializer='he_uniform'))
        actor.summary()

        return actor

    def build_critic(self):
        critic = Sequential()
        critic.add(Dense(24, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        critic.add(Dense(24, activation='relu', kernel_initializer='he_uniform'))
        critic.add(Dense(self.action_size, activation='linear', kernel_initializer='he_uniform'))
        critic.summary()

        return critic

    def get_action(self, state):
        state = np.reshape(state, (1, self.state_size))
        policy = self.actor.predict(state).flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0]

    def actor_optimize(self):
        action = K.placeholder([None, self.action_size])
        advantage = K.placeholder([None, ])

        action_probs = K.sum(action * self.actor.output, axis=1)
        cross_entropy = K.log(action_probs) * advantage
        loss = - K.sum(cross_entropy)

        optimizer = Adam(lr=self.actor_learning_rate)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], loss)
        train = K.function([self.actor.input, action, advantage], [], updates=updates)

        return train

    def critic_optimize(self):
        target = K.placeholder([None,])

        loss = K.mean(K.square(target - self.critic.output))

        optimizer = Adam(lr=self.critic_learning_rate)
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        train = K.function([self.critic.input, target], [], updates=updates)

        return train

    def train_model(self, state, action, reward, next_state, done):
        state = np.reshape(state, (1, self.state_size))
        next_state = np.reshape(next_state, (1, self.state_size))
        value = self.critic.predict(state)[0]
        next_value = self.critic.predict(next_state)[0]

        act = np.zeros([1, self.action_size])
        act[0][action] = 1

        if done:
            advantage = reward - value
            target = [reward]
        else:
            advantage = reward + self.discount_rate * next_value - value
            target = reward + self.discount_rate * next_value

        self.avg_advantage += advantage
        self.actor_updater([state, act, advantage])
        self.critic_updater([state, target])

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    agent = A2CAgent(env.observation_space.shape[0], env.action_space.n)
    scores, episodes = [], []

    for e in range(EPISODES):
        state = env.reset()
        done = False
        score = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)

            reward = reward if not done or score == 499 else -100
            agent.train_model(state, action, reward, next_state, done)

            score += reward
            state = next_state

            if done:
                if score == 500:
                    continue
                else:
                    score += 100

                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, color='black')
                pylab.savefig('./save_graph/a2c_trained.png')
                print("Episode ", e, " Score ", score)

            if np.mean(scores[-min(10, len(scores)):]) > 490 or e == 500:
                agent.actor.save_weights('./save_model/A2C_trained_actor.h5')
                agent.critic.save_weights('./save_model/A2C_trained_critic.h5')
                sys.exit()