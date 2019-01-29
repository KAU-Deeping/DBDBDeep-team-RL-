import tensorflow as tf
import gym
import numpy as np

env = gym.make("Breakout-ram-v0")

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "./breakout_model.ckpt")

    state = env.reset()
    reward_sum = 0
    #미완성(Model 재사용)
'''
    while True:

        env.render()
        X = tf.placeholder(tf.float32, [None, env.observation_space.shape[0]], name="input_x")
        net1 = X

        net1 = tf.layers.dense(net1, h_size, tf.nn.relu)
        net2 = tf.layers.dense(net1, h_size, tf.nn.relu)

        output = tf.layers.dense(net2, self.output_size)
        self._Qpred = output
        x = np.reshape(state, [-1, self.input_size])
        action = self.session.run(self._Qpred, feed_dict={self._X: x})
        action = np.argmax(sess.predict(state))
        state, reward, done, _ = env.step(action)
        reward_sum += reward

        if done:
            print("Total score: {}".format(reward_sum))
            break
'''