import tensorflow as tf
import gym
import numpy as np

env = gym.make("Breakout-v0")

# Shape[0]이 맞는지 확인 필요
input_size = env.observation_space.shape[0]
output_size = env.action_space.n

# Make model
X = tf.placeholder(tf.float32, [None, env.observation_space.shape[0]], name="input_x")
net1 = X

# Set appropriate h_size selected by learn_with_different_params
net1 = tf.layers.dense(net1, 10, tf.nn.relu)
net2 = tf.layers.dense(net1, 10, tf.nn.relu)

output = tf.layers.dense(net2, output_size)
Qpred = output

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('./breakout_model.ckpt.meta')
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "./breakout_model.ckpt")

    state = env.reset()
    reward_sum = 0


    #미완성(Model 재사용)
    while True:

        env.render()

        # Reshape x for predict next action
        x = np.reshape(state, [-1, input_size])

        # Calculate action with model
        action = np.argmax(sess.run(Qpred, feed_dict={X: x}))

        # Get reward, done, next state information
        state, reward, done, _ = env.step(action)
        reward_sum += reward

        if done:
            print("Total score: {}".format(reward_sum))
            break