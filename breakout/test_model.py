import tensorflow as tf
import gym
import numpy as np

from skimage.transform import resize
from skimage.color import rgb2gray

def pre_proc(x):
    x = np.uint8(resize(rgb2gray(x), (84, 84), mode='constant') * 255)
    return x

env = gym.make("BreakoutDeterministic-v0")

input_size = (84, 84, 4)
output_size = env.action_space.n

# Make model
X = tf.placeholder(tf.float32, shape=(None, 84, 84, 4), name="input_x")
input_data = X

# Set appropriate h_size selected by learn_with_different_params
conv1 = tf.layers.conv2d(inputs=input_data, filters=64, kernel_size=[8, 8], strides=[8, 8], activation=tf.nn.relu)
conv2 = tf.layers.conv2d(inputs=conv1, filters=32, kernel_size=[4, 4], strides=[4, 4], activation=tf.nn.relu)
conv3 = tf.layers.conv2d(inputs=conv2, filters=32, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)

flat = tf.layers.flatten(conv3)

net1 = tf.layers.dense(flat, 128, tf.nn.relu)
net2 = tf.layers.dense(net1, 128, tf.nn.relu)

output = tf.layers.dense(net2, output_size)
Qpred = output

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('./breakout_model.ckpt.meta')
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "./breakout_model.ckpt")

    state = env.reset()
    if np.shape(state) == (1, 84, 84, 1):
        pass
    else:
        state = pre_proc(state)

    history = np.stack((state, state, state, state), axis=2)
    history = np.reshape([history], (1, 84, 84, 4))
    reward_sum = 0


    #미완성(Model 재사용)
    while True:
        env.render()

        # Reshape x for predict next action
        x = np.reshape(history, newshape=[-1, 84, 84, 4])

        # Calculate action with model
        action = np.argmax(sess.run(Qpred, feed_dict={X: x}))

        # Get reward, done, next state information
        state, reward, done, _ = env.step(action)

        # Pre processing next_states
        state = pre_proc(state)
        state = np.reshape(state, (1, 84, 84, 1))
        history = np.append(state, history[:, :, :, :3], axis=3)

        reward_sum += reward

        if done:
            print("Total score: {}".format(reward_sum))
            break

