import gym
import tensorflow as tf
import numpy as np
import random

import breakout_DQN.dqn as dqn
import breakout_DQN.learn as learn
import breakout_DQN.preprocessing as preprocessing

env = gym.make('BreakoutDeterministic-v4')
start_life = 5
total_episode = 500000
no_op_step = 30
train_start = 500000
update_target_rate = 10000

global_step = 0

with tf.Session() as sess:
    main_model = dqn.DQN(0.00025, "main", env.observation_space.shape, env.action_space.n,sess)
    target_model = dqn.DQN(0.00025, "target", env.observation_space.shape, env.action_space.n, sess)
    learn = learn.Learn(main_model, target_model, 0.99, env)

    for episode in range(total_episode):
        step = 0
        done = False
        life = start_life
        score = 0

        observe = env.reset()

        # 0~30 step 중 random한 step 동안 정지
        for _ in range(random.randint(1, no_op_step)):
            observe, _, _, _ = env.step(1)

        observe = preprocessing.preproc(observe)
        history = np.stack((observe, observe, observe, observe), axis=2)
        history = np.reshape([history], (1, 84, 84, 4))

        while not done:
            step += 1
            global_step += 1
            dead = False

            if dead:
                action = 1
                dead = False
            else:
                action = learn.get_action(history, global_step)

            observe, reward, done, info = env.step(action)
            score += reward

            if info['ale.lives'] != start_life:
                start_life = info['ale.lives']
                dead = True

            next_state = preprocessing.preproc(observe)
            next_state = np.reshape([next_state], (1, 84, 84, 1))
            next_history = np.append(next_state, history[:,:,:,:3], axis=3)

            learn.append_sample(history, action, reward, next_history, done)

            if learn.get_memory_len() >= train_start:
                learn.replay_train()
            if global_step % update_target_rate:
                learn.get_copy_var_ops(dest_scope_name="target", src_scope_name="main")

            history = next_history
            if done:
                print("Episode : {}, Reward : {}, Step : {}, Global Step : {}".format(episode, score, step, global_step))