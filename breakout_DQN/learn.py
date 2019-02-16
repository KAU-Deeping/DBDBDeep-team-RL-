import numpy as np
import random
import tensorflow as tf
import breakout_DQN.dqn

from typing import List
from collections import deque

class Learn:
    def __init__(self, main_model, target_model, discount_rate, env):
        self.memory = deque(maxlen=400000)
        self.history_size = (84, 84, 4)
        self.main_model = main_model
        self.target_model = target_model
        self.batch_size = 32
        self.env = env

        self.DISC_RATE = discount_rate

    def append_sample(self, history, action, reward, next_history, done):
        self.memory.append((history, action, reward, next_history, done))

    def get_memory_len(self):
        return len(self.memory)

    def replay_train(self):
        mini_batch = random.sample(self.memory, 32)

        history = np.zeros((self.batch_size, self.history_size[0], self.history_size[1], self.history_size[2]))

        # Make arrays for each data
        next_history = np.zeros((self.batch_size, self.history_size[0], self.history_size[1], self.history_size[2]))
        action, reward, done = [], [], []
        target = np.zeros((self.batch_size,))

        # Copy vars from mini_batch to each array
        for i in range(self.batch_size):
            # 확인 필요
            history[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            next_history[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        target_val =  self.target_model.predict(next_history)

        for i in range(self.batch_size):
            if done[i]:
                target[i] = reward[i]
            else:
                target[i] = reward[i] + self.DISC_RATE * np.argmax(target_val[i])

        #loss의 학습을 어떻게 해야하는가??
        loss = self.main_model.update(history, target_val)

    def get_action(self, history, global_step):
        epsilon = 1 - 0.9 / 999999 * (global_step - 1)
        if random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            action = self.main_model.predict(history)

        return action

    # Copy main DQN's args to target DQN
    def get_copy_var_ops(self, *, dest_scope_name: str, src_scope_name: str) -> List[tf.Operation]:
        """Creates TF operations that copy weights from `src_scope` to `dest_scope`
        Args:
            dest_scope_name (str): Destination weights (copy to)
            src_scope_name (str): Source weight (copy from)
        Returns:
            List[tf.Operation]: Update operations are created and returned
        """
        # Copy variables src_scope to dest_scope
        op_holder = []

        src_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
        dest_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

        for src_var, dest_var in zip(src_vars, dest_vars):
            op_holder.append(dest_var.assign(src_var.value()))

        return op_holder