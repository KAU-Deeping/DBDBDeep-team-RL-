from cartpole import learn
import collections
import random
import tensorflow as tf

# Save Parameter information
"""
    1 : DISCOUNT_RATE
    2 : REPLAY_MEMORY
    3 : BATCH_SIZE
    4 : Hidden layer size
    5 : Learning rate
    6 : Activation function
    7 : Episodes that needed to train
    8 : Success / Failed
"""
train_data = collections.deque()

# tf.math.tanh have #1, tf.nn. relu have #2
activation_func = ["tf.nn.relu", "tf.nn.tanh"]

for func in activation_func:
    for i in range(20):
        """
        Each vars are according to following descriptions...
            1 : DISCOUNT_RATE (0.70 ~ 0.99)
            2 : REPLAY_MEMORY (40000 ~ 50000)
            3 : BATCH_SIZE (32 ~ 64)
            4 : TARGET_UPDATE_FREQUENCY = 5
            5 : Hidden layer size (12 ~ 20)
            6 : Learning rate (0.01 ~ 0.001)
            7 : Activation function
            8 : Episodes that needed to train
            9 : Success / Failed
        """
        params = [random.uniform(0.70, 0.99), random.randint(40000, 50000), random.randint(32, 64), 5, random.randint(12, 20),
                  random.uniform(0.01, 0.001), func]

        # Create CartPole-v0 training model using params
        model = learn.model("Breakout-ram-v0", DISCOUNT_RATE=params[0], REPLAY_MEMORY=params[1], BATCH_SIZE=params[2], TARGET_UPDATE_FREQUENCY=params[3],
                            h_size=params[4], l_rate=params[5], activation=params[6], MAX_EPISODES=500)

        # Create list for saving episode and step data
        episode_data = []
        step_data = []

        result = model.train(episode_data, step_data)

        # Append ending epsode number
        params.append(result[0])

        # Append training was successful or not
        params.append(True if result[1] == 200 else False)

        print("Discount rate: {}\n"
              "Replay Memory : {}\n"
              "Batch size: {}\n"
              "Target Update Frequency: {}\n"
              "Hidden layer size: {}\n"
              "Learning rate: {}\n"
              "Activation function: {}\n"
              "Episodes that needed to train: {}\n"
              .format(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7]))

        train_data.append(params)

        # Show train result using graph
        model.plot(params, episode_data, step_data)

        # Clear graph for next training
        tf.reset_default_graph()