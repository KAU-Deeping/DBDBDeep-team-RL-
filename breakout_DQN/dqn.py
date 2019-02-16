import tensorflow as tf

class DQN:
    def __init__(self, learning_rate: int, name: str, input_size: int, output_size: int, session):
        self.LEARNING_RATE = learning_rate
        self.NAME = name
        self.INPUT_SIZE = input_size
        self.OUTPUT_SIZE = output_size
        self.session = session

        self.build_model()

    def build_model(self, learning_rate, h_size):
        with tf.VariableScope(name=self.NAME):
            self._X = tf.placeholder(dtype=tf.float32, shape=[None, 84, 84, 4], name='INPUT_DATA')

            conv1 = tf.layers.conv2d(inputs=self._X, filters=32, strides=[4, 4], kernel_size=[8, 8], activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            conv2 = tf.layers.conv2d(inputs=conv1, filters=64, strides=[2, 2], kernel_size=[4, 4],
                                     activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            conv3 = tf.layers.conv2d(inputs=conv2, filters=64, strides=[1, 1], kernel_size=[3, 3],
                                     activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())

            flat = tf.layers.flatten(conv3)

            fc = tf.layers.dense(inputs=flat, units=512, activation=tf.nn.relu)
            output = tf.layers.dense(input=flat, units=self.OUTPUT_SIZE)
            self._Qpred = output

            self._Y = tf.placeholder(dtype=tf.float32, shape=self.OUTPUT_SIZE, name='Y')

            self._loss = tf.losses.mean_squared_error(self._Y, self._Qpred)

            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self._train = optimizer.minimize(self._loss)

    def predict(self, X):
        return self.session.run(self._Qpred, feed_dict={self._X: X})

    def update(self, x, y):
        return self.session.run([self._loss, self._train], feed_dict={self._X: x, self._Y: y})