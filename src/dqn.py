import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
from gym.wrappers.time_limit import TimeLimit
from collections import deque
import numpy as np
import pandas as pd
import random

tf.reset_default_graph()
class DQN():
    def __init__(self, init_epsilon, final_epsilon, env: TimeLimit, replay_size, train_start_size, gamma=0.9):
        self.epsilon_step =(init_epsilon - final_epsilon) / 10000
        self.final_epsilon = final_epsilon
        self.action_dim = env.action_space.n
        self.replay_size = replay_size
        self.train_start_size = train_start_size
        self.gamma = gamma

        self.env = env

        self.state_dim = env.observation_space.shape[0]
        self.n_hidden = 128
        self.batch_size = 32

        self.replay_buffer = deque()
        self.epsilon = init_epsilon
        self.config()

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    def batch_norm_layer(self, value):
        return batch_norm(value, decay=0.9, updates_collections=None, is_training=True)

    def add_layer(self, inputs, in_size, out_size, activation_function=None, layer_name='Layer'):
        with tf.name_scope(layer_name):
            with tf.name_scope("Weights"):
                weights = tf.Variable(tf.random_normal([in_size, out_size]))
            with tf.name_scope("Biases"):
                biases = tf.Variable(tf.random_normal([out_size]))
            with tf.name_scope("fx"):
                fx = self.batch_norm_layer(tf.matmul(inputs, weights) + biases)
            if activation_function is None:
                output = fx
            else:
                output = activation_function(fx)

            return output, weights, biases

    def config(self):
        self.input_x = tf.placeholder(tf.float32, [None, self.state_dim])
        self.input_action = tf.placeholder(tf.float32, [None, self.action_dim])
        self.input_y = tf.placeholder(tf.float32, [None])

        h_0, w_0, b_0 = self.add_layer(self.input_x, self.state_dim, self.n_hidden, activation_function=tf.nn.relu, layer_name="Hidden_1")
        # h_1, w_1, b_1 = self.add_layer(h_0, self.n_hidden, self.n_hidden, activation_function=tf.nn.relu, layer_name="Hidden_2")
        self.Q_value, _, _ = self.add_layer(h_0, self.n_hidden, self.action_dim, activation_function=None, layer_name="Output")

        # value = tf.reduce_sum(tf.multiply(self.Q_value, self.input_action), reduction_indices=1)
        value = tf.reduce_max(tf.multiply(self.Q_value, self.input_action), reduction_indices=1)

        with tf.name_scope("loss"):
            loss = tf.reduce_mean(tf.square(value - self.input_y))
        with tf.name_scope("Adam"):
            self.optimizer = tf.train.AdamOptimizer(0.1).minimize(loss)

    def percieve(self, state, action, reward, next_state, done):
        one_hot_action = np.zeros([self.action_dim])
        one_hot_action[action] = 1
        self.replay_buffer.append([state, one_hot_action, reward, next_state, done])

        if len(self.replay_buffer) > self.replay_size:
            self.replay_buffer.popleft()
        elif len(self.replay_buffer) > self.train_start_size:
            self.train()

    def train(self):
        gamma = self.gamma
        batch_data = random.sample(self.replay_buffer, self.batch_size)
        batch_data = np.array(batch_data)
        state = np.array([i for i in batch_data[:, 0]])
        action = np.array([i for i in batch_data[:, 1]])
        reward = np.array(batch_data[:, 2])
        next_state = np.array([i for i in batch_data[:, 3]])
        done = np.array(batch_data[:, 4])
        next_state_reward = self.sess.run(self.Q_value, feed_dict={self.input_x: next_state})

        batch_y = [-100*reward[i] if done[i] else reward[i] + gamma*np.max(next_state_reward[i]) for i in range(len(reward))]

        _ = self.sess.run(self.optimizer, feed_dict={self.input_x: state, self.input_y: batch_y, self.input_action: action})

    def get_greedy_action(self, state):
        """
        [[ -8.322552 -84.55124 ]]
        :param state:
        :return:
        """
        value = self.sess.run(self.Q_value, feed_dict={self.input_x: [state]})[0]
        return np.argmax(value)

    def get_action(self, state):
        if self.epsilon > self.final_epsilon:
            self.epsilon -= self.epsilon_step
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim-1)
        else:
            return self.get_greedy_action(state)
