from enums import *
import random
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

class DeepQlearning:
    def __init__(self, learning_rate=0.1, discount=0.95, exploration_rate=1.0, iterations= 100000):
        self.learning_rate = learning_rate
        self.discount = discount
        self.exploration_rate = 1.0 
        self.exploration_delta = 1/iterations 

        self.input_count = 36
        self.output_count = 4

        self.session = tf.Session()
        self.define_model()
        self.session.run(self.initializer)
        self.length = 6 

    def define_model(self):
        self.model_input = tf.placeholder(dtype=tf.float32, shape=[None, self.input_count])

        fc1 = tf.layers.dense(self.model_input, 16, activation=tf.sigmoid, kernel_initializer=tf.constant_initializer(np.zeros((self.input_count, 16))))
        fc2 = tf.layers.dense(fc1, 16, activation=tf.sigmoid, kernel_initializer=tf.constant_initializer(np.zeros((16, self.output_count))))

        self.model_output = tf.layers.dense(fc2, self.output_count)

        self.target_output = tf.placeholder(shape=[None, self.output_count], dtype=tf.float32)

        loss = tf.losses.mean_squared_error(self.target_output, self.model_output)
       
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(loss)
        
        self.initializer = tf.global_variables_initializer()

    def get_Q(self, state):
        return self.session.run(self.model_output, feed_dict={self.model_input: self.to_one_hot(state)})[0]

    def to_one_hot(self, state):
        one_hot = np.zeros((1, 36))
        #print(state)
        one_hot[0, [state[0] + state[1]*6]] = 1
        #print(one_hot)
        return one_hot

    def get_next_action(self, state):
        if random.random() > self.exploration_rate: 
            return self.greedy_action(state)
        else:
            return self.random_action()

    def greedy_action(self, state):
        #print(self.get_Q(state))
        return np.argmax(self.get_Q(state))

    def random_action(self):
        r = random.random()
        if(r < 0.25):
            return UP
        elif (r < 0.5):
            return DOWN
        elif (r < 0.75):
            return LEFT
        else:
            return RIGHT

    def reset(self,  old_state, action, reward, new_state):
        old_state_Q_values = self.get_Q(old_state)
        new_state_Q_values = [0,0,0,0,0,0]
        #print(reward)
        old_state_Q_values[action] = reward + self.discount * np.amax(new_state_Q_values)
        training_input = self.to_one_hot(old_state)
        target_output = [old_state_Q_values]
        training_data = {self.model_input: training_input, self.target_output: target_output}
        self.session.run(self.optimizer, feed_dict=training_data)
        self.exploration_rate = 1.0

    def train(self, old_state, action, reward, new_state):
        old_state_Q_values = self.get_Q(old_state)
   
        new_state_Q_values = self.get_Q(new_state)
        
        old_state_Q_values[action] = reward + self.discount * np.amax(new_state_Q_values)
       
        training_input = self.to_one_hot(old_state)
        
        target_output = [old_state_Q_values]
        
        training_data = {self.model_input: training_input, self.target_output: target_output}

        self.session.run(self.optimizer, feed_dict=training_data)

    def update(self, old_state, new_state, action, reward):
       
        if(reward != -100):
            self.train(old_state, action, reward, new_state)
        else:
            self.reset(old_state, action, reward, new_state)
            return False

        if self.exploration_rate > 0:
            self.exploration_rate -= self.exploration_delta
        
        return True