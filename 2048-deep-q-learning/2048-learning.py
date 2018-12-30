import tensorflow as tf      # Deep Learning library
import numpy as np           # Handle matrices
from collections import deque# Ordered collection with ends

from tkinter import *
from logic import *
import random
import time
import math

class DQNetwork:
    def __init__(self, state_size, action_size, learning_rate):

        # Sets network hyperparameters
        self.state_size = state_size     # Number of parameters specifying the state (16)
        self.action_size = action_size    # Number of potential actions (4)
        self.learning_rate = learning_rate

        # Creates input/output placeholders
        self.state_input = tf.placeholder(tf.float32, [None, state_size])
        self.action_output = tf.placeholder(tf.float32, [None, action_size])
        
        # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
        self.target_Q = tf.placeholder(tf.float32, [None])

        # Defines fully connected layer
        self.fc = tf.layers.dense(
            inputs = self.state_input,
            units = 256,
            kernel_initializer = tf.contrib.layers.xavier_initializer()
        )

        # Defines output layer
        self.output = tf.layers.dense(
            inputs = self.fc,
            units = 4,
            kernel_initializer=tf.contrib.layers.xavier_initializer()
        )

        # Q is our predicted Q value.
        self.Q = tf.reduce_sum(tf.multiply(self.output, self.action_output), axis=1)

        # The loss is the difference between our predicted Q_values and the Q_target
        # Sum(Qtarget - Q)^2
        self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)


class Game:
    def __init__(self):

        self.init_matrix()

    def gen(self):
        return randint(0, GRID_LEN - 1)

    def init_matrix(self):
        self.matrix = new_game(4)

        self.matrix=add_two(self.matrix)
        self.matrix=add_two(self.matrix)


    def update_board(self, action):

        action = [up,down,left,right][action]
        self.matrix, done, d_score = action(self.matrix)
        game_over = False
        if done:
            self.matrix = add_two(self.matrix)
            done=False
            if game_state(self.matrix)=='win' or game_state(self.matrix)=='lose':
                game_over = True

        return [d_score, game_over]


    def generate_next(self):
        index = (self.gen(), self.gen())
        while self.matrix[index[0]][index[1]] != 0:
            index = (self.gen(), self.gen())
        self.matrix[index[0]][index[1]] = 2

    def __str__(self):

        return str(self.matrix[0]) + "\n" + str(self.matrix[1]) + "\n" + str(self.matrix[2]) + "\n" + str(self.matrix[3])


def train(game):

    with tf.Session() as sess:

        # Initialize the weights of the network
        sess.run(tf.global_variables_initializer())
        
        # Initialize the decay rate (that will use to reduce epsilon)
        decay_step = 0

        # Trains the network
        for episode in range(total_episodes):

            # Resets the step count
            step = 0
            
            # List to track the rewards gained throughout the episode
            episode_rewards = []
            
            # Make a new episode and observe the first state
            game.init_matrix()
            state = np.asarray([value for row in game.matrix for value in row])

            while step < max_steps:
                step += 1
                
                # Increase decay_step
                decay_step += 1
                
                # Decide next action to take
                action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step, state, sess)

                # Perform action and get reward
                reward, game_over = game.update_board(action)
                
                # Add the reward to total reward
                episode_rewards.append(reward)

                # If the game is finished, end the episode and aggregate rewards
                if game_over:

                    # Set step = max_steps to end the episode
                    step = max_steps

                    # Get the total reward of the episode
                    total_reward = np.sum(episode_rewards)

                # Get the next state value
                next_state = np.asarray([value for row in game.matrix for value in row])

                # Gets all the Q values for actions from the next state
                Qs_next_state = sess.run(DQNetwork.output, feed_dict = {DQNetwork.state_input: [next_state]})

                # Target value is reward gained + all expected future rewards
                target = reward + gamma * np.max(Qs_next_state)

                action_vect = [0, 0, 0, 0]
                action_vect[action] = 1
                loss, _ = sess.run([DQNetwork.loss, DQNetwork.optimizer],
                                    feed_dict={DQNetwork.state_input: [state],
                                            DQNetwork.target_Q: [target],
                                            DQNetwork.action_output: [action_vect]})

            print("Episode complete, loss: {}".format(loss))



def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, sess):
    ## EPSILON GREEDY STRATEGY
    # Choose action a from state s using epsilon greedy.
    ## First we randomize a number
    exp_exp_tradeoff = np.random.rand()

    # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
    
    if (explore_probability > exp_exp_tradeoff):
        # Make a random action (exploration)
        action = random.randint(0,3)
        
    else:
        # Get action from Q-network (exploitation)
        # Estimate the Qs values state
        Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.state_input: [state]})
        
        # Take the biggest Q value (= the best action)
        choice = np.argmax(Qs)
        action = int(choice)

    return action, explore_probability

    

### MODEL HYPERPARAMETERS
state_size = 16
action_size = 4
learning_rate =  0.0002      # Alpha (aka learning rate)
possible_actions = [    # one-hot matrix indicating chosen action
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,0],
    [0,0,0,1]
]

### TRAINING HYPERPARAMETERS
total_episodes = 500        # Total episodes for training
max_steps = 5000              # Max possible steps in an episode
batch_size = 64

# Exploration parameters for epsilon greedy strategy
explore_start = 1.0            # exploration probability at start
explore_stop = 0.01            # minimum exploration probability 
decay_rate = 0.0001            # exponential decay rate for exploration prob

# Q learning hyperparameters
gamma = 0.95               # Discounting rate

### MEMORY HYPERPARAMETERS
pretrain_length = batch_size   # Number of experiences stored in the Memory when initialized for the first time
memory_size = 1000000          # Number of experiences the Memory can keep

# Creates network and memory
DQNetwork = DQNetwork(state_size, action_size, learning_rate)

# Create the environment
game = Game()

# Conduct training
train(game)