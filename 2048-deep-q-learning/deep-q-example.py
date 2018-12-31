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
        self.fc1 = tf.layers.dense(
            inputs = self.state_input,
            units = 512,
            kernel_initializer = tf.contrib.layers.xavier_initializer()
        )

        self.fc2 = tf.layers.dense(
            inputs = self.fc1,
            units = 256,
            kernel_initializer = tf.contrib.layers.xavier_initializer()
        )

        # Defines output layer
        self.output = tf.layers.dense(
            inputs = self.fc2,
            units = 4,
            kernel_initializer=tf.contrib.layers.xavier_initializer()
        )

        # Q is our predicted Q value.
        self.Q = tf.reduce_sum(tf.multiply(self.output, self.action_output), axis=1)

        # The loss is the difference between our predicted Q_values and the Q_target
        # Sum(Qtarget - Q)^2
        self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)


class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen = max_size)
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(
            np.arange(buffer_size),
            size = batch_size,
            replace = False
        )
        return [self.buffer[i] for i in index]


class Environment:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.state = [0]*state_size
        self.map = [
            [0,   0,   0,   0,   0],
            [0,  -1,   0,  -1,   0],
            [0,  -1,   0,  -1,   0],
            [0,   0,   0,   0,   0],
            [20,  0,  -1,   0, 100],
        ]

    def reset(self):

        self.position = (0,0)
        self.retrieved_key = False

        self.state = [0]*state_size
        self.state[0] = 
        self.state = np.asarray(self.state)

        return self.state

    def valid_actions(self):

        





def populate_memory(environment, memory):

    for i in range(pretrain_length):
        
        # If it's the first step, reset the environment
        if i == 0:
            state = environment.reset()
        
        # Choose a random action and evaluate reward
        action = random.choice(possible_actions).index(1)
        reward, episode_done = environment.update(action)
        next_state = environment.get_state()
        memory.add((state, action, reward, next_state))
        
        # If the episode is complete, reset the state
        if episode_done:

            state = environment.reset()
            
        else:

            # Our state is now the next_state
            state = next_state
    
    return memory


def train(environment, memory):

    with tf.Session() as sess:

        # Initialize the weights of the network
        sess.run(tf.global_variables_initializer())
        
        # Initialize the decay rate (that will use to reduce epsilon)
        decay_step = 0

        # Trains the network
        for episode in range(total_episodes):
            if episode % 10 == 0:
                print("Episode {}".format(episode))

            # Resets the step count
            step = 0
            
            # List to track the rewards gained throughout the episode
            episode_rewards = []
            
            # Make a new episode and observe the first state
            game.init_matrix()
            state = convert_matrix(game.matrix)

            while step < max_steps:

                step += 1
                
                # Increase decay_step
                decay_step += 1
                
                # Predict the action to take and take it
                action = predict_action(explore_start, explore_stop, decay_rate, decay_step, state, sess)

                # Do the action and retrieve next state
                reward, game_over = game.update_board(action)
                next_state = convert_matrix(game.matrix)
                
                # Add the reward to total reward
                episode_rewards.append(reward)

                # If the game is finished, end the episode and accumulate rewards
                if game_over:

                    # Set step = max_steps to end the episode
                    step = max_steps

                    # Get the total reward of the episode
                    total_reward = np.sum(episode_rewards)
                    
                # Add experience to memory
                memory.add((state, action, reward, next_state))
                    
                # st+1 is now our current state
                state = next_state

                ### LEARNING PART
                # Obtain random mini-batch from memory
                batch = memory.sample(batch_size)
                states_mb = np.array([each[0] for each in batch], ndmin=2)
                actions_mb = np.array([possible_actions[each[1]] for each in batch])
                rewards_mb = np.array([each[2] for each in batch])
                next_states_mb = np.array([each[3] for each in batch], ndmin=2)

                target_Qs_batch = []

                # Get Q values for next_state
                Qs_next_state = sess.run(DQNetwork.output, feed_dict = {DQNetwork.state_input: next_states_mb})
                
                # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma*maxQ(s', a')
                for i in range(0, len(batch)):

                    target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                    target_Qs_batch.append(target)
                        

                targets_mb = np.array([each for each in target_Qs_batch])
                loss, _ = sess.run([DQNetwork.loss, DQNetwork.optimizer],
                                    feed_dict={DQNetwork.state_input: states_mb,
                                            DQNetwork.target_Q: targets_mb,
                                            DQNetwork.action_output: actions_mb})
            losses.append(loss)

        print(losses)
        to_file(losses)




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

    return action


def convert_matrix(matrix):

    # Flattens matrix
    matrix_value = [value for row in matrix for value in row]

    # Converts 0s to 1s
    for i, val in enumerate(matrix_value):
        if val == 0:
            matrix_value[i] = 1
    
    return np.asarray([math.log2(x) for x in matrix_value])

def to_file(l):
    f = open("losses.txt", "w")
    f.write(str(l))
    f.close()
    

### MODEL HYPERPARAMETERS
state_size = 25
action_size = 4
learning_rate =  0.0002      # Alpha (aka learning rate)

# One-hot list of matrix containing actions
possible_actions = []
for i in range(action_size):
    possible_actions.append([0]*i + [1] + [0]*(action_size - i - 1))

### TRAINING HYPERPARAMETERS
total_episodes = 100        # Total episodes for training
max_steps = 100              # Max possible steps in an episode
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
memory = Memory(max_size = memory_size)

# Create the environment
environment = Environment(state_size, action_size)

# Populate the memory buffer and reset game
memory = populate_memory(environment, memory)

# Conduct training
train(environment, memory)

