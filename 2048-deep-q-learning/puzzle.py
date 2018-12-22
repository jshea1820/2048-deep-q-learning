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
            activation = tf.nn.elu,
            kernel_initializer = tf.contrib.layers.xavier_initializer()
        )

        # Defines output layer
        self.output = tf.layers.dense(
            inputs = self.fc,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            units = 4,
            activation=None
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

        action = [up,down,left,right][action.tolist().index(1)]
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



def populate_memory(game, memory):

    for i in range(pretrain_length):
        
        # If it's the first step, flatten the board matrix and set state variable
        if i == 0:
            state = np.asarray([value for row in game.matrix for value in row])
        
        # Choose a random action and evaluate reward
        action = np.asarray(random.choice(possible_actions))
        reward, game_over = game.update_board(action)
        
        # If we're dead
        if game_over:
            # We finished the episode
            next_state = np.zeros(state.shape)
            
            # Add experience to memory
            memory.add((state, action, reward, next_state, game_over))
            
            # Start a new episode
            game.init_matrix()
            
            # First we need a state
            state = np.asarray([value for row in game.matrix for value in row])
            
        else:
            # Get the next state
            next_state = np.asarray([value for row in game.matrix for value in row])
            
            # Add experience to memory
            memory.add((state, action, reward, next_state, game_over))
            
            # Our state is now the next_state
            state = next_state
    
    return memory

def train(game, memory):

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
                decay_step +=1
                
                # Predict the action to take and take it
                action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step, state, sess)

                # Do the action
                reward, game_over = game.update_board(action)
                
                # Add the reward to total reward
                episode_rewards.append(reward)

                # If the game is finished
                if game_over:
                    # the episode ends so no next state
                    next_state = np.zeros(state.shape)

                    # Set step = max_steps to end the episode
                    step = max_steps

                    # Get the total reward of the episode
                    total_reward = np.sum(episode_rewards)

                    print('Episode: {}'.format(episode),
                            'Total reward: {}'.format(total_reward),
                            'Training loss: {:.4f}'.format(loss),
                            'Explore P: {:.4f}'.format(explore_probability))

                    memory.add((state, action, reward, next_state, game_over))

                else:
                    # Get the next state
                    next_state = np.asarray([value for row in game.matrix for value in row])
                    
                    # Add experience to memory
                    memory.add((state, action, reward, next_state, game_over))
                    
                    # st+1 is now our current state
                    state = next_state


                ### LEARNING PART            
                # Obtain random mini-batch from memory
                batch = memory.sample(batch_size)
                states_mb = np.array([each[0] for each in batch], ndmin=2)
                actions_mb = np.array([each[1] for each in batch])
                rewards_mb = np.array([each[2] for each in batch])
                next_states_mb = np.array([each[3] for each in batch], ndmin=2)
                dones_mb = np.array([each[4] for each in batch])

                target_Qs_batch = []

                # Get Q values for next_state 
                Qs_next_state = sess.run(DQNetwork.output, feed_dict = {DQNetwork.state_input: next_states_mb})
                
                # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma*maxQ(s', a')
                for i in range(0, len(batch)):
                    terminal = dones_mb[i]

                    # If we are in a terminal state, only equals reward
                    if terminal:
                        target_Qs_batch.append(rewards_mb[i])
                        
                    else:
                        target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                        target_Qs_batch.append(target)
                        

                targets_mb = np.array([each for each in target_Qs_batch])
                loss, _ = sess.run([DQNetwork.loss, DQNetwork.optimizer],
                                    feed_dict={DQNetwork.state_input: states_mb,
                                            DQNetwork.target_Q: targets_mb,
                                            DQNetwork.action_output: actions_mb})





def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, sess):
    ## EPSILON GREEDY STRATEGY
    # Choose action a from state s using epsilon greedy.
    ## First we randomize a number
    exp_exp_tradeoff = np.random.rand()

    # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
    
    if (explore_probability > exp_exp_tradeoff):
        # Make a random action (exploration)
        action = random.choice([0,1,2,3])
        action = np.asarray([0]*action + [1] + [0]*(3-action))
        
    else:
        # Get action from Q-network (exploitation)
        # Estimate the Qs values state
        Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.state_input: state.reshape(1,16)})
        
        # Take the biggest Q value (= the best action)
        choice = np.argmax(Qs)
        action = int(choice)
        action = np.asarray([0]*action + [1] + [0]*(3-action))

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
memory = Memory(max_size = memory_size)

# Create the environment
game = Game()

# Populate the memory buffer and reset game
memory = populate_memory(game, memory)
game = Game()

# Conduct training
train(game, memory)

