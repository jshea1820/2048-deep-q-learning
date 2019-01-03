import tensorflow as tf      # Deep Learning library
import numpy as np           # Handle matrices
from collections import deque# Ordered collection with ends

from tkinter import *
from logic import *
import random
import time
import math

class DQNetwork:
    ''' Class for a Deep-Q network '''

    def __init__(self, state_size, action_size, learning_rate):

        # Sets network hyperparameters
        self.state_size = state_size     # Number of parameters specifying the state
        self.action_size = action_size    # Number of potential actions
        self.learning_rate = learning_rate

        # Creates input placeholders
        self.state_input = tf.placeholder(tf.float32, [None, state_size])
        self.action_input = tf.placeholder(tf.float32, [None, action_size])
        
        # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
        self.target_Q = tf.placeholder(tf.float32, [None])

        # Defines fully connected layers
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
            activation = tf.nn.sigmoid,
            kernel_initializer=tf.contrib.layers.xavier_initializer()
        )

        # Q is our predicted Q value.
        self.Q = tf.reduce_sum(tf.multiply(self.output, self.action_input), axis=1)

        # The loss is the difference between our predicted Q_values and the Q_target
        # Sum(Qtarget - Q)^2
        self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)


class Memory():
    ''' Memory buffer that holds (state, action, reward, next_state) tuples '''

    def __init__(self, max_size):
        ''' Initializes an empty buffer '''
        self.buffer = deque(maxlen = max_size)
    
    def add(self, experience):
        ''' Adds an experience tuple to the memory buffer '''
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        ''' Samples a random batch of memory tuples from the buffer '''
        buffer_size = len(self.buffer)
        index = np.random.choice(
            np.arange(buffer_size),
            size = batch_size,
            replace = False
        )
        return [self.buffer[i] for i in index]


class Environment:
    ''' Environment class defining the behavior of the environment where the agent will explore'''

    def __init__(self, state_size, action_size):
        ''' Sets the hyperparameters of the environment '''
        self.state_size = state_size
        self.action_size = action_size
        
        # Map indcating what is present at each position
        # 0 -> empty space, -1 -> wall, 20 -> key, 100 -> door
        self.map = [
            [0,   0,   0,   0,   0],
            [0,  -1,   0,  -1,   0],
            [0,  -1,   0,  -1,   0],
            [0,   0,   0,   0,   0],
            [20,  0,  -1,   0, 100],
        ]

    def reset(self):
        ''' Resets the state of the environment and returns this initial state '''

        self.position = (0,0) # (row, column)
        self.retrieved_key = False # Boolean flag as to whether or not key has been gotten

        # Creates one-hot state tuple indicating state of environment
        state = [0]*self.state_size
        state[0] = 1
        return np.asarray(state)

    def get_valid_actions(self):
        ''' Looks at current position and evaluates actions that are valid '''

        valid_actions_array = [1] * self.action_size

        # Invalidates action if at edge or if there is a wall
        if self.position[0] - 1 < 0 or self.map[self.position[0] - 1][self.position[1]] == -1:
            valid_actions_array[0] = 0
        if self.position[0] + 1 > 4 or self.map[self.position[0] + 1][self.position[1]] == -1:
            valid_actions_array[1] = 0
        if self.position[1] - 1 < 0 or self.map[self.position[0]][self.position[1] - 1] == -1:
            valid_actions_array[2] = 0
        if self.position[1] + 1 > 4 or self.map[self.position[0]][self.position[1] + 1] == -1:
            valid_actions_array[3] = 0

        # Reorients valid actions
        valid_actions = []
        for i in range(action_size):
            if valid_actions_array[i]:
                valid_actions.append(possible_actions[i])

        return valid_actions

    def update(self, action):
        ''' Updates position based on action and returns (reward, game_over) tuple '''
        ''' Action is assumed to be already valid '''

        if action[0] == 1:
            # Up
            self.position = (self.position[0] - 1, self.position[1])
        elif action[1] == 1:
            # Down
            self.position = (self.position[0] + 1, self.position[1])
        elif action[2] == 1:
            # Left
            self.position = (self.position[0], self.position[1] - 1)
        elif action[3] == 1:
            # Right
            self.position = (self.position[0], self.position[1] + 1)
        
        if not self.retrieved_key and self.position == (4,0):
            self.retrieved_key = True
            return (20, False)
        if self.retrieved_key and self.position == (4,4):
            return (100, True)

        return (-1, False)

    def get_state(self):

        state = [0]*self.state_size
        state[self.position[0] * 5 + self.position[1]] = 1
        state[self.state_size - 1] = int(self.retrieved_key)

        return np.asarray(state)

    def __str__(self):
        for i in range(5):
            if self.position[0] == i:
                to_display = [x for x in self.map[i]]
                to_display[self.position[1]] = "X"
                print(to_display)
            else:
                print(self.map[i])
        if self.retrieved_key:
            print("KEY RETRIEVED")
        else:
            print("KEY NOT RETRIEVED")
        return ""


def populate_memory(environment, memory):

    for i in range(pretrain_length):
        
        # If it's the first step, reset the environment
        if i == 0:
            state = environment.reset()

        #print(environment)
        #print("State value: {}".format(environment.get_state()))
        
        # Filters possible_actions array to just valid actions
        valid_actions = environment.get_valid_actions()
        #print("Valid actions: {}".format(valid_actions))

        # Chooses action, performs action, gets reward
        action = random.choice(valid_actions)
        #print("Chosen action")
        reward, game_over = environment.update(action)
        #print("Reward: {}".format(reward))

        # Retrieves the environment state
        next_state = environment.get_state()

        # Adds expereience tuple to memory
        memory.add((state, action, reward, next_state))
        
        # If the episode is complete, reset the state
        if game_over:
            state = environment.reset()
            
        else:
            # Our state is now the next_state
            state = next_state
    
    return memory


def train(environment, memory, net):

    saver = tf.train.Saver()
    with tf.Session() as sess:

        # Initialize the weights of the network
        sess.run(tf.global_variables_initializer())
        
        # Initialize the decay rate (that will use to reduce epsilon)
        decay_step = 0

        # Trains the network
        all_losses = []
        for episode in range(total_episodes):
            print("Episode {}".format(episode))

            # Resets the step count
            step = 0
            
            # List to track the rewards gained throughout the episode
            episode_rewards = []
            
            # Make a new episode and observe the first state
            state = environment.reset()

            episode_losses = []

            while step < max_steps:

                step += 1
                
                # Increase decay_step
                decay_step += 1

                #if episode == 499:
                    #print(environment)
                
                # Predict the action to take and take it
                action = predict_action(explore_start, explore_stop, decay_rate, decay_step, state, sess, episode)
                #if episode == 499:
                    #print("Action chosen: {}".format(action))

                # Do the action and retrieve next state
                reward, game_over = environment.update(action)
                #if episode == 499:
                    #print("Reward: {}".format(reward))
                next_state = environment.get_state()
                
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
                actions_mb = np.array([each[1] for each in batch])
                rewards_mb = np.array([each[2] for each in batch])
                next_states_mb = np.array([each[3] for each in batch], ndmin=2)

                target_Qs_batch = []

                # Get Q values for next_state
                Qs_next_state = sess.run(net.output, feed_dict = {net.state_input: next_states_mb})
                
                # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma*maxQ(s', a')
                for i in range(0, len(batch)):

                    target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                    target_Qs_batch.append(target)
                        

                targets_mb = np.array([each for each in target_Qs_batch])
                loss, _ = sess.run([net.loss, net.optimizer],
                                    feed_dict={net.state_input: states_mb,
                                            net.target_Q: targets_mb,
                                            net.action_input: actions_mb})
                                
                episode_losses.append(loss)
            all_losses.append(sum(episode_losses) / len(episode_losses))
        to_file(all_losses)
        save_path = saver.save(sess, "./model.ckpt")



def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, sess, episode):
    ## EPSILON GREEDY STRATEGY
    # Choose action a from state s using epsilon greedy.
    ## First we randomize a number
    exp_exp_tradeoff = np.random.rand()

    # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

    # Filters possible actions to only valid actions
    valid_actions = environment.get_valid_actions()
    
    if (explore_probability > exp_exp_tradeoff):
        # Make a random action (exploration)
        #if episode == 499:
            #print("Choosing to explore")
        action = random.choice(valid_actions)
        
    else:
        # Get action from Q-network (exploitation)
        # Estimate the Qs values state
        #if episode == 499:
            #print("Choosing to exploit")
        Qs = sess.run(net.output, feed_dict = {net.state_input: [state]})[0].tolist()
        #if episode == 499:
            #print(Qs)
        sorted_Qs = sorted(Qs)
        for i in range(action_size):
            next_best_action = possible_actions[Qs.index(sorted_Qs[action_size - 1 - i])]
            #if episode == 499:
                #print(next_best_action)
            if next_best_action in valid_actions:
                action = next_best_action
                break
        
        #if episode == 499:
            #print(action)

    return action


def perform_test(network, sess):
    print("Testing")
    environment = Environment(state_size, action_size)
    state = environment.reset()
    print(environment)
    game_over = False
    rewards = 0
    while not game_over:
        valid_actions = environment.get_valid_actions()
        print(state)
        Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.state_input: [state]})[0].tolist()
        print("Qs = {}".format(Qs))
        sorted_Qs = sorted(Qs)
        for i in range(action_size):
            next_best_action = possible_actions[Qs.index(sorted_Qs[action_size - 1 - i])]
            if next_best_action in valid_actions:
                action = next_best_action
                break
        print(action)
        reward, game_over = environment.update(action)
        print(environment)
        state = environment.get_state()
        rewards += reward
        if rewards == -100:
            game_over = True
    print("Reward = {}".format(rewards))

def to_file(l):
    f = open("losses.txt", "w")
    f.write(str(l))
    f.close()
    

### MODEL HYPERPARAMETERS
state_size = 26
action_size = 4
learning_rate =  0.0002      # Alpha (aka learning rate)

# One-hot list of matrix containing actions
possible_actions = []
for i in range(action_size):
    possible_actions.append([0]*i + [1] + [0]*(action_size - i - 1))

### TRAINING HYPERPARAMETERS
total_episodes = 500        # Total episodes for training
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
net = DQNetwork(state_size, action_size, learning_rate)
memory = Memory(max_size = memory_size)

# Create the environment
environment = Environment(state_size, action_size)

# Populate the memory buffer and reset game
memory = populate_memory(environment, memory)

# Conduct training
train(environment, memory, net)

