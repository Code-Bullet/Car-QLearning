import pyglet
from pyglet.gl import *
import pygame
import math
from pyglet.window import key
from Drawer import Drawer
# from PygameAdditionalMethods import *
from ShapeObjects import Line
import tensorflow as tf  # Deep Learning library
import numpy as np  # Handle matrices
from collections import deque
import random
import os
from Globals import displayHeight, displayWidth
from Game import Game

frameRate = 30.0

vec2 = pygame.math.Vector2

game = Game()
possible_actions = np.identity(game.no_of_actions, dtype=int).tolist()

### MODEL HYPERPARAMETERS
state_size = [game.state_size]  # Our input is a stack of 4 frames hence 100x120x4 (Width, height, channels)
action_size = game.no_of_actions  # 7 possible actions
learning_rate = 0.00025  # Alpha (aka learning rate)

### TRAINING HYPERPARAMETERS
total_episodes = 50000  # Total episodes for training
max_steps = 5000  # Max possible steps in an episode
batch_size = 64

# FIXED Q TARGETS HYPERPARAMETERS
max_tau = 10000  # Tau is the C step where we update our target network

# EXPLORATION HYPERPARAMETERS for epsilon greedy strategy
explore_start = 1.0  # exploration probability at start
explore_stop = 0.01  # minimum exploration probability
decay_rate = 0.00005  # exponential decay rate for exploration prob

# Q LEARNING hyperparameters
gamma = 0.95  # Discounting rate

### MEMORY HYPERPARAMETERS
## If you have GPU change to 1million
memory_size = 100000  # Number of experiences the Memory can keep
pretrain_length = memory_size  # Number of experiences stored in the Memory when initialized for the first time

### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
training =  False

## TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
episode_render = True

load = True

starting_episode = 0

load_traing_model = False

load_training_model_number = 300


class DDDQNNet:
    def __init__(self, state_size, action_size, learning_rate, name):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.name = name

        # We use tf.variable_scope here to know which network we're using (DQN or target_net)
        # it will be useful when we will update our w- parameters (by copy the DQN parameters)
        with tf.variable_scope(self.name):
            # We create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            # [None, 100, 120, 4]
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")

            #
            self.ISWeights_ = tf.placeholder(tf.float32, [None, 1], name='IS_weights')

            self.actions_ = tf.placeholder(tf.float32, [None, action_size], name="actions_")

            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")

            self.dense1 = tf.layers.dense(inputs=self.inputs_,
                                          units=256,
                                          activation=tf.nn.elu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name="dense1")

            self.dense2 = tf.layers.dense(inputs=self.dense1,
                                          units=256,
                                          activation=tf.nn.elu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name="dense2")

            ## Here we separate into two streams
            # The one that calculate V(s)
            self.value_fc = tf.layers.dense(inputs=self.dense2,
                                            units=256,
                                            activation=tf.nn.elu,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                            name="value_fc")

            self.value = tf.layers.dense(inputs=self.value_fc,
                                         units=1,
                                         activation=None,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name="value")

            # The one that calculate A(s,a)
            self.advantage_fc = tf.layers.dense(inputs=self.dense2,
                                                units=256,
                                                activation=tf.nn.elu,
                                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                name="advantage_fc")

            self.advantage = tf.layers.dense(inputs=self.advantage_fc,
                                             units=self.action_size,
                                             activation=None,
                                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                             name="advantages")

            # Agregating layer
            # Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum A(s,a'))
            self.output = self.value + tf.subtract(self.advantage,
                                                   tf.reduce_mean(self.advantage, axis=1, keepdims=True))

            # Q is our predicted Q value.
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)

            # The loss is modified because of PER
            self.absolute_errors = tf.abs(self.target_Q - self.Q)  # for updating Sumtree

            self.loss = tf.reduce_mean(self.ISWeights_ * tf.squared_difference(self.target_Q, self.Q))

            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)


# Reset the graph
tf.reset_default_graph()

# Instantiate the DQNetwork
DQNetwork = DDDQNNet(state_size, action_size, learning_rate, name="DQNetwork")

# Instantiate the target network
TargetNetwork = DDDQNNet(state_size, action_size, learning_rate, name="TargetNetwork")


class SumTree(object):
    """
    This SumTree code is modified version of Morvan Zhou:
    https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py
    """
    data_pointer = 0

    """
    Here we initialize the tree with all nodes = 0, and initialize the data with all values = 0
    """

    def __init__(self, capacity):
        self.capacity = capacity  # Number of leaf nodes (final nodes) that contains experiences

        # Generate the tree with all nodes values = 0
        # To understand this calculation (2 * capacity - 1) look at the schema above
        # Remember we are in a binary node (each node has max 2 children) so 2x size of leaf (capacity) - 1 (root node)
        # Parent nodes = capacity - 1
        # Leaf nodes = capacity
        self.tree = np.zeros(2 * capacity - 1)

        """ tree:
            0
           / \
          0   0
         / \ / \
        0  0 0  0  [Size: capacity] it's at this line that there is the priorities score (aka pi)
        """

        # Contains the experiences (so the size of data is capacity)
        self.data = np.zeros(capacity, dtype=object)

    """
    Here we add our priority score in the sumtree leaf and add the experience in data
    """

    def add(self, priority, data):
        # Look at what index we want to put the experience
        tree_index = self.data_pointer + self.capacity - 1

        """ tree:
            0
           / \
          0   0
         / \ / \
tree_index  0 0  0  We fill the leaves from left to right
        """

        # Update data frame
        self.data[self.data_pointer] = data

        # Update the leaf
        self.update(tree_index, priority)

        # Add 1 to data_pointer
        self.data_pointer += 1

        if self.data_pointer >= self.capacity:  # If we're above the capacity, you go back to first index (we overwrite)
            self.data_pointer = 0

    """
    Update the leaf priority score and propagate the change through tree
    """

    def update(self, tree_index, priority):
        # Change = new priority score - former priority score
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        # then propagate the change through tree
        while tree_index != 0:  # this method is faster than the recursive loop in the reference code

            """
            Here we want to access the line above
            THE NUMBERS IN THIS TREE ARE THE INDEXES NOT THE PRIORITY VALUES

                0
               / \
              1   2
             / \ / \
            3  4 5  [6] 

            If we are in leaf at index 6, we updated the priority score
            We need then to update index 2 node
            So tree_index = (tree_index - 1) // 2
            tree_index = (6-1)//2
            tree_index = 2 (because // round the result)
            """
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    """
    Here we get the leaf_index, priority value of that leaf and experience associated with that index
    """

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for experiences
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_index = 0

        while True:  # the while loop is faster than the method in the reference code
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break

            else:  # downward search, always search for a higher priority node

                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index

                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0]  # Returns the root node


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    PER_e = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
    PER_a = 0.6  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
    PER_b = 0.4  # importance-sampling, from initial value increasing to 1

    PER_b_increment_per_sampling = 0.001

    absolute_error_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        # Making the tree
        """
        Remember that our tree is composed of a sum tree that contains the priority scores at his leaf
        And also a data array
        We don't use deque because it means that at each timestep our experiences change index by one.
        We prefer to use a simple array and to overwrite when the memory is full.
        """
        self.tree = SumTree(capacity)

    """
    Store a new experience in our tree
    Each new experience have a score of max_prority (it will be then improved when we use this exp to train our DDQN)
    """

    def store(self, experience):
        # Find the max priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])

        # If the max priority = 0 we can't put priority = 0 since this exp will never have a chance to be selected
        # So we use a minimum priority
        if max_priority == 0:
            max_priority = self.absolute_error_upper

        self.tree.add(max_priority, experience)  # set the max p for new p

    """
    - First, to sample a minibatch of k size, the range [0, priority_total] is / into k ranges.
    - Then a value is uniformly sampled from each range
    - We search in the sumtree, the experience where priority score correspond to sample values are retrieved from.
    - Then, we calculate IS weights for each minibatch element
    """

    def sample(self, n):
        # Create a sample array that will contains the minibatch
        memory_b = []

        b_idx, b_ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, 1), dtype=np.float32)

        # Calculate the priority segment
        # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges
        priority_segment = self.tree.total_priority / n  # priority segment

        # Here we increasing the PER_b each time we sample a new minibatch
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])  # max = 1

        # Calculating the max_weight
        p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority

        max_weight = (p_min * n) ** (-self.PER_b)
        # print(p_min, self.tree.total_priority)
        # print(p_min, self.tree.total_priority)
        # print(self.tree.tree[-self.tree.capacity:])
        for i in range(n):
            """
            A value is uniformly sample from each range
            """
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            """
            Experience that correspond to each value is retrieved
            """
            index, priority, data = self.tree.get_leaf(value)

            # P(j)
            sampling_probabilities = priority / self.tree.total_priority

            #  IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
            b_ISWeights[i, 0] = np.power(n * sampling_probabilities, -self.PER_b) / max_weight
            if b_ISWeights[i, 0] == 0:
                print(n, sampling_probabilities, self.PER_b, max_weight)
            b_idx[i] = index

            experience = [data]

            memory_b.append(experience)

        return b_idx, memory_b, b_ISWeights

    """
    Update the priorities on the tree
    """

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.PER_e  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


# Instantiate memory
memory = Memory(memory_size)

# Render the environment
game.new_episode()

""" PRETRAIN """
print("pretraining")
if training:
    for i in range(pretrain_length):
        # If it's the first step
        if i == 0:
            # First we need a state

            state = game.get_state()
            # state, stacked_frames = stack_frames(stacked_frames, state, True)

        # Random action
        action = random.choice(possible_actions)

        # Get the rewards
        reward = game.make_action(action)

        # Look if the episode is finished
        done = game.is_episode_finished()

        # If we're dead
        if done:
            # We finished the episode so the next state is just a blank screen
            next_state = np.zeros(state.shape)
            # print(state.shape)
            # Add experience to memory
            # experience = np.hstack((state, [action, reward], next_state, done))

            experience = state, action, reward, next_state, done
            memory.store(experience)

            # Start a new episode
            game.new_episode()

            # First we need a state
            state = game.get_state()


        else:
            # Get the next state
            next_state = game.get_state()

            # Add experience to memory
            experience = state, action, reward, next_state, done
            memory.store(experience)

            # Our state is now the next_state
            state = next_state


def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions):
    ## EPSILON GREEDY STRATEGY
    # Choose action a from state s using epsilon greedy.
    ## First we randomize a number
    exp_exp_tradeoff = np.random.rand()

    # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

    if (explore_probability > exp_exp_tradeoff):
        # Make a random action (exploration)
        action = random.choice(possible_actions)

    else:
        # Get action from Q-network (exploitation)
        # Estimate the Qs values state
        Qs = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: state.reshape((1, *state.shape))})

        # Take the biggest Q value (= the best action)
        choice = np.argmax(Qs)
        action = possible_actions[int(choice)]

    return action, explore_probability


# This function helps us to copy one set of variables to another
# In our case we use it when we want to copy the parameters of DQN to Target_network
# Thanks of the very good implementation of Arthur Juliani https://github.com/awjuliani
def update_target_graph():
    # Get the parameters of our DQNNetwork
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "DQNetwork")

    # Get the parameters of our Target_network
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "TargetNetwork")

    op_holder = []

    # Update our target_network parameters with DQNNetwork parameters
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


saver = tf.train.Saver()


class MyWindow(pyglet.window.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_minimum_size(400, 300)

        # set background color
        backgroundColor = [10, 0, 0, 255]
        backgroundColor = [i / 255 for i in backgroundColor]
        glClearColor(*backgroundColor)
        # load background image
        self.sess = tf.Session()
        game.new_episode()
        self.state = game.get_state()
        self.nextState = []
        self.loadSession()

    def loadSession(self):
        # if load_traing_model:
        #     directory = "./allModels/model{}/models/model.ckpt".format(load_training_model_number)
        #     saver.restore(self.sess, directory)
        # else:
        saver.restore(self.sess, "./models/model.ckpt")

    def on_draw(self):
        game.render()

    def update(self, dt):
        exp_exp_tradeoff = np.random.rand()

        if load_traing_model:
            explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * load_training_model_number* 100)
        else:
            explore_probability = 0.0001

        if explore_probability > exp_exp_tradeoff:
            # Make a random action (exploration)
            action = random.choice(possible_actions)

        else:
            # Get action from Q-network (exploitation)
            # Estimate the Qs values state
            Qs = self.sess.run(DQNetwork.output,
                               feed_dict={DQNetwork.inputs_: self.state.reshape((1, *self.state.shape))})

            # Take the biggest Q value (= the best action)
            choice = np.argmax(Qs)
            action = possible_actions[int(choice)]

        game.make_action(action)
        # game.render()
        done = game.is_episode_finished()

        if done:
            game.new_episode()
            self.state = game.get_state()
        else:
            self.next_state = game.get_state()
            self.state = self.next_state


# Saver will help us to save our model
print("training")
if training:
    with tf.Session() as sess:
        # Initialize the variables
        # if load:

        if load:
            saver.restore(sess, "./models/model.ckpt")
        else:
            sess.run(tf.global_variables_initializer())

        # Initialize the decay rate (that will use to reduce epsilon)
        decay_step = 0

        # Set tau = 0
        tau = 0

        # Init the game
        game.new_episode()

        # Update the parameters of our TargetNetwork with DQN_weights
        update_target = update_target_graph()
        sess.run(update_target)

        for episode in range(starting_episode, total_episodes):
            # Set step to 0
            step = 0

            # Initialize the rewards of the episode
            episode_rewards = []

            # Make a new episode and observe the first state
            game.new_episode()

            state = game.get_state()

            while step < max_steps:
                step += 1

                # Increase the C step
                tau += 1

                # Increase decay_step
                decay_step += 1

                # With ϵ select a random action atat, otherwise select a = argmaxQ(st,a)
                action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step, state,
                                                             possible_actions)

                # Do the action
                reward = game.make_action(action)

                # Look if the episode is finished
                done = game.is_episode_finished()

                # Add the reward to total reward
                episode_rewards.append(reward)
                if step >= max_steps:
                    print("fuckin nice mate")
                    print('Episode: {}'.format(episode),
                          'Total reward: {}'.format(np.sum(episode_rewards)),
                          'Explore P: {:.4f}'.format(explore_probability))
                # If the game is finished
                if done:
                    # the episode ends so no next state
                    next_state = np.zeros(state.shape, dtype=np.int)  # changed

                    # Set step = max_steps to end the episode
                    step = max_steps

                    # Get the total reward of the episode
                    total_reward = np.sum(episode_rewards)

                    print('Episode: {}'.format(episode),
                          '\tTotal reward: {:.4f}'.format(total_reward),
                          # '\tTraining loss: {:.4f}'.format(loss),
                          '\tExplore P: {:.4f}'.format(explore_probability),
                          '\tScore: {}'.format(game.get_score()),
                          '\tlifespan: {}'.format(game.get_lifespan()),
                          '\tactions per reward gate: {:.4f}'.format(game.get_lifespan() / (max(1, game.get_score()))))

                    # Add experience to memory
                    experience = state, action, reward, next_state, done
                    memory.store(experience)

                else:
                    # Get the next state
                    next_state = game.get_state()

                    # Add experience to memory
                    experience = state, action, reward, next_state, done
                    memory.store(experience)

                    # st+1 is now our current state
                    state = next_state

                ### LEARNING PART
                # Obtain random mini-batch from memory
                tree_idx, batch, ISWeights_mb = memory.sample(batch_size)

                states_mb = np.array([each[0][0] for each in batch], ndmin=2)
                actions_mb = np.array([each[0][1] for each in batch])
                rewards_mb = np.array([each[0][2] for each in batch])
                next_states_mb = np.array([each[0][3] for each in batch], ndmin=2)
                dones_mb = np.array([each[0][4] for each in batch])

                target_Qs_batch = []

                # DOUBLE DQN Logic
                # Use DQNNetwork to select the action to take at next_state (a') (action with the highest Q-value)
                # Use TargetNetwork to calculate the Q_val of Q(s',a')

                # Get Q values for next_state
                q_next_state = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: next_states_mb})

                # Calculate Qtarget for all actions that state
                q_target_next_state = sess.run(TargetNetwork.output, feed_dict={TargetNetwork.inputs_: next_states_mb})

                # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma * Qtarget(s',a')
                for i in range(0, len(batch)):
                    terminal = dones_mb[i]

                    # We got a'
                    action = np.argmax(q_next_state[i])

                    # If we are in a terminal state, only equals reward
                    if terminal:
                        target_Qs_batch.append(rewards_mb[i])

                    else:
                        # Take the Qtarget for action a'
                        target = rewards_mb[i] + gamma * q_target_next_state[i][action]
                        target_Qs_batch.append(target)

                targets_mb = np.array([each for each in target_Qs_batch])

                _, loss, absolute_errors = sess.run([DQNetwork.optimizer, DQNetwork.loss, DQNetwork.absolute_errors],
                                                    feed_dict={DQNetwork.inputs_: states_mb,
                                                               DQNetwork.target_Q: targets_mb,
                                                               DQNetwork.actions_: actions_mb,
                                                               DQNetwork.ISWeights_: ISWeights_mb})
                if loss == 0:
                    print(ISWeights_mb)

                # Update priority
                memory.batch_update(tree_idx, absolute_errors)

                # Write TF Summaries
                # summary = sess.run(write_op, feed_dict={DQNetwork.inputs_: states_mb,
                #                                         DQNetwork.target_Q: targets_mb,
                #                                         DQNetwork.actions_: actions_mb,
                #                                         DQNetwork.ISWeights_: ISWeights_mb})
                # writer.add_summary(summary, episode)
                # writer.flush()

                if tau > max_tau:
                    # Update the parameters of our TargetNetwork with DQN_weights
                    update_target = update_target_graph()
                    sess.run(update_target)
                    tau = 0
                    print("Model updated")

            if (episode < 100 and episode % 5 == 0) or (episode % 1000 == 0):
                directory = "./allModels/model{}".format(episode)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                save_path = saver.save(sess, "./allModels/model{}/models/model.ckpt".format(episode))
                # print("Model Saved")

            # Save model every 5 episodes
            if episode % 5 == 0:
                save_path = saver.save(sess, "./models/model.ckpt")
                print("Model Saved")
else:
    print("setting up window")
    window = MyWindow(displayWidth, displayHeight, "AI Learns to Drive", resizable=False)
    pyglet.clock.schedule_interval(window.update, 1 / frameRate)
    pyglet.app.run()

# print("testing")
# with tf.Session() as sess:
#
#     # Load the model
#     saver.restore(sess, "./models/model.ckpt")
#
#     for i in range(10):
#         print(i)
#         game.new_episode()
#         state = game.get_state()
#
#         while not game.is_episode_finished():
#             ## EPSILON GREEDY STRATEGY
#             # Choose action a from state s using epsilon greedy.
#             ## First we randomize a number
#             exp_exp_tradeoff = np.random.rand()
#
#             explore_probability = 0.01
#
#             if (explore_probability > exp_exp_tradeoff):
#                 # Make a random action (exploration)
#                 action = random.choice(possible_actions)
#
#             else:
#                 # Get action from Q-network (exploitation)
#                 # Estimate the Qs values state
#                 Qs = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: state.reshape((1, *state.shape))})
#
#                 # Take the biggest Q value (= the best action)
#                 choice = np.argmax(Qs)
#                 action = possible_actions[int(choice)]
#
#             game.make_action(action)
#             window.draw(game)
#             # game.render()
#             done = game.is_episode_finished()
#
#             if done:
#                 break
#
#             else:
#                 next_state = game.get_state()
#                 state = next_state
#
#
