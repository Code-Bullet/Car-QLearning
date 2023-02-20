from pyglet.window import key
from pyglet.gl import *
import pyglet
from Global import *
import pygame
from Games import Game
import random
import os
import numpy as np
from collections import deque
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

vec2 = pygame.math.Vector2


class QLearning:
    def __init__(self, game):

        self.game = game
        self.game.new_episode()

        self.stateSize = [game.state_size]
        self.actionSize = game.no_of_actions
        self.learningRate = 0.00030 #default 0.00025
        self.possibleActions = np.identity(self.actionSize, dtype=int)

        self.totalTrainingEpisodes = 100000
        self.maxSteps = 3600

        self.batchSize = 64
        self.memorySize = 100000

        self.maxEpsilon = 1
        self.minEpsilon = 0.01
        self.decayRate = 0.00001
        self.decayStep = 0
        self.gamma = 0.9
        self.training = True

        self.pretrainLength = self.batchSize

        self.maxTau = 10000
        self.tau = 0
        # reset the graph i guess, I don't know why there is already a graph happening but who cares
        tf.compat.v1.reset_default_graph()

        self.sess = tf.compat.v1.Session()

        self.DQNetwork = DQN(self.stateSize, self.actionSize, self.learningRate, name='DQNetwork')
        self.TargetNetwork = DQN(self.stateSize, self.actionSize, self.learningRate, name='TargetNetwork')

        self.memoryBuffer = PrioritisedMemory(self.memorySize)
        self.pretrain()

        self.state = []
        self.trainingStepNo = 0

        self.newEpisode = False
        self.stepNo = 0
        self.episodeNo = 0
        self.saver = tf.compat.v1.train.Saver()

        load = False
        loadFromEpisodeNo = 15800
        if load:
            self.episodeNo = loadFromEpisodeNo
            self.saver.restore(self.sess, "./allModels/modelMatin{}/models/model.ckpt".format(self.episodeNo))
        else:
            self.sess.run(tf.compat.v1.global_variables_initializer())
        # self.sess.graph.finalize()
        self.sess.run(self.update_target_graph())

    # This function helps us to copy one set of variables to another
    # In our case we use it when we want to copy the parameters of DQN to Target_network
    # Thanks of the very good implementation of Arthur Juliani https://github.com/awjuliani
    def update_target_graph(self):

        # Get the parameters of our DQNNetwork
        from_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "DQNetwork")

        # Get the parameters of our Target_network
        to_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "TargetNetwork")

        op_holder = []

        # Update our target_network parameters with DQNNetwork parameters
        for from_var, to_var in zip(from_vars, to_vars):
            op_holder.append(to_var.assign(from_var))
        return op_holder

    def pretrain(self):
        for i in range(self.pretrainLength):
            if i == 0:
                state = self.game.get_state()

            # pick a random movement and do it to populate the memory thing
            # choice = random.randInt(self.actionSize)
            # action = self.possibleActions[choice]
            action = random.choice(self.possibleActions)
            #print(action)
            actionNo = np.argmax(action)
            #print(actionNo)
            # now we need to get next state
            reward = self.game.make_action(actionNo)
            nextState = self.game.get_state()
            self.newEpisode = False

            if self.game.is_episode_finished(): #if car is dead
                reward = -100
                self.memoryBuffer.store((state, action, reward, nextState, True))
                self.game.new_episode()
                state = self.game.get_state()
                self.newEpisode = True
            else:
                self.memoryBuffer.store((state, action, reward, nextState, False))
                self.game.render()
                state = nextState

        print("pretrainingDone")

    def train(self):

        if self.trainingStepNo == 0:
            self.state = self.game.get_state()

        if self.newEpisode:
            self.state = self.game.get_state()

        if self.stepNo < self.maxSteps:
            self.stepNo += 1
            self.decayStep += 1
            self.trainingStepNo += 1
            self.tau += 1

            # choose best action if not exploring choose random otherwise

            epsilon = self.minEpsilon + (self.maxEpsilon - self.minEpsilon) * np.exp(
                -self.decayRate * self.decayStep)

            if np.random.rand() < epsilon:
                choice = random.randint(1, len(self.possibleActions)) - 1
                action = self.possibleActions[choice]

            else:
                QValues = self.sess.run(self.DQNetwork.output,
                                        feed_dict={self.DQNetwork.inputs_: np.array([self.state])})
                choice = np.argmax(QValues)
                action = self.possibleActions[choice]

            actionNo = np.argmax(action)
            # now we need to get next state
            reward = self.game.make_action(actionNo)

            nextState = self.game.get_state()
            #window.clear()
            #self.game.render()
            if (reward > 0):
                #print("Hell YEAH, Reward {}".format(reward))
                pass
            # if car is dead then finish episode
            if self.game.is_episode_finished():
                reward = -100
                self.stepNo = self.maxSteps
                #print("DEAD!! Reward =  -100")

            # print("Episode {} Step {} Action {} reward {} epsilon {} experiences stored {}"
            #       .format(self.episodeNo, self.stepNo, actionNo, reward, epsilon, self.trainingStepNo))

            # add the experience to the memory buffer
            self.memoryBuffer.store((self.state, action, reward, nextState, self.game.is_episode_finished()))

            self.state = nextState

            # learning part
            # first we are gonna need to grab a random batch of experiences from out memory
            treeIndexes, batch, ISWeights = self.memoryBuffer.sample(self.batchSize)

            statesFromBatch = np.array([exp[0][0] for exp in batch])
            actionsFromBatch = np.array([exp[0][1] for exp in batch])
            rewardsFromBatch = np.array([exp[0][2] for exp in batch])
            nextStatesFromBatch = np.array([exp[0][3] for exp in batch])
            carDieBooleansFromBatch = np.array([exp[0][4] for exp in batch])

            targetQsFromBatch = []

            # predict the q values of the next state for each experience in the batch
            QValueOfNextStates = self.sess.run(self.TargetNetwork.output,
                                               feed_dict={self.TargetNetwork.inputs_: nextStatesFromBatch})

            for i in range(self.batchSize):
                action = np.argmax(QValueOfNextStates[i])  # double DQN
                terminalState = carDieBooleansFromBatch[i]
                if terminalState:
                    targetQsFromBatch.append(rewardsFromBatch[i])
                else:
                    # target = rewardsFromBatch[i] + self.gamma * np.max(QValueOfNextStates[i])
                    target = rewardsFromBatch[i] + self.gamma * QValueOfNextStates[i][action]  # double DQN
                    targetQsFromBatch.append(target)

            targetsForBatch = np.array([t for t in targetQsFromBatch])

            loss, _, absoluteErrors = self.sess.run(
                [self.DQNetwork.loss, self.DQNetwork.optimizer, self.DQNetwork.absoluteError],
                feed_dict={self.DQNetwork.inputs_: statesFromBatch,
                           self.DQNetwork.actions_: actionsFromBatch,
                           self.DQNetwork.targetQ: targetsForBatch,
                           self.DQNetwork.ISWeights_: ISWeights})

            # update priorities
            self.memoryBuffer.batchUpdate(treeIndexes, absoluteErrors)

        if self.stepNo >= self.maxSteps:
            self.episodeNo += 1
            self.stepNo = 0
            self.newEpisode = True
            self.game.new_episode()
            if self.episodeNo >= self.totalTrainingEpisodes:
                self.training = False
            if self.episodeNo % 100 == 0:
                directory = "./allModels/model{}".format(self.episodeNo)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                save_path = self.saver.save(self.sess,
                                            "./allModels/model{}/models/model.ckpt".format(self.episodeNo))
                print("Model Saved")
        if self.tau > self.maxTau:
            self.sess.run(self.update_target_graph())
            self.tau = 0
            print("Target Network Updated")

    def test(self):

        self.state = self.game.get_state()

        QValues = self.sess.run(self.DQNetwork.output,
                                feed_dict={self.DQNetwork.inputs_: np.array([self.state])})
        choice = np.argmax(QValues)
        action = self.possibleActions[choice]

        actionNo = np.argmax(action)
        # now we need to get next state
        self.game.make_action(actionNo)

        if self.game.is_episode_finished():
            self.game.new_episode()


class Memory:
    def __init__(self, maxSize):
        self.buffer = deque(maxlen=maxSize)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batchSize):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                 size=batchSize,
                                 replace=False)
        return [self.buffer[i] for i in index]


class DQN:
    def __init__(self, stateSize, actionSize, learningRate, name):
        self.stateSize = stateSize
        self.actionSize = actionSize
        self.learningRate = learningRate
        self.name = name

        with tf.compat.v1.variable_scope(self.name):
            # the inputs describing the state
            self.inputs_ = tf.compat.v1.placeholder(tf.float32, [None, *self.stateSize], name="inputs")

            # the one hotted action that we took
            # e.g. if we took the 3rd action action_ = [0,0,1,0,0,0,0]
            self.actions_ = tf.compat.v1.placeholder(tf.float32, [None, self.actionSize], name="actions")

            # the target = reward + the discounted maximum possible q value of hte next state
            self.targetQ = tf.compat.v1.placeholder(tf.float32, [None], name="target")

            self.ISWeights_ = tf.compat.v1.placeholder(tf.float32, [None, 1], name='ISWeights')

            self.dense1 = tf.compat.v1.layers.dense(inputs=self.inputs_,
                                          units=16,
                                          activation=tf.nn.elu,
                                          kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                                          name="dense1")
            self.dense2 = tf.compat.v1.layers.dense(inputs=self.dense1,
                                          units=16,
                                          activation=tf.nn.elu,
                                          kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                                          name="dense2")
            self.output = tf.compat.v1.layers.dense(inputs=self.dense2,
                                          units=self.actionSize,
                                          kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                                          activation=None,
                                          name="outputs")

            # by multiplying the output by the one hotted action space we only get the q value we desire
            # all other values are 0, therefore taking the sum of these values gives us our qValue
            self.QValue = tf.reduce_sum(tf.multiply(self.output, self.actions_))

            self.absoluteError = abs(self.QValue - self.targetQ)  # used for prioritising experiences

            # calculate the loss by using mean squared error
            self.loss = tf.reduce_mean(self.ISWeights_ * tf.square(self.targetQ - self.QValue))

            # use adam optimiser (its good shit)
            self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learningRate).minimize(self.loss)


class DDQN:
    def __init__(self, stateSize, actionSize, learningRate, name):
        self.stateSize = stateSize
        self.actionSize = actionSize
        self.learningRate = learningRate
        self.name = name

        with tf.compat.v1.variable_scope(self.name):
            # the inputs describing the state
            self.inputs_ = tf.compat.v1.placeholder(tf.float32, [None, *self.stateSize], name="inputs")

            # the one hotted action that we took
            # e.g. if we took the 3rd action action_ = [0,0,1,0,0,0,0]
            self.actions_ = tf.compat.v1.placeholder(tf.float32, [None, self.actionSize], name="actions")

            # the target = reward + the discounted maximum possible q value of hte next state
            self.targetQ = tf.compat.v1.placeholder(tf.float32, [None], name="target")

            self.ISWeights_ = tf.compat.v1.placeholder(tf.float32, [None, 1], name='ISWeights')

            self.dense1 = tf.compat.v1.layers.dense(inputs=self.inputs_,
                                          units=16,
                                          activation=tf.nn.elu,
                                          kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                                          name="dense1")

            ## Here we separate into two streams
            # The one that calculate V(s) which is the value of the input state
            # in other words how good this state is

            self.valueLayer = tf.compat.v1.layers.dense(inputs=self.dense1,
                                              units=16,
                                              activation=tf.nn.elu,
                                              kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                                              name="valueLayer")

            self.value = tf.compat.v1.layers.dense(inputs=self.valueLayer,
                                         units=1,
                                         activation=None,
                                         kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                                         name="value")

            # The one that calculate A(s,a)
            # which is the advantage of taking each action in this given state
            self.advantageLayer = tf.compat.v1.layers.dense(inputs=self.dense1,
                                                  units=16,
                                                  activation=tf.nn.elu,
                                                  kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                                                  name="advantageLayer")

            self.advantage = tf.compat.v1.layers.dense(inputs=self.advantageLayer,
                                             units=self.actionSize,
                                             activation=None,
                                             kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                                             name="advantages")

            # Aggregating layer
            # Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum A(s,a'))
            # output  = value of the state + the advantage of taking the given action relative to other actions
            self.output = self.value + tf.subtract(self.advantage,
                                                   tf.reduce_mean(self.advantage, axis=1, keepdims=True))

            # by multiplying the output by the one hotted action space we only get the q value we desire
            # all other values are 0, therefore taking the sum of these values gives us our qValue
            self.QValue = tf.reduce_sum(tf.multiply(self.output, self.actions_))

            self.absoluteError = abs(self.QValue - self.targetQ)  # used for prioritising experiences

            # calculate the loss by using mean squared error
            self.loss = tf.reduce_mean(self.ISWeights_ * tf.square(self.targetQ - self.QValue))

            # use adam optimiser (its good shit)
            self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learningRate).minimize(self.loss)


class PrioritisedMemory:
    # some cheeky hyperparameters
    e = 0.01
    a = 0.06
    b = 0.04
    bIncreaseRate = 0.001
    errorsClippedAt = 1.0

    def __init__(self, capacity):
        self.sumTree = SumTree(capacity)
        self.capacity = capacity

    def store(self, experience):
        """ when an experience is first added to memory it has the highest priority
            so each experience is run through at least once
        """
        # get max priority
        maxPriority = np.max(self.sumTree.tree[self.sumTree.indexOfFirstData:])

        # if the max is 0 then this means that this is the first element
        # so might as well give it the highest priority possible
        if maxPriority == 0:
            maxPriority = self.errorsClippedAt

        self.sumTree.add(maxPriority, experience)

    def sample(self, n):
        batch = []
        batchIndexes = np.zeros([n], dtype=np.int32)
        batchISWeights = np.zeros([n, 1], dtype=np.float32)

        # so we divide the priority space up into n different priority segments
        totalPriority = self.sumTree.total_priority()
        prioritySegmentSize = totalPriority / n

        # also we need to increase b with every value to anneal it towards 1
        self.b += self.bIncreaseRate
        self.b = min(self.b, 1)

        # ok very nice now in order to normalize all the weights in order to ensure they are all within 0 and 1
        # we are going to need to get the maximum weight and divide all weights by that

        # the largest weight will have the lowest priority and thus the lowest probability of being chosen
        minPriority = np.min(np.maximum(self.sumTree.tree[self.sumTree.indexOfFirstData:], self.e))
        minProbability = minPriority / self.sumTree.total_priority()

        # formula
        maxWeight = (minProbability * n) ** (-self.b)
        for i in range(n):
            # get the upper and lower bounds of the segment
            segmentMin = prioritySegmentSize * i
            segmentMax = segmentMin + prioritySegmentSize

            value = np.random.uniform(segmentMin, segmentMax)

            treeIndex, priority, data = self.sumTree.getLeaf(value)

            samplingProbability = priority / totalPriority

            #  IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi

            batchISWeights[i, 0] = np.power(n * samplingProbability, -self.b) / maxWeight

            batchIndexes[i] = treeIndex
            experience = [data]
            batch.append(experience)

        return batchIndexes, batch, batchISWeights

    def batchUpdate(self, treeIndexes, absoluteErrors):
        absoluteErrors += self.e  # do this to avoid 0 values
        clippedErrors = np.minimum(absoluteErrors, self.errorsClippedAt)

        priorities = np.power(clippedErrors, self.a)
        for treeIndex, priority in zip(treeIndexes, priorities):
            self.sumTree.update(treeIndex, priority)


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.size = 2 * capacity - 1
        self.tree = np.zeros(self.size)
        self.data = np.zeros(capacity, dtype=object)
        self.dataPointer = 0
        self.indexOfFirstData = capacity - 1

    """
    adds a new element to the sub tree (or overwrites an old one) and updates all effected nodes 
    """

    def add(self, priority, data):
        treeIndex = self.indexOfFirstData + self.dataPointer

        # overwrite data

        self.data[self.dataPointer] = data
        self.update(treeIndex, priority)
        self.dataPointer += 1
        self.dataPointer = self.dataPointer % self.capacity

    """
    updates the priority of the indexed leaf as well as updating the value of all effected
    elements in the sum tree
    """

    def update(self, index, priority):
        change = priority - self.tree[index]
        self.tree[index] = priority

        while index != 0:
            # set index to parent
            index = (index - 1) // 2
            self.tree[index] += change

    def getLeaf(self, value):
        parent = 0
        LChild = 1
        RChild = 2

        while LChild < self.size:
            if self.tree[LChild] >= value:
                parent = LChild
            else:
                value -= self.tree[LChild]
                parent = RChild

            LChild = 2 * parent + 1
            RChild = 2 * parent + 2

        treeIndex = parent
        dataIndex = parent - self.indexOfFirstData

        return treeIndex, self.tree[treeIndex], self.data[dataIndex]

    def total_priority(self):
        return self.tree[0]  # Returns the root node


"""
a class inheriting from the pyglet window class which controls the game window and acts as the main class of the program
"""

class MyWindow(pyglet.window.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_minimum_size(400, 300)

        # set background color
        backgroundColor = [0,0,0,1]
        glClearColor(*backgroundColor)

        self.game = Game()
        self.ai = QLearning(self.game)

        self.firstClick = True

    def on_key_press(self, symbol, modifiers):
        pass

    def on_close(self):
        self.ai.sess.close()
        pass

    def on_key_release(self, symbol, modifiers):
        if symbol == key.SPACE:
            self.ai.training = not self.ai.training

    def on_mouse_press(self, x, y, button, modifiers):
        pass

    def on_draw(self):
        window.set_size(width=displayWidth, height=displayHeight)
        self.clear()
        self.game.render()


    def update(self, dt):
        for i in range(5):
            if self.ai.training:
                self.ai.train()
            else:
                self.ai.test()
                return
        pass



if __name__ == "__main__":
    window = MyWindow(displayWidth, displayHeight, "AI Learns to Drive", resizable=False)
    pyglet.clock.schedule_interval(window.update, 1 / frameRate)
    pyglet.app.run()
