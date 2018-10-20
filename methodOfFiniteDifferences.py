import gym
import numpy as np
import copy
import statistics

#to normalize the data (used to normalize the state)
def normalize(data):
    mean = statistics.mean(data)
    sigma = statistics.stdev(data)
    return (data - mean) / sigma

#given the environment and weights, run the enviroment, and render if render. 
#return the total reward obtained in the enviroment
def runEnvironment(env,weights,render = False):
    countSteps = 0                          #number of steps taken
    numberOfSteps = env.spec.timestep_limit #length of episode
    done = False                            #environment reached terminal state
    state = env.reset()                     #initial observation
    totReward = 0                           #to keep track of the total reward
    if render:
        env.render()                        
    while countSteps < numberOfSteps and not done:
        countSteps += 1      
        state = normalize(state)
        action = np.dot(weights,state)              #output = dot-products of inputs and weights
        state, reward, done, info = env.step(action)
        if render:
            env.render()
        totReward += reward
    if render:
        env.close()
    return totReward

#main process
if __name__ == '__main__':
    #create the environment
    env = gym.make('BipedalWalker-v2')
    #set parameters   
    learningRate = 0.3      #learning rate with which the weights are updated. (lower = more influence of past.)
    decay = 0               #decay to let the learning rate drop during time (if 0, learning rate stays the same)
    numUpdates = 0          #total number of times the weights have been updated
    maxIterations = 2000    #number of tries to improve weights
    notUpdatedCount = 0     #keeps track of number of iterations the weights did not update without intermission
    goCrazy = 1000000       #goCrazy may be used to update the weights anyway if there was no improvement after goCrazy times
                            #will not be used if set to maxIterations.
    #env specific parameters
    inputSize = env.observation_space.shape[0]  #number of input values (state to be observed)
    outputSize = env.action_space.shape[0]      #number of output values (action)
    #initialize the weights
    weights = np.zeros((outputSize,inputSize))
    prevWeights = copy.deepcopy(weights)   
    oldReward = -1000
    #for maxIterations times, try to improve the weights
    for i in range(maxIterations):
        #random noise
        noise = np.random.rand(outputSize,inputSize)
        #positive and negative weights
        posWeights = weights + noise;
        negWeights = weights - noise;
        #evaluate the positive and negative weights
        posReward = runEnvironment(env,posWeights)
        negReward = runEnvironment(env,negWeights)
        #determine the new weights and evaluate
        newWeights = weights + learningRate*(posReward-negReward)*noise
        newReward = runEnvironment(env,newWeights)
        #if the new reward is better than the current reward, or we did not update for goCrazy times:
        if newReward > oldReward or notUpdatedCount == goCrazy:
            #update parameters / counters
            notUpdatedCount = 0
            numUpdates += 1
            learningRate = learningRate * 1/(1 + decay * numUpdates)
            #to show the new performance in the screen + let know the iterationnumber and new reward. 
            #comment the following three lines if unwanted:
            weights = copy.deepcopy(newWeights)   
            runEnvironment(env,weights,True)
            print(i)
            print(newReward)
            #update history
            oldReward = copy.deepcopy(newReward)
            prevWeights = copy.deepcopy(weights)
        else: #no update, counter + 1
            notUpdatedCount += 1
    #print how many updates we did
    print(numUpdates)    
            