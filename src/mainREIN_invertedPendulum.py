from req.NNFS_Cont import NNFS_Cont
from req.Replay import Replay
import gymnasium as gym
import time

env = gym.make("InvertedPendulum-v5", render_mode=None)
import numpy as np

policyNetwork = NNFS_Cont([4, 64, 64, 2], alpha=1e-3)
replay = Replay()

episodeBatch = 64 # reducing variance

for iterations in range(200):

    for episode in range(episodeBatch):
        observation, info = env.reset()
        terminated = False
        truncated = False
        rewards = []

        while not (terminated or truncated):
            replay.observations.append(observation)

            action = policyNetwork.sample(observation)
            replay.actions.append(action)
            
            observation, reward, terminated, truncated, info = env.step(
                action
            )

            reward -= float((np.abs(observation[0]))) * 1.0
            reward -= float((np.abs(observation[1]))) * 1.0
            rewards.append(reward)

        returns = replay.calculateReturnFor(rewards, 0.99)
        for return_ in returns:
            replay.returns.append(return_)

    evalStates = np.array(replay.observations)
    mus, logStds = policyNetwork.forwardPolicy(evalStates)

    stds = np.exp(logStds)

    evalActions = np.array(replay.actions)

    npReturns = np.array(replay.returns)
    npRetMean = npReturns.mean()
    npReturns = ((npReturns - npRetMean) / (npReturns.std()+1e-8))

    batchSize = len(evalActions)

    sqar = ((evalActions - mus)/stds)**2
    logActionThingy = -(0.5)*(sqar + (2 * np.log(stds)) + np.log(2*np.pi))
    loss = -(1/batchSize) * (np.sum(npReturns.reshape(-1, 1) * logActionThingy))

    print(iterations, loss, replay.returns[0])

    dLdmu = - ((evalActions - mus) / (stds**2)) * npReturns.reshape(-1, 1) # mus and stds has the number of actions in it after the batchSize
    dLdlogstd = - ( ((evalActions-mus)/(stds))**2 - 1 ) * npReturns.reshape(-1, 1)
    dL = np.stack((dLdmu, dLdlogstd), axis=2).reshape(batchSize, 2)
    dL /= batchSize

    policyNetwork.backward(dL)
    policyNetwork.optimize()

    replay.clear()

env.close()

env = gym.make("InvertedPendulum-v5", render_mode="human")

for episode in range(17):
    observation, info = env.reset()
    terminated = False
    truncated = False
    episodicReward = 0

    while not (terminated or truncated):
        action = policyNetwork.PredictFromPolicy(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        episodicReward += reward
    print(episodicReward)

env.close()