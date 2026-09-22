from req.NNFS_Cont import NNFS_Cont
from req.Replay import Replay
import gymnasium as gym
import time

env = gym.make("InvertedPendulum-v5", render_mode=None)
import numpy as np

policyNetwork = NNFS_Cont([4, 64, 64, 1], alpha=1e-3)
replay = Replay()

episodeBatch = 15 # reducing variance

for iterations in range(300):

    for episode in range(episodeBatch):
        observation, info = env.reset()
        terminated = False
        truncated = False
        rewards = []

        while not (terminated or truncated):
            replay.observations.append(observation)

            action, logProb = policyNetwork.sample_(observation)
            replay.actions.append(action)
            replay.logProbs.append(logProb)
            
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
    mus = policyNetwork.doThisToForward_(evalStates)

    stds = np.zeros_like(mus)
    stds += 0.5

    evalActions = np.array(replay.actions)
    evalActions = np.array(replay.actions).reshape(-1, 1)

    npReturns = np.array(replay.returns)
    npRetMean = npReturns.mean()
    npReturns = ((npReturns - npRetMean) / (npReturns.std()+1e-8))

    batchSize = len(evalActions)

    sqar = ((evalActions - mus)/stds)**2
    logActionThingy = -(0.5)*(sqar + (2 * np.log(stds)) + np.log(2*np.pi))
    loss = -(1/batchSize) * (np.sum(npReturns.reshape(-1, 1) * logActionThingy))

    print(iterations, loss, replay.returns[0])

    dLdmu = - ((evalActions - mus) / (stds**2)) * npReturns.reshape(-1, 1) # mus and stds has the number of actions in it after the batchSize
    dL = dLdmu
    dL /= batchSize

    policyNetwork.backward(dL)
    policyNetwork.optimize()

    replay.clear()

env.close()

env = gym.make("InvertedPendulum-v5", render_mode="human")

for episode in range(10):
    observation, info = env.reset()
    terminated = False
    truncated = False
    episodicReward = 0

    while not (terminated or truncated):
        action = policyNetwork.PredictFromPolicy_(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        episodicReward += reward
    print(episodicReward)

env.close()