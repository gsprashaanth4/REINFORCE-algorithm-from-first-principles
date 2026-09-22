from req.REINNN import REINNN
from req.Replay import Replay
import gymnasium as gym
import time


print("hello")
env = gym.make("CartPole-v1", render_mode=None)
import numpy as np

policyNetwork = REINNN([4, 64, 64, 2])
replay = Replay()

episodeBatch = 10 # reducing variance

for iterations in range(300):

    for episode in range(episodeBatch):
        observation, info = env.reset()
        terminated = False
        truncated = False
        rewards = []

        while not (terminated or truncated):
            replay.observations.append(observation)
            action = policyNetwork.sampleFromPolicy(observation)
            replay.actions.append(action)
            observation, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)

        returns = replay.calculateReturnFor(rewards, 0.99)
        for return_ in returns:
            replay.returns.append(return_)

    evalStates = np.array(replay.observations)

    probs, preds = policyNetwork.predict(evalStates)

    evalActions = np.array(replay.actions) # 0 or 1 action chosen
    evalChoosen = probs[np.arange(len(evalActions)), evalActions] # prob of the choosen action
    
    npReturns = np.array(replay.returns)
    npRetMean = npReturns.mean()

    npReturns = ((npReturns - npRetMean) / (npReturns.std()+1e-8)) # among local episodes (batch kinda)

    loss = -np.mean(np.log(evalChoosen) * npReturns) # trickling down

    print(iterations, loss, replay.returns[0])

    batchSize = len(evalActions)
    
    y = np.eye(2)[evalActions]
    dL = probs - y
    dL *= npReturns[:, None]
    dL /= batchSize

    policyNetwork.backward(dL)
    policyNetwork.optimize()

    replay.clear()

env.close()

env = gym.make("CartPole-v1", render_mode="human")

for episode in range(10):
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