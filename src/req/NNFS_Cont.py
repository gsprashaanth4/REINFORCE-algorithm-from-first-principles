import numpy as np
import time
from req.NNFS import Dense, ReLU, SoftmaxCCEL, ADAM


def logProb(mus, stds, actions):
    variances = stds ** 2

    return (
        -0.5 * ((actions - mus) ** 2) / variances
        - np.log(stds)
        - 0.5 * np.log(2 * np.pi)
    )

def logProbSquashed(mus, stds, z, actions):
    logGaussian = (
        -0.5 * ((z - mus) ** 2) / (stds ** 2)
        - np.log(stds)
        - 0.5 * np.log(2 * np.pi)
    )

    epsilon = 1e-8

    JaccobCorrection = np.log(
        1 - actions ** 2 + epsilon
    )

    return logGaussian - JaccobCorrection


class NNFS_Cont:
    def __init__(self, layersD, dropoutRate=0.0, alpha=1e-4):
        self.dropoutRate = dropoutRate
        
        self.layers = []
        self.eps = 1e-8

        for i in range(1,len(layersD)-1):
            self.layers.append(Dense(layersD[i-1], layersD[i]))
            self.layers.append(ReLU())

        self.layers.append(Dense(layersD[-2], layersD[-1]))
        self.activationLoss = SoftmaxCCEL()

        self.optimizer = ADAM (
            alpha    = alpha,
            decay    = 0,
            epsilon  = self.eps,
            momentum = 0.9,
            rho      = 0.999
        )

        self.std = 0.5

    # for both mean and stdDev
    
    def doThisToForward(self, observations):
        out = np.atleast_2d(observations)
        
        for layer in self.layers:
            out = layer.forward(out)

        mus = out[:, ::2]
        logStds = out[:, 1::2]

        return mus, logStds
    

    def sample(self, observation):
        
        mus, logStds = self.doThisToForward(observation)
        
        logStds = np.clip(logStds, -5.0, 2.0)
        stds = np.exp(logStds)
        
        actions = np.random.normal(mus, stds)
        return actions[0]


    def forwardPolicy(self, observations):
        mus, logStds = self.doThisToForward(observations)
        
        logStds = np.clip(logStds, -5.0, 2.0)
        return mus, logStds # (batchSize, actions)


    def PredictFromPolicy(self, observation):
        mus, _ = self.doThisToForward(observation)
        return mus[0]


    # for only mean and fixed std..................................................
    # std = self.std
    
    def doThisToForward_(self, observations):
        mus = np.atleast_2d(observations)

        for layer in self.layers:
            mus = layer.forward(mus)

        return mus # (batchSize, actions)


    def sample_(self, observation):
        mus = self.doThisToForward_(observation)
        
        actions = np.random.normal(mus, self.std)

        logProbs = logProb(mus, self.std, actions)
        return actions[0], logProbs[0]
    


    # def sample_(self, observation):
    #     mus = self.doThisToForward_(observation)
    #     stds = np.full_like(mus, self.std)
        
    #     z = np.random.normal(mus, self.std)
    #     actions = np.tanh(z)

    #     logProbs = logProbSquashed(mus, stds, z, actions)
    #     return actions[0], logProbs[0]



    def getValue_(self, observation):
        value = self.doThisToForward_(observation)
        return value[0]



    def getLogProbs_(self, observations, actions):
        actions = np.atleast_2d(actions)

        mus = self.doThisToForward_(observations)
        logProbs = logProb(mus, self.std, actions)

        return np.sum(logProbs, axis=1, keepdims=True)     # if in case multiple actions, then sum of probs



    # def getLogProbs_(self, observations, actions):
    #     actions = np.atleast_2d(actions)
    #     mus = self.doThisToForward_(observations)
    #     stds = np.full_like(mus, self.std)
        
    #     epsilon = 1e-8
    #     actionsSafe = np.clip(actions, -1 + epsilon, 1 - epsilon)

    #     z = np.arctanh(actionsSafe)

    #     logProbs = logProbSquashed(mus, stds, z, actionsSafe)
    #     return mus, np.sum(logProbs, axis=1, keepdims=True)      # if in case multiple actions, then sum of probs



    def PredictFromPolicy_(self, observation):
            mus = self.doThisToForward_(observation)
            # return np.tanh(mus[0])
            return mus[0]
            
    def backward(self, dL):
        prevDVals = dL

        for layer in reversed(self.layers):
            prevDVals = layer.backward(prevDVals)

    def optimize(self):
        self.optimizer.preUpdateParams()

        for layer in self.layers:
            if isinstance(layer, Dense):
                self.optimizer.updateParams(layer)

        self.optimizer.postUpdateParams()

    def save(self, filename):
        data = {
            "iterations": self.optimizer.iterations,
            "alpha": self.optimizer.alpha,
            "decay": self.optimizer.decay,
            "epsilon": self.optimizer.epsilon,
            "momentum": self.optimizer.momentum,
            "rho": self.optimizer.rho,
        }
        denseIndex = 0

        for layer in self.layers:
            if isinstance(layer, Dense):
                data[f"W{denseIndex}"] = layer.weights
                data[f"b{denseIndex}"] = layer.biases

                # Adam state
                data[f"WM{denseIndex}"] = layer.weightMomentums
                data[f"bM{denseIndex}"] = layer.biasesMomentums

                data[f"WC{denseIndex}"] = layer.weightCache
                data[f"bC{denseIndex}"] = layer.biasesCache
                denseIndex += 1

        np.savez(filename, **data)

    def load(self, filename):
        data = np.load(filename)
        self.optimizer.iterations = int(data["iterations"])
        denseIndex = 0

        for layer in self.layers:
            if isinstance(layer, Dense):
                layer.weights = data[f"W{denseIndex}"].copy()
                layer.biases  = data[f"b{denseIndex}"].copy()

                # adam state
                layer.weightMomentums = data[f"WM{denseIndex}"].copy()
                layer.biasesMomentums  = data[f"bM{denseIndex}"].copy()

                layer.weightCache = data[f"WC{denseIndex}"].copy()
                layer.biasesCache  = data[f"bC{denseIndex}"].copy()

                denseIndex += 1