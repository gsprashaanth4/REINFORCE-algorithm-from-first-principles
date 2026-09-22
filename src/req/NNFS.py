import numpy as np
import time

class Dense:
    def __init__(self, inD, nD,
                 weightLamda1=0,
                 biasLamda1=0,
                 weightLamda2=0,
                 biasLamda2=0
                ):
        
        stdDev = np.sqrt(2/inD)

        self.weights = np.random.normal(0, stdDev, (inD, nD))
        self.biases = np.zeros((1, nD))

        self.weightMomentums = np.zeros_like(self.weights)
        self.biasesMomentums = np.zeros_like(self.biases)
        self.weightCache     = np.zeros_like(self.weights)
        self.biasesCache     = np.zeros_like(self.biases)

        self.weightLamda1 = weightLamda1
        self.biasesLamda1 = biasLamda1
        self.weightLamda2 = weightLamda2
        self.biasesLamda2 = biasLamda2

    def regularizationLoss(self):
        regularizationLoss = 0

        if self.weightLamda1 > 0:
            regularizationLoss += np.sum(np.abs(self.weights)) * self.weightLamda1
        if self.biasesLamda1 > 0:
            regularizationLoss += np.sum(np.abs(self.biases))  * self.biasesLamda1

        if self.weightLamda2 > 0:
            regularizationLoss += np.sum(self.weights ** 2)    * self.weightLamda2
        if self.biasesLamda2 > 0:
            regularizationLoss += np.sum(self.biases ** 2)     * self.biasesLamda2
        
        return regularizationLoss
    
    def dRegularizationLoss(self):

        if self.weightLamda1 > 0:
            self.dWeights += self.weightLamda1 * (np.where(self.weights >= 0, 1, -1))
        if self.biasesLamda1 > 0:
            self.dBiases += self.biasesLamda1 * (np.where( self.biases >= 0, 1, -1))

        if self.weightLamda2 > 0:
            self.dWeights += 2 * self.weightLamda2 * self.weights
        if self.biasesLamda2 > 0:
            self.dBiases += 2 * self.biasesLamda2 * self.biases

    def forward(self, inputs):
        self.inputs = inputs
        self.pushed = np.dot(self.inputs, self.weights) + self.biases
        return self.pushed
    
    def backward(self, dValues):
        self.dWeights = np.dot(self.inputs.T, dValues)
        self.dBiases  = np.sum(dValues, axis=0, keepdims=True)
        self.dInputs  = np.dot(dValues, self.weights.T)
        self.dRegularizationLoss()
        return self.dInputs


class Dropout:
    def __init__(self, dropoutRate):
        self.dropoutRate = dropoutRate
    
    def forward(self, inputs):
        self.dropout = np.random.binomial(1, 1-self.dropoutRate, size=inputs.shape) / (1-self.dropoutRate)
        return inputs * self.dropout
    
    def backward(self, dvalues):
        if self.dropoutRate != 0:
            self.dInputs = dvalues * self.dropout
            return self.dInputs
        return dvalues


class TanH:
    def forward(self, inputs):
        return np.tanh(inputs)

    def backward(self, dValues):
        return 1.0/np.cosh(dValues)

class ReLU:
    def forward(self, z):
        self.z = z
        self.a = np.maximum(0, z)
        return self.a
    
    def backward(self, dValues):
        self.dInputs = dValues.copy()
        self.dInputs[self.z <= 0] = 0
        return self.dInputs


class Softmax:
    def forward(self, z):
        self.z = z
        exps = np.exp(z - np.max(z, axis=1, keepdims=True))
        self.a = exps/np.sum(exps, axis=1, keepdims=True)
        return self.a


class CCEL:
    def forward(self, yCap, y):
        self.yCap = np.clip(yCap, 1e-8, 1 - 1e-8)
        loss = -np.mean(np.sum(y * np.log(self.yCap), axis=1))
        return loss

# SoftmaxCCEL handles the backward pass of Softmax and CCEL
class SoftmaxCCEL:
    def __init__(self):
        self.af = Softmax()
        self.lf = CCEL()

    def forward(self, z, y):
        self.z = z
        self.y = y
        self.softed = self.af.forward(self.z)
        loss = self.lf.forward(self.softed, y)
        return loss
    
    def backward(self):
        self.dLdZ = self.softed - self.y
        batch_size = self.y.shape[0]
        self.dLdZ /= batch_size
        return self.dLdZ


class ADAM:
    def __init__(
            self,
            alpha    = 7e-4,
            decay    = 0,
            epsilon  = 1e-8,
            momentum = 0.9,
            rho      = 0.999
        ):
        
        self.alpha    = alpha
        self.decay    = decay
        self.epsilon  = epsilon
        self.momentum = momentum
        self.rho      = rho

        self.currentAlpha = alpha
        self.iterations = 0

    def preUpdateParams(self):
        if self.decay:
            self.currentAlpha = self.alpha / (1. + self.decay * self.iterations)
   
    def updateParams(self, layer):

        if isinstance(layer, Dense):

            layer.weightMomentums = self.momentum * layer.weightMomentums + \
                                    (1-self.momentum) * layer.dWeights
            layer.biasesMomentums = self.momentum * layer.biasesMomentums + \
                                    (1-self.momentum) * layer.dBiases

            weightMomentumsCorrected = layer.weightMomentums / (1 - self.momentum ** (self.iterations + 1))
            biasesMomentumsCorrected = layer.biasesMomentums / (1 - self.momentum ** (self.iterations + 1))

            layer.weightCache = self.rho*layer.weightCache + (1-self.rho)*layer.dWeights**2
            layer.biasesCache = self.rho*layer.biasesCache + (1-self.rho)*layer.dBiases**2

            weightCacheCorrected = layer.weightCache / (1 - self.rho ** (self.iterations + 1))
            biasesCacheCorrected = layer.biasesCache / (1 - self.rho ** (self.iterations + 1))

            layer.weights -= self.currentAlpha * weightMomentumsCorrected / (np.sqrt(weightCacheCorrected) + self.epsilon)
            layer.biases  -= self.currentAlpha * biasesMomentumsCorrected / (np.sqrt(biasesCacheCorrected) + self.epsilon)
        
    def postUpdateParams(self):
        self.iterations+=1