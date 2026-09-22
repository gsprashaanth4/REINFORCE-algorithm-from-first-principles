import numpy as np
from req.NNFS import Dense, Dropout, ReLU, Softmax, SoftmaxCCEL, CCEL, ADAM

class REINNN:
    def __init__(self, layersD, dropoutRate=0.0):
        self.dropoutRate = dropoutRate
        
        self.layers = []

        for i in range(1,len(layersD)-1):
            self.layers.append(Dense(layersD[i-1], layersD[i]))
            self.layers.append(ReLU())

        self.layers.append(Dense(layersD[-2], layersD[-1]))
        self.activationLoss = SoftmaxCCEL()
        self.optimizer = ADAM()

    def predict(self, inputs):
        inputs = np.atleast_2d(inputs)
        out = inputs

        for layer in self.layers:
            out = layer.forward(out)
        
        probs = self.activationLoss.af.forward(out)
        preds = np.argmax(probs, axis=1)
        return probs, preds
    
    def sampleFromPolicy(self, observation):
        probs, _ = self.predict(observation)
        action = np.random.choice(len(probs[0]), p=probs[0])
        return action
    
    def PredictFromPolicy(self, observation):
        probs, _ = self.predict(observation)
        action = np.argmax(probs[0])
        return action
    
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