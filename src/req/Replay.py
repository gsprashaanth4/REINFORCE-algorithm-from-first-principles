class Replay:
    def __init__(self):
        self.observations = []
        self.actions      = []
        self.rewards      = []
        self.returns      = []
        self.nObs         = []
        self.logProbs     = []
        self.values       = []
        self.nValues      = []
        self.terminated   = []
        self.truncated    = []

    def clear(self):
        self.observations = []
        self.actions      = []
        self.rewards      = []
        self.returns      = []
        self.nObs         = []
        self.logProbs     = []
        self.values       = []
        self.nValues      = []
        self.terminated   = []
        self.truncated    = []

    def calculateReturn(self, discount):
        for reward in reversed(self.rewards):
            if len(self.returns) == 0:
                self.returns.append(reward)
            else:
                self.returns.append(reward + discount* self.returns[-1])

        self.returns.reverse()

    def calculateReturnFor(self, rewards, discount):
        returns = []
        for reward in reversed(rewards):
            if len(returns) == 0:
                returns.append(reward)
            else:
                returns.append(reward + discount* returns[-1])
        returns.reverse()
        return returns