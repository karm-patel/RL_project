class Config:
    def __init__(self):
        self.FEATURE_DIM = 784
        self.AGENTS = 1000
        self.CLASSES = 10
        self.TERMINAL_ACTIONS = self.CLASSES
        self.ACTION_DIM = self.CLASSES + self.FEATURE_DIM
        self.REWARD_CORRECT   =  0
        self.REWARD_INCORRECT = -1
        self.LAMBDA = 0.001

config = Config()


