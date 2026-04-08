from env import SupportOpsEnv

class Client:
    def __init__(self):
        self.env = SupportOpsEnv()

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def state(self):
        return self.env.state()
