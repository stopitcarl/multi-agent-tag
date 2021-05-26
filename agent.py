from abc import ABC, abstractmethod
from utils import *
from tensorflow import keras


class Agent(ABC):
    def __init__(self):
        self.current_decision = LEFT

    @abstractmethod
    def observe(self, observation):
        pass

    @abstractmethod
    def decide(self):
        pass


class SmartAgent(Agent):
    def __init__(self, model_path):
        # Main model
        super().__init__()
        self.model = keras.models.load_model(model_path)

    def observe(self, current_state):
        self.current_state = current_state

    def decide(self):
        action = np.argmax(
            self.model.predict(
                np.array(self.current_state).reshape(-1, *self.current_state.shape))[0])
        return action

    def is_smart(self):
        return True


class SmartController(Agent):
    def __init__(self, model_path):
        # Main model
        super().__init__()
        self.model = keras.models.load_model(model_path)

    def observe(self, current_state):
        self.current_state = current_state

    def decide(self):
        predictions = self.model.predict(np.atleast_2d(self.current_state))
        action = [np.argmax(prediction) for prediction in predictions]
        return action

    def is_smart(self):
        return True