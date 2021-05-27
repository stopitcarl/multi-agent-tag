import numpy as np
import tensorflow as tf
from agent import Agent
from utils import *


# Agent class
class SmartAgent(Agent):
    def __init__(self, model_path):

        # Main model
        self.model = tf.keras.models.load_model(model_path)
		# Model state
		self.current_state = []

    def observe(self, current_state):
        self.current_state = current_state

    def decide(self):
        action = np.argmax(
            self.model.predict(
                np.array(self.current_state).reshape(-1, *self.current_state.shape))[0])
        return action

    def is_smart(self):
        return True
