from response_generator import ResponseGenerator_Cond_on_Plh
from random import choice
import time
import copy
import json
import requests

# paths are on dialog server
RESPONSE_GENERATOR_PATH="/home/yuli23/ParlAI/experiments/inspired_response_generator/model/blender_1110_exp7"

class Chatbot():
    def __init__(self):
        self.response_generator = None

    # Parlai framework model, define the gpu when create the agent
    def load_response_generator(self, checkpoint, cond_on_entity = False, gpu_num=0):
        if None == self.response_generator:
            self.response_generator = ResponseGenerator_Cond_on_Plh(checkpoint)
            self.response_generator.load_model(gpu_num)
        
    def chat(self, history_with_entities, user_text, entity_condition):
        response = self.response_generator.process(history_with_entities, user_text, entity_condition)
        return response