from response_generator import Raw_Blender
from random import choice
import time
import copy
import json
import requests

# paths are on dialog server, raw blender model fine tuned on inspired.
RAW_BLENDER_PATH="/home/yuli23/ParlAI/experiments/inspired_blender_baseline/model/blender_1028_exp2"

class RawBlenderBot():
    def __init__(self):
        self.response_generator = None

    # Parlai framework model, define the gpu when create the agent
    def load_response_generator(self, checkpoint, cond_on_entity = False, gpu_num=0):
        if None == self.response_generator:
            self.response_generator = Raw_Blender(checkpoint)
            self.response_generator.load_model(gpu_num)
        
    def chat(self, history_with_entities, user_text):
        end_chat = False
        if user_text == "[quit]" or user_text == "[accept]" or user_text == "quit" or user_text == "accept":
            end_chat = True
            response = "TASK COMPLETE"
        else:
            response = self.response_generator.process(history_with_entities, user_text)
        return response, end_chat

