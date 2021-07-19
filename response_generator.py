from torch import cuda
from parlai.core.agents import create_agent_from_model_file

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

class ResponseGenerator_Cond_on_Plh():
    def __init__(self, model_checkpoint):
        self.model_checkpoint = model_checkpoint
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gpu_num = 0

    def load_model(self, gpu_num):
        if self.model == None:
            opt_overrides = {}
            self.gpu_num = gpu_num
            opt_overrides['gpu'] = gpu_num
            opt_overrides['datatype'] = 'test'
            opt_overrides['inference'] = 'nucleus'
            opt_overrides['skip_generation'] = False
            
            self.model = create_agent_from_model_file(self.model_checkpoint, opt_overrides=opt_overrides)
            print("load Response Generator model from:{}".format(self.model_checkpoint))
            print("allocate Response Generator model to gpu_{}".format(gpu_num))
    
    def _build_up_model_input(self, history, user_text, next_state_placeholders):
        prev_input = ""
        for turn_text in history:
            prev_input = prev_input + " " + turn_text
        if prev_input:
            prev_input =  prev_input + " " + user_text
        if prev_input:
            next_prefix = " [SEP] " + ' '.join(next_state_placeholders) + " [SEP] "
        else:
            next_prefix = ' '.join(next_state_placeholders) + " [SEP] "
        text = prev_input + next_prefix
        # if not history:
        #     text = " [SEP] " + next_state_placeholders
        # else:
        #     text = history + " B: " + user_text + " [SEP] " + next_state_placeholders
        text = text.lower()
        return text
    
    def process(self, history, user_text, next_state_placeholders):
        torch.cuda.set_device(self.gpu_num)
        self.model.reset()
        inputs = self._build_up_model_input(history, user_text, next_state_placeholders)
        print("input to the response generator:{}".format(inputs))
        self.model.observe({'text': inputs, 'episode_done': False})
        output = self.model.act()
        while output and "[movie_p_people" in output['text']:
            self.model.observe({'text': inputs, 'episode_done': False})
            output = self.model.act()
        if output is not None:
            return output['text']
        else:
            return "SYSTEM ERROR!"

class Raw_Blender():
    def __init__(self, model_checkpoint):
        self.model_checkpoint = model_checkpoint
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gpu_num = 0

    def load_model(self, gpu_num):
        if self.model == None:
            opt_overrides = {}
            self.gpu_num = gpu_num
            opt_overrides['gpu'] = gpu_num
            opt_overrides['datatype'] = 'test'
            opt_overrides['inference'] = 'nucleus'
            opt_overrides['skip_generation'] = False
            
            self.model = create_agent_from_model_file(self.model_checkpoint, opt_overrides=opt_overrides)
            print("load Raw Blender model from:{}".format(self.model_checkpoint))
            print("allocate Raw Blender model to gpu_{}".format(gpu_num))
    
    def _build_up_model_input(self, history, user_text):
        prev_input = ""
        for turn_text in history:
            prev_input += turn_text
        if prev_input:
            prev_input =  prev_input + " " + user_text
        else:
            prev_input = user_text
        text = prev_input
        text = text.lower()
        return text
    
    def process(self, history, user_text):
        if not user_text:
            user_text = " [SEP] "
        torch.cuda.set_device(self.gpu_num)
        self.model.reset()
        inputs = self._build_up_model_input(history, user_text)
        print("input to the raw blender:{}".format(inputs))
        self.model.observe({'text': inputs, 'episode_done': False})
        output = self.model.act()
        if output is not None:
            return output['text']
        else:
            return "Raw Blender SYSTEM ERROR!"