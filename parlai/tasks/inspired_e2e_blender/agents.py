#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import FixedDialogTeacher
from parlai.utils.io import PathManager
# from .build import build
import os
import json
import pandas as pd
import ast

def _path(opt):
    # build the data if it does not exist
    # build(opt)

    # set up path to data (specific to each dataset)
    inspired_response_generator_folder_path = os.path.join(opt['datapath'], 'inspired', 'e2e_blender')
    # training_file_path = os.path.join(inspired_response_generator_folder_path, 'inspired_response_generator_training_file_with_placeholders.csv')
    # inspired_response_generator_folder_path = os.path.join(opt['datapath'], 'redial')
    training_file_path = os.path.join(inspired_response_generator_folder_path, 'e2e_blender_training_file_1117.csv')
    return training_file_path, inspired_response_generator_folder_path


class InspiredE2eBlenderTeacher(FixedDialogTeacher):
    """
    Inspired Response Generator Teacher.
    GPT2 based
    conditional generation
    Diff = B(t+1) - B(t)
    Rt = Generator([Diff])
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        opt['datafile'], inspired_response_generator_folder_path = _path(opt)
        self._setup_data(opt['datafile'], inspired_response_generator_folder_path)
        self.id = 'inspired_e2e_blender'
        self.reset()

    @classmethod
    def add_cmdline_args(cls, argparser):
        agent = argparser.add_argument_group('Inspired e2e blender Teacher Args')
        agent.add_argument(
            '-instokens',
            '--add_inspired_special_tokens',
            type='bool',
            default=False,
            help="True if add special place holders tokens in tokenizer.",
        )

    def _setup_data(self, data_path, folder_path):
        print('loading: ' + data_path)
        df_training = pd.read_csv(data_path)
        df_training = df_training.fillna("")
        self.messages = self._get_messages(df_training)
        # if self.datatype.startswith('test'):
        #     self.messages = self.messages[9507:]
        # elif self.datatype.startswith('valid'):
        #     self.messages = self.messages[8451:9507]
        # else:
        #     self.messages = self.messages[:8451]
        data_length = len(self.messages)

        if self.datatype.startswith('test'):
            self.messages = self.messages[int(0.9*data_length):]
        elif self.datatype.startswith('valid'):
            self.messages = self.messages[int(0.8*data_length):int(0.9*data_length)]
        else:
            self.messages = self.messages[:int(0.8*data_length)]
        
        
    
    def _get_messages(self, df_training):
        print("[Inspired_Dataset]processing dataset...")
        messages = []
        for index, row in df_training.iterrows():
            context = row["context"].lower()
            bt = row["bt"].lower()
            # bt = ast.literal_eval(bt)
            entities = row["entities"].lower()
            response = row["response"].lower()
            row_dic = {"context": context, "bt": bt, "entities": entities, "response": response}
            messages.append(row_dic)
        return messages

    def num_examples(self):
        return len(self.messages)

    def num_episodes(self):
        return len(self.messages)

    def get(self, episode_idx, entry_idx=0):
        text = self.messages[episode_idx]["context"] + self.messages[episode_idx]["bt"] + " [sep] " + self.messages[episode_idx]["entities"] + " [sep] "
        action = {
            'id': self.id,
            'text': text,
            'episode_done': True,
            'labels': [self.messages[episode_idx]['response']],
        }
        return action


class DefaultTeacher(InspiredE2eBlenderTeacher):
    pass
