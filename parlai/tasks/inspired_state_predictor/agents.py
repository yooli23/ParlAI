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
    inspired_state_predictor_folder_path = os.path.join(opt['datapath'], 'inspired', 'state_predictor')
    training_file_path = os.path.join(inspired_state_predictor_folder_path, 'gpt2_state_predictor_training_file.csv')
    return training_file_path, inspired_state_predictor_folder_path


class InspiredStatePredictorTeacher(FixedDialogTeacher):
    """
    Inspired State Predictor Teacher.
    GPT2 based
    U(t)B(t) -> B(t+1)
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        opt['datafile'], inspired_state_predictor_folder_path = _path(opt)
        self._setup_data(opt['datafile'], inspired_state_predictor_folder_path)
        self.id = 'inspired_state_predictor'
        self.reset()

    @classmethod
    def add_cmdline_args(cls, argparser):
        agent = argparser.add_argument_group('Inspired State Predictor Teacher Args')
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
        if self.datatype.startswith('test'):
            self.messages = self.messages[2410:]
        elif self.datatype.startswith('valid'):
            self.messages = self.messages[2149:2410]
        else:
            self.messages = self.messages[:2149]
        
        
    
    def _get_messages(self, df_training):
        print("[Inspired_Dataset]processing dataset...")
        messages = []
        for index, row in df_training.iterrows():
            context = row["context"].lower()
            ut = row["ut"].lower()
            bt = row["bt"].lower()
            bt_plus_1 = row["bt_plus_1"].lower()
            row_dic = {"context": context, "ut": ut, "bt": bt, "bt_plus_1": bt_plus_1}
            messages.append(row_dic)
        return messages

    def num_examples(self):
        return len(self.messages)

    def num_episodes(self):
        return len(self.messages)

    def get(self, episode_idx, entry_idx=0):
        text = self.messages[episode_idx]["ut"] + " [SEP] " + self.messages[episode_idx]["bt"] + " [EOS]"
        action = {
            'id': self.id,
            'text': text,
            'episode_done': True,
            'labels': [self.messages[episode_idx]['bt_plus_1']],
        }
        return action


class DefaultTeacher(InspiredStatePredictorTeacher):
    pass
