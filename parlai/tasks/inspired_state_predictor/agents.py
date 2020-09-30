#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import FixedDialogTeacher
from parlai.utils.io import PathManager
from parlai.core.message import Message
from typing import List, Tuple, Optional, TypeVar
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
    training_file_path = os.path.join(inspired_state_predictor_folder_path, 'gpt2_state_predictor_training_file_0920_with_related_placeholders.csv')
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
        len_messages = len(self.messages)
        print("message length:{}".format(len_messages))
        train_length = round(len_messages * 0.8)
        valid_length = round(len_messages * 0.9)
        if self.datatype.startswith('test'):
            self.messages = self.messages[valid_length:]
        elif self.datatype.startswith('valid'):
            self.messages = self.messages[train_length:valid_length]
        else:
            self.messages = self.messages[:train_length]
        
    def custom_evaluation(
        self,
        teacher_action: Message,
        labels: Optional[Tuple[str]],
        model_response: Message,
    ) -> None:
        """
        A method designated for hooking custom evaluations into teachers.

        Generally, a user will want to use `self.metrics.add` to record any
        specialized metrics that only make sense for this one dataset.

        :param teacher_action:
            The message last sent from this teacher.
        :param labels:
            The previous correct labels, if there were any.
        :param model_response:
            The raw response from the model. Generally you want to rely on the
            text field, but others may be necessary in specific situations.
        """
        pass
    
    def _get_messages(self, df_training):
        print("[Inspired_Dataset]processing dataset...")
        messages = []
        impatience = 0
        for index, row in df_training.iterrows():
            context = row["context"].lower()
            ut = row["ut"].lower()
            bt = row["bt"].lower()
            
            try:
                bt = ast.literal_eval(bt)
            except:
                bt = []
            if not bt:
                bt = "[chitchat]"
            else:
                bt = " ".join(bt)

            bt_plus_1 = row["bt_plus_1"].lower()
            try:
                bt_plus_1 = ast.literal_eval(bt_plus_1)
            except:
                bt_plus_1 = []
            if not bt_plus_1:
                bt_plus_1 = "[chitchat]"
            else:
                bt_plus_1 = " ".join(bt_plus_1)

            diff = row["diff"].lower()
            try:
                diff = ast.literal_eval(diff)
            except:
                diff = []
            if not diff:
                impatience += 1
            else:
                impatience = 0
            if not diff:
                diff = "[chitchat]"
            else:
                diff = " ".join(diff)
            
            related_placeholders = row["related_placeholders"].lower()
            try:
                related_placeholders = ast.literal_eval(related_placeholders)
            except:
                related_placeholders = []
            if not related_placeholders:
                related_placeholders = "[chitchat]"
            else:
                related_placeholders = " ".join(related_placeholders)

            # if impatience < 3:
            #     row_dic = {"context": context, "ut": ut, "bt": bt, "bt_plus_1": bt_plus_1, "diff": diff, "related_placeholders": related_placeholders}
            #     messages.append(row_dic)
            row_dic = {"context": context, "ut": ut, "bt": bt, "bt_plus_1": bt_plus_1, "diff": diff, "related_placeholders": related_placeholders}
            messages.append(row_dic)
        return messages

    def num_examples(self):
        return len(self.messages)

    def num_episodes(self):
        return len(self.messages)

    def get(self, episode_idx, entry_idx=0):
        text = self.messages[episode_idx]["ut"] + " [SEP] " + self.messages[episode_idx]["bt"]
        # text = self.messages[episode_idx]["bt"]
        action = {
            'id': self.id,
            'text': text,
            'episode_done': True,
            'labels': [self.messages[episode_idx]['related_placeholders']],
        }
        return action


class DefaultTeacher(InspiredStatePredictorTeacher):
    pass
