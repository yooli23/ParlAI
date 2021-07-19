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
    inspired_response_generator_folder_path = os.path.join(opt['datapath'], 'inspired', 'raw_blender_baseline')
    # training_file_path = os.path.join(inspired_response_generator_folder_path, 'inspired_response_generator_training_file_with_placeholders.csv')
    # inspired_response_generator_folder_path = os.path.join(opt['datapath'], 'redial')
    training_file_path = os.path.join(inspired_response_generator_folder_path, 'raw_blender_baseline_model_training_file_0202.csv')
    return training_file_path, inspired_response_generator_folder_path


class InspiredBlenderBaselineTeacher(FixedDialogTeacher):
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
        self.episodes = []
        self._setup_data(opt['datafile'], inspired_response_generator_folder_path)
        self.id = 'inspired_blender_baseline'
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
        #self.messages = self._get_messages(df_training)
        self.episodes = self._get_episodes(df_training)
        # if self.datatype.startswith('test'):
        #     self.messages = self.messages[9507:]
        # elif self.datatype.startswith('valid'):
        #     self.messages = self.messages[8451:9507]
        # else:
        #     self.messages = self.messages[:8451]
        #data_length = len(self.messages)
        num_episode = len(self.episodes)

        if self.datatype.startswith('test'):
            #self.messages = self.messages[int(0.9*data_length):]
            self.episodes = self.episodes[int(0.9*num_episode):]
        elif self.datatype.startswith('valid'):
            #self.messages = self.messages[int(0.8*data_length):int(0.9*data_length)]
            self.episodes = self.episodes[int(0.8*num_episode):int(0.9*num_episode)]
        else:
            #self.messages = self.messages[:int(0.8*data_length)]
            self.episodes = self.episodes[:int(0.8*num_episode)]
        
        
    
    def _get_messages(self, df_training):
        print("[Inspired_Dataset]processing dataset...")
        messages = []
        for index, row in df_training.iterrows():
            context = row["context"].lower()
            # positive_placeholders = row["positive_placeholders"].lower()
            entities = row["entities"].lower()
            response = row["response"].lower()
            # try:
            #     positive_placeholders = ast.literal_eval(positive_placeholders)
            # except:
            #     positive_placeholders = []
            # if not positive_placeholders:
            #     positive_placeholders = "[chitchat]"
            # else:
            #     positive_placeholders = " ".join(positive_placeholders)
            # response = row["response"].lower()
            # if response and response[:4] == " a: ":
            #     response = response[4:]
            # # if response and response[:3] == "a: ":
            # #     response = response[3:]
            row_dic = {"context": context, "entities": entities, "response": response}
            messages.append(row_dic)
        return messages
    
    def _get_episodes(self, df_training):
        print("[Inspired_Dataset]processing dataset...")
        episodes = []
        episode = []
        for index, row in df_training.iterrows():
            context = row["context"].lower()
            user_utterance = row["user_utterance"].lower()
            entities = row["entities"].lower()
            response = row["response"].lower()
            if not user_utterance and episode:
                episodes.append(episode)
                episode = []
                if not user_utterance:
                    user_utterance = "[start]"
                episode.append(user_utterance)
                episode.append(response)
            else:
                if not user_utterance:
                    user_utterance = "[start]"
                episode.append(user_utterance)
                episode.append(response)
        return episodes

    def num_examples(self):
        examples = 0
        for data in self.episodes:
            examples += len(data) // 2
        return examples

    def num_episodes(self):
        return len(self.episodes)

    # def get(self, episode_idx, entry_idx=0):
    #     text = self.messages[episode_idx]["context"] + " [sep] "
    #     action = {
    #         'id': self.id,
    #         'text': text,
    #         'episode_done': True,
    #         'labels': [self.messages[episode_idx]['response']],
    #     }
    #     return action
    
    def get(self, episode_idx, entry_idx=0):
        text_idx = entry_idx * 2
        entry = self.episodes[episode_idx][text_idx]
        final_speaker_idx = len(self.episodes[episode_idx]) - 2
        # sometimes the first speaker is at the end with no reply
        if len(self.episodes[episode_idx]) % 2 == 1:
            final_speaker_idx -= 1
        labels = [self.episodes[episode_idx][text_idx + 1]]
        episode_done = text_idx >= final_speaker_idx
        action = {
            'id': self.id,
            'text': entry,
            'episode_done': episode_done,
            'labels': labels,
        }
        return action


class DefaultTeacher(InspiredBlenderBaselineTeacher):
    pass
