# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: an implementation of a deep learning recommendation model (DLRM)
# The model input consists of dense and sparse features. The former is a vector
# of floating point values. The latter is a list of sparse indices into
# embedding tables, which consist of vectors of floating point values.
# The selected vectors are passed to mlp networks denoted by triangles,
# in some cases the vectors are interacted through operators (Ops).
#
# output:
#                         vector of values
# model:                        |
#                              /\
#                             /__\
#                               |
#       _____________________> Op  <___________________
#     /                         |                      \
#    /\                        /\                      /\
#   /__\                      /__\           ...      /__\
#    |                          |                       |
#    |                         Op                      Op
#    |                    ____/__\_____           ____/__\____
#    |                   |_Emb_|____|__|    ...  |_Emb_|__|___|
# input:
# [ dense features ]     [sparse indices] , ..., [sparse indices]
#
# More precise definition of model layers:
# 1) fully connected layers of an mlp
# z = f(y)
# y = Wx + b
#
# 2) embedding lookup (for a list of sparse indices p=[p1,...,pk])
# z = Op(e1,...,ek)
# obtain vectors e1=E[:,p1], ..., ek=E[:,pk]
#
# 3) Operator Op can be one of the following
# Sum(e1,...,ek) = e1 + ... + ek
# Dot(e1,...,ek) = [e1'e1, ..., e1'ek, ..., ek'e1, ..., ek'ek]
# Cat(e1,...,ek) = [e1', ..., ek']'
# where ' denotes transpose operation
#
# References:
# [1] Maxim Naumov, Dheevatsa Mudigere, Hao-Jun Michael Shi, Jianyu Huang,
# Narayanan Sundaram, Jongsoo Park, Xiaodong Wang, Udit Gupta, Carole-Jean Wu,
# Alisson G. Azzolini, Dmytro Dzhulgakov, Andrey Mallevich, Ilia Cherniavskii,
# Yinghai Lu, Raghuraman Krishnamoorthi, Ansha Yu, Volodymyr Kondratenko,
# Stephanie Pereira, Xianjie Chen, Wenlin Chen, Vijay Rao, Bill Jia, Liang Xiong,
# Misha Smelyanskiy, "Deep Learning Recommendation Model for Personalization and
# Recommendation Systems", CoRR, arXiv:1906.00091, 2019

from __future__ import absolute_import, division, print_function, unicode_literals

# miscellaneous
import builtins
import functools
# import bisect
# import shutil
import time
import json
# data generation
import dlrm_data_pytorch as dp

# numpy
import numpy as np

# onnx
# The onnx import causes deprecation warnings every time workers
# are spawned during testing. So, we filter out those warnings.
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
import onnx

# pytorch
import torch
import torch.nn as nn
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import gather, scatter
# quotient-remainder trick
from tricks.qr_embedding_bag import QREmbeddingBag
# mixed-dimension trick
from tricks.md_embedding_bag import PrEmbeddingBag, md_solver

import sklearn.metrics

# from torchviz import make_dot
# import torch.nn.functional as Functional
# from torch.nn.parameter import Parameter

from torch.optim.lr_scheduler import _LRScheduler

import os.path as osp
from pathlib import Path
from shutil import copyfile
from nncf import NNCFConfig, create_compressed_model
from nncf.initialization import register_default_init_args
from examples.common.utils import configure_logging, configure_paths, create_code_snapshot
from examples.common.sample_config import SampleConfig, create_sample_config
from examples.common.example_logger import logger
from nncf.initialization import InitializingDataLoader

import os
import json
import math
import numpy as np
import pandas as pd
import json
import re
import tarfile
from io import StringIO
from collections import OrderedDict
from copy import deepcopy
from types import SimpleNamespace
from autox.environment.nncf.quantization_env import QuantizationEnv
from autox.agent.ddpg.ddpg import DDPG
from autox.utils.utils import AverageMeter, topk_accuracy, annotate_model_attr
from datetime import datetime

exc = getattr(builtins, "IOError", "FileNotFoundError")

class LRPolicyScheduler(_LRScheduler):
    def __init__(self, optimizer, num_warmup_steps, decay_start_step, num_decay_steps):
        self.num_warmup_steps = num_warmup_steps
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_start_step + num_decay_steps
        self.num_decay_steps = num_decay_steps

        if self.decay_start_step < self.num_warmup_steps:
            sys.exit("Learning rate warmup must finish before the decay starts")

        super(LRPolicyScheduler, self).__init__(optimizer)

    def get_lr(self):
        step_count = self._step_count
        if step_count < self.num_warmup_steps:
            # warmup
            scale = 1.0 - (self.num_warmup_steps - step_count) / self.num_warmup_steps
            lr = [base_lr * scale for base_lr in self.base_lrs]
            self.last_lr = lr
        elif self.decay_start_step <= step_count and step_count < self.decay_end_step:
            # decay
            decayed_steps = step_count - self.decay_start_step
            scale = ((self.num_decay_steps - decayed_steps) / self.num_decay_steps) ** 2
            min_lr = 0.0000001
            lr = [max(min_lr, base_lr * scale) for base_lr in self.base_lrs]
            self.last_lr = lr
        else:
            if self.num_decay_steps > 0:
                # freeze at last, either because we're after decay
                # or because we're between warmup and decay
                lr = self.last_lr
            else:
                # do not adjust
                lr = self.base_lrs
        return lr

### define dlrm in PyTorch ###
class DLRM_Net(nn.Module):
    def create_mlp(self, ln, sigmoid_layer):
        # build MLP layer by layer
        layers = nn.ModuleList()
        for i in range(0, ln.size - 1):
            n = ln[i]
            m = ln[i + 1]

            # construct fully connected operator
            LL = nn.Linear(int(n), int(m), bias=True)

            # initialize the weights
            # with torch.no_grad():
            # custom Xavier input, output or two-sided fill
            mean = 0.0  # std_dev = np.sqrt(variance)
            std_dev = np.sqrt(2 / (m + n))  # np.sqrt(1 / m) # np.sqrt(1 / n)
            W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
            std_dev = np.sqrt(1 / m)  # np.sqrt(2 / (m + 1))
            bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)
            # approach 1
            LL.weight.data = torch.tensor(W, requires_grad=True)
            LL.bias.data = torch.tensor(bt, requires_grad=True)
            # approach 2
            # LL.weight.data.copy_(torch.tensor(W))
            # LL.bias.data.copy_(torch.tensor(bt))
            # approach 3
            # LL.weight = Parameter(torch.tensor(W),requires_grad=True)
            # LL.bias = Parameter(torch.tensor(bt),requires_grad=True)
            layers.append(LL)

            # construct sigmoid or relu operator
            if i == sigmoid_layer:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU())

        # approach 1: use ModuleList
        # return layers
        # approach 2: use Sequential container to wrap all layers
        return torch.nn.Sequential(*layers)

    def create_emb(self, m, ln):
        emb_l = nn.ModuleList()
        for i in range(0, ln.size):
            n = ln[i]
            # construct embedding operator
            if self.qr_flag and n > self.qr_threshold:
                EE = QREmbeddingBag(n, m, self.qr_collisions,
                    operation=self.qr_operation, mode="sum", sparse=True)
            elif self.md_flag:
                base = max(m)
                _m = m[i] if n > self.md_threshold else base
                EE = PrEmbeddingBag(n, _m, base)
                # use np initialization as below for consistency...
                W = np.random.uniform(
                    low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, _m)
                ).astype(np.float32)
                EE.embs.weight.data = torch.tensor(W, requires_grad=True)

            else:
                EE = nn.EmbeddingBag(n, m, mode="sum", sparse=True)

                # initialize embeddings
                # nn.init.uniform_(EE.weight, a=-np.sqrt(1 / n), b=np.sqrt(1 / n))
                W = np.random.uniform(
                    low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
                ).astype(np.float32)
                # approach 1
                EE.weight.data = torch.tensor(W, requires_grad=True)
                # approach 2
                # EE.weight.data.copy_(torch.tensor(W))
                # approach 3
                # EE.weight = Parameter(torch.tensor(W),requires_grad=True)

            emb_l.append(EE)

        return emb_l

    def __init__(
        self,
        m_spa=None,
        ln_emb=None,
        ln_bot=None,
        ln_top=None,
        arch_interaction_op=None,
        arch_interaction_itself=False,
        sigmoid_bot=-1,
        sigmoid_top=-1,
        sync_dense_params=True,
        loss_threshold=0.0,
        ndevices=-1,
        qr_flag=False,
        qr_operation="mult",
        qr_collisions=0,
        qr_threshold=200,
        md_flag=False,
        md_threshold=200,
    ):
        super(DLRM_Net, self).__init__()

        if (
            (m_spa is not None)
            and (ln_emb is not None)
            and (ln_bot is not None)
            and (ln_top is not None)
            and (arch_interaction_op is not None)
        ):

            # save arguments
            self.ndevices = ndevices
            self.output_d = 0
            self.parallel_model_batch_size = -1
            self.parallel_model_is_not_prepared = True
            self.arch_interaction_op = arch_interaction_op
            self.arch_interaction_itself = arch_interaction_itself
            self.sync_dense_params = sync_dense_params
            self.loss_threshold = loss_threshold
            # create variables for QR embedding if applicable
            self.qr_flag = qr_flag
            if self.qr_flag:
                self.qr_collisions = qr_collisions
                self.qr_operation = qr_operation
                self.qr_threshold = qr_threshold
            # create variables for MD embedding if applicable
            self.md_flag = md_flag
            if self.md_flag:
                self.md_threshold = md_threshold
            # create operators
            if ndevices <= 1:
                self.emb_l = self.create_emb(m_spa, ln_emb)
            self.bot_l = self.create_mlp(ln_bot, sigmoid_bot)
            self.top_l = self.create_mlp(ln_top, sigmoid_top)

    def apply_mlp(self, x, layers):
        # approach 1: use ModuleList
        # for layer in layers:
        #     x = layer(x)
        # return x
        # approach 2: use Sequential container to wrap all layers
        return layers(x)

    def apply_emb(self, lS_o, lS_i, emb_l):
        # WARNING: notice that we are processing the batch at once. We implicitly
        # assume that the data is laid out such that:
        # 1. each embedding is indexed with a group of sparse indices,
        #   corresponding to a single lookup
        # 2. for each embedding the lookups are further organized into a batch
        # 3. for a list of embedding tables there is a list of batched lookups

        ly = []
        # for k, sparse_index_group_batch in enumerate(lS_i):
        for k in range(len(lS_i)):
            sparse_index_group_batch = lS_i[k]
            sparse_offset_group_batch = lS_o[k]

            # embedding lookup
            # We are using EmbeddingBag, which implicitly uses sum operator.
            # The embeddings are represented as tall matrices, with sum
            # happening vertically across 0 axis, resulting in a row vector
            E = emb_l[k]
            V = E(sparse_index_group_batch, sparse_offset_group_batch)

            ly.append(V)

        # print(ly)
        return ly

    def interact_features(self, x, ly):
        if self.arch_interaction_op == "dot":
            # concatenate dense and sparse features
            (batch_size, d) = x.shape
            T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d))
            # perform a dot product
            Z = torch.bmm(T, torch.transpose(T, 1, 2))
            # append dense feature with the interactions (into a row vector)
            # approach 1: all
            # Zflat = Z.view((batch_size, -1))
            # approach 2: unique
            _, ni, nj = Z.shape
            # approach 1: tril_indices
            # offset = 0 if self.arch_interaction_itself else -1
            # li, lj = torch.tril_indices(ni, nj, offset=offset)
            # approach 2: custom
            offset = 1 if self.arch_interaction_itself else 0
            li = torch.tensor([i for i in range(ni) for j in range(i + offset)])
            lj = torch.tensor([j for i in range(nj) for j in range(i + offset)])
            Zflat = Z[:, li, lj]
            # concatenate dense features and interactions
            R = torch.cat([x] + [Zflat], dim=1)
        elif self.arch_interaction_op == "cat":
            # concatenation features (into a row vector)
            R = torch.cat([x] + ly, dim=1)
        else:
            sys.exit(
                "ERROR: --arch-interaction-op="
                + self.arch_interaction_op
                + " is not supported"
            )

        return R

    def forward(self, dense_x, lS_o, lS_i):
        if self.ndevices <= 1:
            return self.sequential_forward(dense_x, lS_o, lS_i)
        else:
            return self.parallel_forward(dense_x, lS_o, lS_i)

    def sequential_forward(self, dense_x, lS_o, lS_i):
        # process dense features (using bottom mlp), resulting in a row vector
        x = self.apply_mlp(dense_x, self.bot_l)
        # debug prints
        # print("intermediate")
        # print(x.detach().cpu().numpy())

        # process sparse features(using embeddings), resulting in a list of row vectors
        ly = self.apply_emb(lS_o, lS_i, self.emb_l)
        # for y in ly:
        #     print(y.detach().cpu().numpy())

        # interact features (dense and sparse)
        z = self.interact_features(x, ly)
        # print(z.detach().cpu().numpy())

        # obtain probability of a click (using top mlp)
        p = self.apply_mlp(z, self.top_l)

        # clamp output if needed
        if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
            z = torch.clamp(p, min=self.loss_threshold, max=(1.0 - self.loss_threshold))
        else:
            z = p

        return z

    def parallel_forward(self, dense_x, lS_o, lS_i):
        ### prepare model (overwrite) ###
        # WARNING: # of devices must be >= batch size in parallel_forward call
        batch_size = dense_x.size()[0]
        ndevices = min(self.ndevices, batch_size, len(self.emb_l))
        device_ids = range(ndevices)
        # WARNING: must redistribute the model if mini-batch size changes(this is common
        # for last mini-batch, when # of elements in the dataset/batch size is not even
        if self.parallel_model_batch_size != batch_size:
            self.parallel_model_is_not_prepared = True

        if self.parallel_model_is_not_prepared or self.sync_dense_params:
            # replicate mlp (data parallelism)
            self.bot_l_replicas = replicate(self.bot_l, device_ids)
            self.top_l_replicas = replicate(self.top_l, device_ids)
            self.parallel_model_batch_size = batch_size

        if self.parallel_model_is_not_prepared:
            # distribute embeddings (model parallelism)
            t_list = []
            for k, emb in enumerate(self.emb_l):
                d = torch.device("cuda:" + str(k % ndevices))
                emb.to(d)
                t_list.append(emb.to(d))
            self.emb_l = nn.ModuleList(t_list)
            self.parallel_model_is_not_prepared = False

        ### prepare input (overwrite) ###
        # scatter dense features (data parallelism)
        # print(dense_x.device)
        dense_x = scatter(dense_x, device_ids, dim=0)
        # distribute sparse features (model parallelism)
        if (len(self.emb_l) != len(lS_o)) or (len(self.emb_l) != len(lS_i)):
            sys.exit("ERROR: corrupted model input detected in parallel_forward call")

        t_list = []
        i_list = []
        for k, _ in enumerate(self.emb_l):
            d = torch.device("cuda:" + str(k % ndevices))
            t_list.append(lS_o[k].to(d))
            i_list.append(lS_i[k].to(d))
        lS_o = t_list
        lS_i = i_list

        ### compute results in parallel ###
        # bottom mlp
        # WARNING: Note that the self.bot_l is a list of bottom mlp modules
        # that have been replicated across devices, while dense_x is a tuple of dense
        # inputs that has been scattered across devices on the first (batch) dimension.
        # The output is a list of tensors scattered across devices according to the
        # distribution of dense_x.
        x = parallel_apply(self.bot_l_replicas, dense_x, None, device_ids)
        # debug prints
        # print(x)

        # embeddings
        ly = self.apply_emb(lS_o, lS_i, self.emb_l)
        # debug prints
        # print(ly)

        # butterfly shuffle (implemented inefficiently for now)
        # WARNING: Note that at this point we have the result of the embedding lookup
        # for the entire batch on each device. We would like to obtain partial results
        # corresponding to all embedding lookups, but part of the batch on each device.
        # Therefore, matching the distribution of output of bottom mlp, so that both
        # could be used for subsequent interactions on each device.
        if len(self.emb_l) != len(ly):
            sys.exit("ERROR: corrupted intermediate result in parallel_forward call")

        t_list = []
        for k, _ in enumerate(self.emb_l):
            d = torch.device("cuda:" + str(k % ndevices))
            y = scatter(ly[k], device_ids, dim=0)
            t_list.append(y)
        # adjust the list to be ordered per device
        ly = list(map(lambda y: list(y), zip(*t_list)))
        # debug prints
        # print(ly)

        # interactions
        z = []
        for k in range(ndevices):
            zk = self.interact_features(x[k], ly[k])
            z.append(zk)
        # debug prints
        # print(z)

        # top mlp
        # WARNING: Note that the self.top_l is a list of top mlp modules that
        # have been replicated across devices, while z is a list of interaction results
        # that by construction are scattered across devices on the first (batch) dim.
        # The output is a list of tensors scattered across devices according to the
        # distribution of z.
        p = parallel_apply(self.top_l_replicas, z, None, device_ids)

        ### gather the distributed results ###
        p0 = gather(p, self.output_d, dim=0)

        # clamp output if needed
        if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
            z0 = torch.clamp(
                p0, min=self.loss_threshold, max=(1.0 - self.loss_threshold)
            )
        else:
            z0 = p0

        return z0


def dash_separated_ints(value):
    vals = value.split('-')
    for val in vals:
        try:
            int(val)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "%s is not a valid dash separated list of ints" % value)

    return value


def dash_separated_floats(value):
    vals = value.split('-')
    for val in vals:
        try:
            float(val)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "%s is not a valid dash separated list of floats" % value)

    return value

def search(agent, env, config):
    # def map_precision(action):
    #     action = float(action)
    #     min_bit, max_bit = (1,3)
    #     action = (max_bit - min_bit) * action + min_bit
    #     action = int(np.round(action, 0))
    #     action = 2**action
    #     return int(action)

    def map_precision(action):
        precision_set = [2,4,8]
        precision_set = np.array(sorted(precision_set))
        tuned_point = precision_set+3
        max_bit = max(precision_set)
    
        for i, point in enumerate(tuned_point):
            if action <= 2**point/2**max_bit:
                return int(precision_set[i])
        return int(precision_set[i])

    # def map_precision(action):
    #     action = float(action)
    #     min_exp, max_exp = (1,3)
    #     lbound, rbound = min_exp - 0.5, max_exp + 0.5
    #     action = (rbound - lbound) * action + lbound
    #     action = np.round(action, 0)
    #     action = np.clip(action, min_exp, max_exp)
    #     action = int(2**action)
    #     return action

    args = SimpleNamespace(**config.get('auto_quantization', {}))

    policy_dict=OrderedDict() #key: episode
    best_policy_dict=OrderedDict() #key: episode

    num_episode = args.train_episode

    # best record
    best_reward = -math.inf
    best_policy = []

    tfwriter = config['tb']

    log_cfg=OrderedDict()
    log_cfg['compression']=config['compression']
    log_cfg['auto_quantization']=config['auto_quantization']
    tfwriter.add_text('info/run_config', json.dumps(log_cfg, indent=4, sort_keys=False).replace("\n","\n\n"), 0)
    tfwriter.add_text('info/state_embedding', env.master_df[env.state_list].to_markdown())

    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    observation = None
    T = []  # Transition buffer

    while episode < num_episode:  # counting based on episode
        episode_start_ts = time.time()
        if observation is None:
            # reset if it is the start of episode
            env.reset()
            observation = deepcopy(env.get_normalized_obs(0))
            agent.reset(observation)
            
        if episode <= args.warmup:
            action = agent.random_action()

        else:
            action = agent.select_action(observation, episode=episode)

        # env response with next_observation, reward, terminate_info
        observation2, reward, done, info = env.step(map_precision(action))
        observation2 = deepcopy(observation2)
        T.append([reward, deepcopy(observation), deepcopy(observation2), action, done])

        # [optional] save intermideate model
        if episode % int(num_episode / 10) == 0:
            agent.save_model(config['checkpoint_save_dir'])

        # update
        step += 1   
        episode_steps += 1
        episode_reward += reward
        observation = deepcopy(observation2)

        if done:  # end of episode
            logger.info(
                '#{}: episode_reward:{:.3f} acc: {:.3f}, model_ratio: {:.3f}, model_size(MB): {:.2f}\n'.format(
                    episode, episode_reward, info['accuracy'], info['model_ratio'], info['model_size']/8e6))

            final_reward = T[-1][0]

            for i, (r_t, s_t, s_t1, a_t, done) in enumerate(T):
                # Revision of prev_action as it could be modified by constrainer -------
                if i == 0:
                    prev_action = 0.0
                else:
                    if env.tie_quantizers:
                        prev_action = env.master_df['action'][env.master_df.is_pred][i-1] / 8 #ducktape scaling
                    else:
                        prev_action = env.master_df['action'][i-1] / 8 #ducktape scaling

                if prev_action != s_t['prev_action']:
                    print(i, prev_action, " <= ", s_t['prev_action'])
                    s_t['prev_action'] = prev_action
                # EO ------------------------

                agent.observe(final_reward, s_t, s_t1, a_t, done)
                if episode > args.warmup:
                    for i in range(args.n_update):
                        agent.update_policy()

            agent.memory.append(
                observation,
                agent.select_action(observation, episode=episode),
                0., False
            )

            # reset
            observation = None
            episode_steps = 0
            episode_reward = 0.
            episode += 1
            T = []
            
            # Save nncf compression cfg
            episode_cfgfile = osp.join(env.config['episodic_nncfcfg'], '{0:03d}_nncfcfg.json'.format(episode))
            with open(episode_cfgfile, "w") as outfile: 
                json.dump(env.config.nncf_config, outfile, indent=4, sort_keys=False) 

            bit_stats_tt = env.qctrl.statistics()['Bitwidth distribution:']
            bit_stats_tt.set_max_width(100)
            bit_stats_df = pd.read_csv(StringIO(re.sub(r'[-+|=]', '', bit_stats_tt.draw())), sep='\s{2,}', engine='python').reset_index(drop=True)
        
            policy_dict[episode]=env.master_df['action'].astype('int')
            pd.DataFrame(policy_dict.values(), index=policy_dict.keys()).T.sort_index(axis=1, ascending=False).to_csv(osp.join(config.log_dir, "policy_per_episode.csv"), index_label="nodestr")
            
            if final_reward > best_reward:
                best_reward = final_reward
                best_policy = env.strategy
                
                # log best policy to tensorboard
                best_policy_dict[episode]=env.master_df['action'].astype('int')
                pd.DataFrame(best_policy_dict.values(), index=best_policy_dict.keys()).T.sort_index(axis=1, ascending=False).to_csv(osp.join(config.log_dir, "best_policy.csv"), index_label="nodestr")
            
                best_policy_string = bit_stats_df.to_markdown() + "\n\n\n"
                best_policy_string += "Episode: {}, Reward: {:.3f}, Accuracy: {:.3f}, Model_Ratio: {:.3f}\n\n\n".format(episode, final_reward, info['accuracy'], info['model_ratio'])
                for i, nodestr in enumerate(env.master_df.index.tolist()):
                    Qtype=' (W)' if env.master_df.is_wt_quantizer[nodestr] else ' (A)'
                    if env.skip_wall:
                        best_policy_string += str(int(env.master_df.loc[nodestr, 'action'])) + " | " + nodestr + Qtype + "  \n"
                    else:
                        if env.master_df.loc[nodestr, 'action'] == env.master_df.loc[nodestr, 'unconstrained_action']:
                            best_policy_string += str(int(env.master_df.loc[nodestr, 'action'])) + " | " + nodestr + Qtype + "  \n"
                        else:
                            best_policy_string += str(int(env.master_df.loc[nodestr, 'action'])) + " <= " + str(int(env.master_df.loc[nodestr, 'unconstrained_action'])) + " | " + nodestr + Qtype + "  \n"

                tfwriter.add_text('info/best_policy', best_policy_string, episode)


            # log current policy to tensorboard
            current_strategy_string = bit_stats_df.to_markdown() + "\n\n\n"
            current_strategy_string += "Episode: {}, Reward: {:.3f}, Accuracy: {:.3f}, Model_Ratio: {:.3f}\n\n\n".format(episode, final_reward, info['accuracy'], info['model_ratio'])
            for i, nodestr in enumerate(env.master_df.index.tolist()):
                Qtype=' (W)' if env.master_df.is_wt_quantizer[nodestr] else ' (A)'
                if env.skip_wall is True:
                    current_strategy_string += str(int(env.master_df.loc[nodestr, 'action'])) + " | " + nodestr + Qtype + "  \n"
                else:
                    if env.master_df.loc[nodestr, 'action'] == env.master_df.loc[nodestr, 'unconstrained_action']:
                        current_strategy_string += str(int(env.master_df.loc[nodestr, 'action'])) + " | " + nodestr + Qtype + "  \n"
                    else:
                        current_strategy_string += str(int(env.master_df.loc[nodestr, 'action'])) + " <= " + str(int(env.master_df.loc[nodestr, 'unconstrained_action'])) + " | " + nodestr + Qtype + "  \n"
                    
            tfwriter.add_text('info/current_policy', current_strategy_string, episode)

            value_loss = agent.get_value_loss()
            policy_loss = agent.get_policy_loss()
            delta = agent.get_delta()

            tfwriter.add_scalar('reward/last', final_reward, episode)
            tfwriter.add_scalar('reward/best', best_reward, episode)
            tfwriter.add_scalar('info/accuracy', info['accuracy'], episode)
            tfwriter.add_scalar('info/model_ratio', info['model_ratio'], episode)
            tfwriter.add_scalar('agent/value_loss', value_loss, episode)
            tfwriter.add_scalar('agent/policy_loss', policy_loss, episode)
            tfwriter.add_scalar('agent/delta', delta, episode)
            
            logger.info('best reward: {}\n'.format(best_reward))
            logger.info('best policy: {}\n'.format(best_policy))

            episode_elapsed = time.time() - episode_start_ts
            logger.info('\n### Episode[{}] Elapsed: {:.3f}\n'.format(episode-1, episode_elapsed))

    return best_policy, best_reward

if __name__ == "__main__":
    ### import packages ###
    import sys
    import argparse

    ### parse arguments ###
    parser = argparse.ArgumentParser(
        description="Train Deep Learning Recommendation Model (DLRM)"
    )
    # model related parameters
    parser.add_argument("--arch-sparse-feature-size", type=int, default=2)
    parser.add_argument(
        "--arch-embedding-size", type=dash_separated_ints, default="4-3-2")
    # j will be replaced with the table number
    parser.add_argument(
        "--arch-mlp-bot", type=dash_separated_ints, default="4-3-2")
    parser.add_argument(
        "--arch-mlp-top", type=dash_separated_ints, default="4-2-1")
    parser.add_argument(
        "--arch-interaction-op", type=str, choices=['dot', 'cat'], default="dot")
    parser.add_argument("--arch-interaction-itself", action="store_true", default=False)
    # embedding table options
    parser.add_argument("--md-flag", action="store_true", default=False)
    parser.add_argument("--md-threshold", type=int, default=200)
    parser.add_argument("--md-temperature", type=float, default=0.3)
    parser.add_argument("--md-round-dims", action="store_true", default=False)
    parser.add_argument("--qr-flag", action="store_true", default=False)
    parser.add_argument("--qr-threshold", type=int, default=200)
    parser.add_argument("--qr-operation", type=str, default="mult")
    parser.add_argument("--qr-collisions", type=int, default=4)
    # activations and loss
    parser.add_argument("--activation-function", type=str, default="relu")
    parser.add_argument("--loss-function", type=str, default="mse")  # or bce or wbce
    parser.add_argument(
        "--loss-weights", type=dash_separated_floats, default="1.0-1.0")  # for wbce
    parser.add_argument("--loss-threshold", type=float, default=0.0)  # 1.0e-7
    parser.add_argument("--round-targets", type=bool, default=False)
    # data
    parser.add_argument("--data-size", type=int, default=1)
    parser.add_argument("--num-batches", type=int, default=0)
    parser.add_argument(
        "--data-generation", type=str, default="random"
    )  # synthetic or dataset
    parser.add_argument("--data-trace-file", type=str, default="./input/dist_emb_j.log")
    parser.add_argument("--data-set", type=str, default="kaggle")  # or terabyte
    parser.add_argument("--raw-data-file", type=str, default="")
    parser.add_argument("--processed-data-file", type=str, default="")
    parser.add_argument("--data-randomize", type=str, default="total")  # or day or none
    parser.add_argument("--data-trace-enable-padding", type=bool, default=False)
    parser.add_argument("--max-ind-range", type=int, default=-1)
    parser.add_argument("--data-sub-sample-rate", type=float, default=0.0)  # in [0, 1]
    parser.add_argument("--num-indices-per-lookup", type=int, default=10)
    parser.add_argument("--num-indices-per-lookup-fixed", type=bool, default=False)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--memory-map", action="store_true", default=False)
    parser.add_argument("--dataset-multiprocessing", action="store_true", default=False,
                        help="The Kaggle dataset can be multiprocessed in an environment \
                        with more than 7 CPU cores and more than 20 GB of memory. \n \
                        The Terabyte dataset can be multiprocessed in an environment \
                        with more than 24 CPU cores and at least 1 TB of memory.")
    # training
    parser.add_argument("--mini-batch-size", type=int, default=1)
    parser.add_argument("--nepochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--print-precision", type=int, default=5)
    parser.add_argument("--numpy-rand-seed", type=int, default=123)
    parser.add_argument("--sync-dense-params", type=bool, default=True)
    # inference
    parser.add_argument("--inference-only", action="store_true", default=False)
    # onnx
    parser.add_argument("--save-onnx", action="store_true", default=False)
    # gpu
    parser.add_argument("--use-gpu", action="store_true", default=False)
    # debugging and profiling
    parser.add_argument("--print-freq", type=int, default=1)
    parser.add_argument("--test-freq", type=int, default=-1)
    parser.add_argument("--test-mini-batch-size", type=int, default=-1)
    parser.add_argument("--test-num-workers", type=int, default=-1)
    parser.add_argument("--print-time", action="store_true", default=False)
    parser.add_argument("--debug-mode", action="store_true", default=False)
    parser.add_argument("--enable-profiling", action="store_true", default=False)
    parser.add_argument("--plot-compute-graph", action="store_true", default=False)
    # store/load model
    parser.add_argument("--save-model", type=str, default="")
    parser.add_argument("--load-model", type=str, default="")
    # mlperf logging (disables other output and stops early)
    parser.add_argument("--mlperf-logging", action="store_true", default=False)
    # stop at target accuracy Kaggle 0.789, Terabyte (sub-sampled=0.875) 0.8107
    parser.add_argument("--mlperf-acc-threshold", type=float, default=0.0)
    # stop at target AUC Terabyte (no subsampling) 0.8025
    parser.add_argument("--mlperf-auc-threshold", type=float, default=0.0)
    parser.add_argument("--mlperf-bin-loader", action='store_true', default=False)
    parser.add_argument("--mlperf-bin-shuffle", action='store_true', default=False)
    # LR policy
    parser.add_argument("--lr-num-warmup-steps", type=int, default=0)
    parser.add_argument("--lr-decay-start-step", type=int, default=0)
    parser.add_argument("--lr-num-decay-steps", type=int, default=0)

    # NNCF
    parser.add_argument('--nncf_config', type=str, help='path to NNCF config .json file to be used for compressed model')
    parser.add_argument("--log-dir", type=str, default='runs',
        help="The directory where models and TensorboardX summaries"
             " are saved. Default: runs")

    print_fn = print

    args = parser.parse_args()

    if args.mlperf_logging:
        print('command line args: ', json.dumps(vars(args)))

    ### some basic setup ###
    np.random.seed(args.numpy_rand_seed)
    np.set_printoptions(precision=args.print_precision)
    torch.set_printoptions(precision=args.print_precision)
    torch.manual_seed(args.numpy_rand_seed)

    if (args.test_mini_batch_size < 0):
        # if the parameter is not set, use the training batch size
        args.test_mini_batch_size = args.mini_batch_size
    if (args.test_num_workers < 0):
        # if the parameter is not set, use the same parameter for training
        args.test_num_workers = args.num_workers

    use_gpu = args.use_gpu and torch.cuda.is_available()
    if use_gpu:
        torch.cuda.manual_seed_all(args.numpy_rand_seed)
        torch.backends.cudnn.deterministic = True
        device = torch.device("cuda", 0)
        ngpus = torch.cuda.device_count()  # 1
        print("Using {} GPU(s)...".format(ngpus))
    else:
        device = torch.device("cpu")
        print("Using CPU...")

    ### prepare training data ###
    dataprep_t1 = time.time()
    ln_bot = np.fromstring(args.arch_mlp_bot, dtype=int, sep="-")
    # input data
    if (args.data_generation == "dataset"):

        train_data, train_ld, test_data, test_ld = \
            dp.make_criteo_data_and_loaders(args)
        nbatches = args.num_batches if args.num_batches > 0 else len(train_ld)
        nbatches_test = len(test_ld)

        ln_emb = train_data.counts
        # enforce maximum limit on number of vectors per embedding
        if args.max_ind_range > 0:
            ln_emb = np.array(list(map(
                lambda x: x if x < args.max_ind_range else args.max_ind_range,
                ln_emb
            )))
        m_den = train_data.m_den
        ln_bot[0] = m_den
    else:
        # input and target at random
        ln_emb = np.fromstring(args.arch_embedding_size, dtype=int, sep="-")
        m_den = ln_bot[0]
        train_data, train_ld = dp.make_random_data_and_loader(args, ln_emb, m_den)
        nbatches = args.num_batches if args.num_batches > 0 else len(train_ld)

    print("Data Preparation Elapse Time: {}".format(time.time()-dataprep_t1))
    ### parse command line arguments ###
    m_spa = args.arch_sparse_feature_size
    num_fea = ln_emb.size + 1  # num sparse + num dense features
    m_den_out = ln_bot[ln_bot.size - 1]
    if args.arch_interaction_op == "dot":
        # approach 1: all
        # num_int = num_fea * num_fea + m_den_out
        # approach 2: unique
        if args.arch_interaction_itself:
            num_int = (num_fea * (num_fea + 1)) // 2 + m_den_out
        else:
            num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out
    elif args.arch_interaction_op == "cat":
        num_int = num_fea * m_den_out
    else:
        sys.exit(
            "ERROR: --arch-interaction-op="
            + args.arch_interaction_op
            + " is not supported"
        )
    arch_mlp_top_adjusted = str(num_int) + "-" + args.arch_mlp_top
    ln_top = np.fromstring(arch_mlp_top_adjusted, dtype=int, sep="-")

    # sanity check: feature sizes and mlp dimensions must match
    if m_den != ln_bot[0]:
        sys.exit(
            "ERROR: arch-dense-feature-size "
            + str(m_den)
            + " does not match first dim of bottom mlp "
            + str(ln_bot[0])
        )
    if args.qr_flag:
        if args.qr_operation == "concat" and 2 * m_spa != m_den_out:
            sys.exit(
                "ERROR: 2 arch-sparse-feature-size "
                + str(2 * m_spa)
                + " does not match last dim of bottom mlp "
                + str(m_den_out)
                + " (note that the last dim of bottom mlp must be 2x the embedding dim)"
            )
        if args.qr_operation != "concat" and m_spa != m_den_out:
            sys.exit(
                "ERROR: arch-sparse-feature-size "
                + str(m_spa)
                + " does not match last dim of bottom mlp "
                + str(m_den_out)
            )
    else:
        if m_spa != m_den_out:
            sys.exit(
                "ERROR: arch-sparse-feature-size "
                + str(m_spa)
                + " does not match last dim of bottom mlp "
                + str(m_den_out)
            )
    if num_int != ln_top[0]:
        sys.exit(
            "ERROR: # of feature interactions "
            + str(num_int)
            + " does not match first dimension of top mlp "
            + str(ln_top[0])
        )

    # assign mixed dimensions if applicable
    if args.md_flag:
        m_spa = md_solver(
            torch.tensor(ln_emb),
            args.md_temperature,  # alpha
            d0=m_spa,
            round_dim=args.md_round_dims
        ).tolist()

    # test prints (model arch)
    if args.debug_mode:
        print("model arch:")
        print(
            "mlp top arch "
            + str(ln_top.size - 1)
            + " layers, with input to output dimensions:"
        )
        print(ln_top)
        print("# of interactions")
        print(num_int)
        print(
            "mlp bot arch "
            + str(ln_bot.size - 1)
            + " layers, with input to output dimensions:"
        )
        print(ln_bot)
        print("# of features (sparse and dense)")
        print(num_fea)
        print("dense feature size")
        print(m_den)
        print("sparse feature size")
        print(m_spa)
        print(
            "# of embeddings (= # of sparse features) "
            + str(ln_emb.size)
            + ", with dimensions "
            + str(m_spa)
            + "x:"
        )
        print(ln_emb)

        print("data (inputs and targets):")
        for j, (X, lS_o, lS_i, T) in enumerate(train_ld):
            # early exit if nbatches was set by the user and has been exceeded
            if nbatches > 0 and j >= nbatches:
                break

            print("mini-batch: %d" % j)
            print(X.detach().cpu().numpy())
            # transform offsets to lengths when printing
            print(
                [
                    np.diff(
                        S_o.detach().cpu().tolist() + list(lS_i[i].shape)
                    ).tolist()
                    for i, S_o in enumerate(lS_o)
                ]
            )
            print([S_i.detach().cpu().tolist() for S_i in lS_i])
            print(T.detach().cpu().numpy())

    ndevices = min(ngpus, args.mini_batch_size, num_fea - 1) if use_gpu else -1

    ### construct the neural network specified above ###
    # WARNING: to obtain exactly the same initialization for
    # the weights we need to start from the same random seed.
    # np.random.seed(args.numpy_rand_seed)
    dlrm = DLRM_Net(
        m_spa,
        ln_emb,
        ln_bot,
        ln_top,
        arch_interaction_op=args.arch_interaction_op,
        arch_interaction_itself=args.arch_interaction_itself,
        sigmoid_bot=-1,
        sigmoid_top=ln_top.size - 2,
        sync_dense_params=args.sync_dense_params,
        loss_threshold=args.loss_threshold,
        ndevices=ndevices,
        qr_flag=args.qr_flag,
        qr_operation=args.qr_operation,
        qr_collisions=args.qr_collisions,
        qr_threshold=args.qr_threshold,
        md_flag=args.md_flag,
        md_threshold=args.md_threshold,
    )
    # test prints
    if args.debug_mode:
        print("initial parameters (weights and bias):")
        for param in dlrm.parameters():
            print(param.detach().cpu().numpy())
        # print(dlrm)

    if use_gpu:
        # Custom Model-Data Parallel
        # the mlps are replicated and use data parallelism, while
        # the embeddings are distributed and use model parallelism
        dlrm = dlrm.to(device)  # .cuda()
        if dlrm.ndevices > 1:
            dlrm.emb_l = dlrm.create_emb(m_spa, ln_emb)

    # specify the loss function
    if args.loss_function == "mse":
        loss_fn = torch.nn.MSELoss(reduction="mean")
    elif args.loss_function == "bce":
        loss_fn = torch.nn.BCELoss(reduction="mean")
    elif args.loss_function == "wbce":
        loss_ws = torch.tensor(np.fromstring(args.loss_weights, dtype=float, sep="-"))
        loss_fn = torch.nn.BCELoss(reduction="none")
    else:
        sys.exit("ERROR: --loss-function=" + args.loss_function + " is not supported")

    if not args.inference_only:
        # specify the optimizer algorithm
        optimizer = torch.optim.SGD(dlrm.parameters(), lr=args.learning_rate)
        lr_scheduler = LRPolicyScheduler(optimizer, args.lr_num_warmup_steps, args.lr_decay_start_step,
                                         args.lr_num_decay_steps)

    ### main loop ###
    def time_wrap(use_gpu):
        if use_gpu:
            torch.cuda.synchronize()
        return time.time()

    def dlrm_wrap(dlrm, X, lS_o, lS_i, use_gpu, device):
        if use_gpu:  # .cuda()
            # lS_i can be either a list of tensors or a stacked tensor.
            # Handle each case below:
            lS_i = [S_i.to(device) for S_i in lS_i] if isinstance(lS_i, list) \
                else lS_i.to(device)
            lS_o = [S_o.to(device) for S_o in lS_o] if isinstance(lS_o, list) \
                else lS_o.to(device)
            return dlrm(
                X.to(device),
                lS_o,
                lS_i
            )
        else:
            return dlrm(X, lS_o, lS_i)

    def loss_fn_wrap(Z, T, use_gpu, device):
        if args.loss_function == "mse" or args.loss_function == "bce":
            if use_gpu:
                return loss_fn(Z, T.to(device))
            else:
                return loss_fn(Z, T)
        elif args.loss_function == "wbce":
            if use_gpu:
                loss_ws_ = loss_ws[T.data.view(-1).long()].view_as(T).to(device)
                loss_fn_ = loss_fn(Z, T.to(device))
            else:
                loss_ws_ = loss_ws[T.data.view(-1).long()].view_as(T)
                loss_fn_ = loss_fn(Z, T.to(device))
            loss_sc_ = loss_ws_ * loss_fn_
            # debug prints
            # print(loss_ws_)
            # print(loss_fn_)
            return loss_sc_.mean()

    # training or inference
    best_gA_test = 0
    best_auc_test = 0
    skip_upto_epoch = 0
    skip_upto_batch = 0
    total_time = 0
    total_loss = 0
    total_accu = 0
    total_iter = 0
    total_samp = 0
    k = 0

    # Load model is specified
    if not (args.load_model == ""):
        print("Loading saved model {}".format(args.load_model))
        if use_gpu:
            if dlrm.ndevices > 1:
                # NOTE: when targeting inference on multiple GPUs,
                # load the model as is on CPU or GPU, with the move
                # to multiple GPUs to be done in parallel_forward
                ld_model = torch.load(args.load_model)
            else:
                # NOTE: when targeting inference on single GPU,
                # note that the call to .to(device) has already happened
                ld_model = torch.load(
                    args.load_model,
                    map_location=torch.device('cuda')
                    # map_location=lambda storage, loc: storage.cuda(0)
                )
        else:
            # when targeting inference on CPU
            ld_model = torch.load(args.load_model, map_location=torch.device('cpu'))
        dlrm.load_state_dict(ld_model["state_dict"])
        ld_j = ld_model["iter"]
        ld_k = ld_model["epoch"]
        ld_nepochs = ld_model["nepochs"]
        ld_nbatches = ld_model["nbatches"]
        ld_nbatches_test = ld_model["nbatches_test"]
        ld_gA = ld_model["train_acc"]
        ld_gL = ld_model["train_loss"]
        ld_total_loss = ld_model["total_loss"]
        ld_total_accu = ld_model["total_accu"]
        ld_gA_test = ld_model["test_acc"]
        ld_gL_test = ld_model["test_loss"]
        if not args.inference_only:
            optimizer.load_state_dict(ld_model["opt_state_dict"])
            best_gA_test = ld_gA_test
            total_loss = ld_total_loss
            total_accu = ld_total_accu
            skip_upto_epoch = ld_k  # epochs
            skip_upto_batch = ld_j  # batches
        else:
            args.print_freq = ld_nbatches
            args.test_freq = 0

        print(
            "Saved at: epoch = {:d}/{:d}, batch = {:d}/{:d}, ntbatch = {:d}".format(
                ld_k, ld_nepochs, ld_j, ld_nbatches, ld_nbatches_test
            )
        )
        print(
            "Training state: loss = {:.6f}, accuracy = {:3.3f} %".format(
                ld_gL, ld_gA * 100
            )
        )
        print(
            "Testing state: loss = {:.6f}, accuracy = {:3.3f} %".format(
                ld_gL_test, ld_gA_test * 100
            )
        )

    # print("time/loss/accuracy (if enabled):")
    # with torch.autograd.profiler.profile(args.enable_profiling, use_gpu) as prof:
    #     while k < args.nepochs:
    #         if k < skip_upto_epoch:
    #             continue

    #         accum_time_begin = time_wrap(use_gpu)

    #         if args.mlperf_logging:
    #             previous_iteration_time = None

    #         for j, (X, lS_o, lS_i, T) in enumerate(train_ld):
    #             if j == 0 and args.save_onnx:
    #                 (X_onnx, lS_o_onnx, lS_i_onnx) = (X, lS_o, lS_i)

    #             if j < skip_upto_batch:
    #                 continue

    #             if args.mlperf_logging:
    #                 current_time = time_wrap(use_gpu)
    #                 if previous_iteration_time:
    #                     iteration_time = current_time - previous_iteration_time
    #                 else:
    #                     iteration_time = 0
    #                 previous_iteration_time = current_time
    #             else:
    #                 t1 = time_wrap(use_gpu)

    #             # early exit if nbatches was set by the user and has been exceeded
    #             if nbatches > 0 and j >= nbatches:
    #                 break
    #             '''
    #             # debug prints
    #             print("input and targets")
    #             print(X.detach().cpu().numpy())
    #             print([np.diff(S_o.detach().cpu().tolist()
    #                    + list(lS_i[i].shape)).tolist() for i, S_o in enumerate(lS_o)])
    #             print([S_i.detach().cpu().numpy().tolist() for S_i in lS_i])
    #             print(T.detach().cpu().numpy())
    #             '''

    #             # forward pass
    #             Z = dlrm_wrap(X, lS_o, lS_i, use_gpu, device)

    #             # loss
    #             E = loss_fn_wrap(Z, T, use_gpu, device)
    #             '''
    #             # debug prints
    #             print("output and loss")
    #             print(Z.detach().cpu().numpy())
    #             print(E.detach().cpu().numpy())
    #             '''
    #             # compute loss and accuracy
    #             L = E.detach().cpu().numpy()  # numpy array
    #             S = Z.detach().cpu().numpy()  # numpy array
    #             T = T.detach().cpu().numpy()  # numpy array
    #             mbs = T.shape[0]  # = args.mini_batch_size except maybe for last
    #             A = np.sum((np.round(S, 0) == T).astype(np.uint8))

    #             if not args.inference_only:
    #                 # scaled error gradient propagation
    #                 # (where we do not accumulate gradients across mini-batches)
    #                 optimizer.zero_grad()
    #                 # backward pass
    #                 E.backward()
    #                 # debug prints (check gradient norm)
    #                 # for l in mlp.layers:
    #                 #     if hasattr(l, 'weight'):
    #                 #          print(l.weight.grad.norm().item())

    #                 # optimizer
    #                 optimizer.step()
    #                 lr_scheduler.step()

    #             if args.mlperf_logging:
    #                 total_time += iteration_time
    #             else:
    #                 t2 = time_wrap(use_gpu)
    #                 total_time += t2 - t1
    #             total_accu += A
    #             total_loss += L * mbs
    #             total_iter += 1
    #             total_samp += mbs

    #             should_print = ((j + 1) % args.print_freq == 0) or (j + 1 == nbatches)
    #             should_test = (
    #                 (args.test_freq > 0)
    #                 and (args.data_generation == "dataset")
    #                 and (((j + 1) % args.test_freq == 0) or (j + 1 == nbatches))
    #             )

    #             # print time, loss and accuracy
    #             if should_print or should_test:
    #                 gT = 1000.0 * total_time / total_iter if args.print_time else -1
    #                 total_time = 0

    #                 gA = total_accu / total_samp
    #                 total_accu = 0

    #                 gL = total_loss / total_samp
    #                 total_loss = 0

    #                 str_run_type = "inference" if args.inference_only else "training"
    #                 print(
    #                     "Finished {} it {}/{} of epoch {}, {:.2f} ms/it, ".format(
    #                         str_run_type, j + 1, nbatches, k, gT
    #                     )
    #                     + "loss {:.6f}, accuracy {:3.3f} %".format(gL, gA * 100)
    #                 )
    #                 # Uncomment the line below to print out the total time with overhead
    #                 # print("Accumulated time so far: {}" \
    #                 # .format(time_wrap(use_gpu) - accum_time_begin))
    #                 total_iter = 0
    #                 total_samp = 0

    # testing
    # if should_test and not args.inference_only:
    # =========================================================================================================================
    def test_dlrm(model, test_ld):
        # don't measure training iter time in a test iteration
        if args.mlperf_logging:
            previous_iteration_time = None

        test_accu = 0
        test_loss = 0
        test_samp = 0

        accum_test_time_begin = time_wrap(use_gpu)
        if args.mlperf_logging:
            scores = []
            targets = []

        test_nbatch = len(test_ld)            
        print_period =  test_nbatch // 20
        for i, (X_test, lS_o_test, lS_i_test, T_test) in enumerate(test_ld):
            # early exit if nbatches was set by the user and was exceeded
            if nbatches > 0 and i >= nbatches:
                break

            t1_test = time_wrap(use_gpu)

            # forward pass
            Z_test = dlrm_wrap(
                model, X_test, lS_o_test, lS_i_test, use_gpu, device
            )
            if args.mlperf_logging:
                S_test = Z_test.detach().cpu().numpy()  # numpy array
                T_test = T_test.detach().cpu().numpy()  # numpy array
                scores.append(S_test)
                targets.append(T_test)
            else:
                # loss
                E_test = loss_fn_wrap(Z_test, T_test, use_gpu, device)

                # compute loss and accuracy
                L_test = E_test.detach().cpu().numpy()  # numpy array
                S_test = Z_test.detach().cpu().numpy()  # numpy array
                T_test = T_test.detach().cpu().numpy()  # numpy array
                mbs_test = T_test.shape[0]  # = mini_batch_size except last
                A_test = np.sum((np.round(S_test, 0) == T_test).astype(np.uint8))
                test_accu += A_test
                test_loss += L_test * mbs_test
                test_samp += mbs_test

                if (i+1) % print_period == 0:
                    print(
                        "Testing at - {}/{},".format(i + 1, test_nbatch) + 
                        " loss {:.6f}, accuracy {:3.3f} %".format(
                            test_loss / test_samp, test_accu / test_samp * 100)
                    )

            t2_test = time_wrap(use_gpu)

        if args.mlperf_logging:
            scores = np.concatenate(scores, axis=0)
            targets = np.concatenate(targets, axis=0)

            metrics = {
                'loss' : sklearn.metrics.log_loss,
                'recall' : lambda y_true, y_score:
                sklearn.metrics.recall_score(
                    y_true=y_true,
                    y_pred=np.round(y_score)
                ),
                'precision' : lambda y_true, y_score:
                sklearn.metrics.precision_score(
                    y_true=y_true,
                    y_pred=np.round(y_score)
                ),
                'f1' : lambda y_true, y_score:
                sklearn.metrics.f1_score(
                    y_true=y_true,
                    y_pred=np.round(y_score)
                ),
                'ap' : sklearn.metrics.average_precision_score,
                'roc_auc' : sklearn.metrics.roc_auc_score,
                'accuracy' : lambda y_true, y_score:
                sklearn.metrics.accuracy_score(
                    y_true=y_true,
                    y_pred=np.round(y_score)
                ),
                # 'pre_curve' : sklearn.metrics.precision_recall_curve,
                # 'roc_curve' :  sklearn.metrics.roc_curve,
            }

            # print("Compute time for validation metric : ", end="")
            # first_it = True
            validation_results = {}
            for metric_name, metric_function in metrics.items():
                # if first_it:
                #     first_it = False
                # else:
                #     print(", ", end="")
                # metric_compute_start = time_wrap(False)
                validation_results[metric_name] = metric_function(
                    targets,
                    scores
                )
                # metric_compute_end = time_wrap(False)
                # met_time = metric_compute_end - metric_compute_start
                # print("{} {:.4f}".format(metric_name, 1000 * (met_time)),
                #      end="")
            # print(" ms")
            gA_test = validation_results['accuracy']
            gL_test = validation_results['loss']
        else:
            gA_test = test_accu / test_samp
            gL_test = test_loss / test_samp
        
        best_gA_test = gA_test # bypass error "local variable 'best_gA_test' referenced before assignment"
        is_best = gA_test > best_gA_test
        if is_best:
            best_gA_test = gA_test
            if not (args.save_model == ""):
                print("Saving model to {}".format(args.save_model))
                torch.save(
                    {
                        "epoch": k,
                        "nepochs": args.nepochs,
                        "nbatches": nbatches,
                        "nbatches_test": nbatches_test,
                        "iter": j + 1,
                        "state_dict": dlrm.state_dict(),
                        "train_acc": gA,
                        "train_loss": gL,
                        "test_acc": gA_test,
                        "test_loss": gL_test,
                        "total_loss": total_loss,
                        "total_accu": total_accu,
                        "opt_state_dict": optimizer.state_dict(),
                    },
                    args.save_model,
                )

        if args.mlperf_logging:
            is_best = validation_results['roc_auc'] > best_auc_test
            if is_best:
                best_auc_test = validation_results['roc_auc']

            print(
                "Testing at - {}/{} of epoch {},".format(j + 1, nbatches, k)
                + " loss {:.6f}, recall {:.4f}, precision {:.4f},".format(
                    validation_results['loss'],
                    validation_results['recall'],
                    validation_results['precision']
                )
                + " f1 {:.4f}, ap {:.4f},".format(
                    validation_results['f1'],
                    validation_results['ap'],
                )
                + " auc {:.4f}, best auc {:.4f},".format(
                    validation_results['roc_auc'],
                    best_auc_test
                )
                + " accuracy {:3.3f} %, best accuracy {:3.3f} %".format(
                    validation_results['accuracy'] * 100,
                    best_gA_test * 100
                )
            )
        else:
            print(
                # "Testing at - {}/{} of epoch {},".format(j + 1, nbatches, 0) + 
                " loss {:.6f}, accuracy {:3.3f} %, best {:3.3f} %".format(
                    gL_test, gA_test * 100, best_gA_test * 100
                )
            )
        # Uncomment the line below to print out the total time with overhead
        print("Total test time for this group: {}" \
        .format(time_wrap(use_gpu) - accum_test_time_begin))

        if (args.mlperf_logging
            and (args.mlperf_acc_threshold > 0)
            and (best_gA_test > args.mlperf_acc_threshold)):
            print("MLPerf testing accuracy threshold "
                    + str(args.mlperf_acc_threshold)
                    + " reached, stop training")
            # break

        if (args.mlperf_logging
            and (args.mlperf_auc_threshold > 0)
            and (best_auc_test > args.mlperf_auc_threshold)):
            print("MLPerf testing auc threshold "
                    + str(args.mlperf_auc_threshold)
                    + " reached, stop training")
            # break

            # k += 1  # nepochs
        return gA_test

    if args.nncf_config is not None:
        args.config = args.nncf_config
        config = create_sample_config(args, None)
        config.checkpoint_save_dir = config.log_dir
        configure_paths(config)
        copyfile(args.config, osp.join(config.log_dir, 'config.json'))
        source_root = Path(__file__).absolute().parents[0]  # nncf root
        create_code_snapshot(source_root, osp.join(config.log_dir, "snapshot.tar.gz"))
        config['episodic_nncfcfg'] = osp.join(config.log_dir, "episodic_nncfcfg")
        os.makedirs(config['episodic_nncfcfg'], exist_ok=True)
        configure_logging(logger, config)
        print_fn = logger.info

        class DLRMInitializingDataLoader(InitializingDataLoader):
            def get_inputs(self, dataloader_output):
                return (dataloader_output[0:3]), dict()

            def __len__(self):
                return len(self.data_loader)

        initializing_train_ld = DLRMInitializingDataLoader(train_ld)
        initializing_test_ld = DLRMInitializingDataLoader(test_ld)

        def autoq_eval_fn(eval_loader, model, criterion, config):
            return 100*test_dlrm(model, eval_loader)

        config.nncf_config = register_default_init_args(config.nncf_config, initializing_train_ld)

        compression_ctrl, dlrm = create_compressed_model(dlrm, config.nncf_config)

        # Instantiate AutoX/NNCF Quantization Environment
        env = QuantizationEnv(
                compression_ctrl,
                None,
                train_ld,
                test_ld,
                None,
                autoq_eval_fn,
                config,
                tie_quantizers=True,
                skip_wall=False)

        nb_state = len(env.state_list)
        nb_action = 1

        # Instantiate Automation Agent
        agent = DDPG(nb_state, nb_action, config)

        # Log the quantizer to be predicted by agent
        for i, qid in enumerate(env.master_df.index):
            pred_marker = '/' if env.master_df.is_pred[qid] else 'X'
            logger.info("[AutoX] Quantizer {:4} [{}]: {}".format(i, pred_marker, qid))

        best_policy, best_reward = search(agent, env, config)
        logger.info('best_reward: ', best_reward)
        logger.info('best_policy: ', best_policy)
        logger.info("[AutoX] Search Complete")
        
        end_ts = datetime.now()
        logger.info("Elapsed time of main_worker(): {}".format(end_ts-start_ts))

    # mostly irrelevant for mixed-precision search
    # profiling
    if args.enable_profiling:
        with open("dlrm_s_pytorch.prof", "w") as prof_f:
            prof_f.write(prof.key_averages().table(sort_by="cpu_time_total"))
            prof.export_chrome_trace("./dlrm_s_pytorch.json")
        # print(prof.key_averages().table(sort_by="cpu_time_total"))

    # plot compute graph
    if args.plot_compute_graph:
        sys.exit(
            "ERROR: Please install pytorchviz package in order to use the"
            + " visualization. Then, uncomment its import above as well as"
            + " three lines below and run the code again."
        )
        # V = Z.mean() if args.inference_only else E
        # dot = make_dot(V, params=dict(dlrm.named_parameters()))
        # dot.render('dlrm_s_pytorch_graph') # write .pdf file

    # test prints
    if not args.inference_only and args.debug_mode:
        print("updated parameters (weights and bias):")
        for param in dlrm.parameters():
            print(param.detach().cpu().numpy())

    # export the model in onnx
    if args.save_onnx:
        dlrm_pytorch_onnx_file = "dlrm_s_pytorch.onnx"
        batch_size = X_onnx.shape[0]
        # debug prints
        # print("batch_size", batch_size)
        # print("inputs", X_onnx, lS_o_onnx, lS_i_onnx)
        # print("output", dlrm_wrap(X_onnx, lS_o_onnx, lS_i_onnx, use_gpu, device))

        # force list conversion
        # if torch.is_tensor(lS_o_onnx):
        #    lS_o_onnx = [lS_o_onnx[j] for j in range(len(lS_o_onnx))]
        # if torch.is_tensor(lS_i_onnx):
        #    lS_i_onnx = [lS_i_onnx[j] for j in range(len(lS_i_onnx))]
        # force tensor conversion
        # if isinstance(lS_o_onnx, list):
        #     lS_o_onnx = torch.stack(lS_o_onnx)
        # if isinstance(lS_i_onnx, list):
        #     lS_i_onnx = torch.stack(lS_i_onnx)
        # debug prints
        print("X_onnx.shape", X_onnx.shape)
        if torch.is_tensor(lS_o_onnx):
            print("lS_o_onnx.shape", lS_o_onnx.shape)
        else:
            for oo in lS_o_onnx:
                print("oo.shape", oo.shape)
        if torch.is_tensor(lS_i_onnx):
            print("lS_i_onnx.shape", lS_i_onnx.shape)
        else:
            for ii in lS_i_onnx:
                print("ii.shape", ii.shape)

        # name inputs and outputs
        o_inputs = ["offsets"] if torch.is_tensor(lS_o_onnx) else ["offsets_"+str(i) for i in range(len(lS_o_onnx))]
        i_inputs = ["indices"] if torch.is_tensor(lS_i_onnx) else ["indices_"+str(i) for i in range(len(lS_i_onnx))]
        all_inputs = ["dense_x"] + o_inputs + i_inputs
        #debug prints
        print("inputs", all_inputs)

        # create dynamic_axis dictionaries
        do_inputs = [{'offsets': {1 : 'batch_size' }}] if torch.is_tensor(lS_o_onnx) else [{"offsets_"+str(i) :{0 : 'batch_size'}} for i in range(len(lS_o_onnx))]
        di_inputs = [{'indices': {1 : 'batch_size' }}] if torch.is_tensor(lS_i_onnx) else [{"indices_"+str(i) :{0 : 'batch_size'}} for i in range(len(lS_i_onnx))]
        dynamic_axes = {'dense_x' : {0 : 'batch_size'}, 'pred' : {0 : 'batch_size'}}
        for do in do_inputs:
            dynamic_axes.update(do)
        for di in di_inputs:
            dynamic_axes.update(di)
        # debug prints
        print(dynamic_axes)

        # export model
        torch.onnx.export(
            dlrm, (X_onnx, lS_o_onnx, lS_i_onnx), dlrm_pytorch_onnx_file, verbose=True, use_external_data_format=True, opset_version=11, input_names=all_inputs, output_names=["pred"], dynamic_axes=dynamic_axes
        )
        # recover the model back
        dlrm_pytorch_onnx = onnx.load(dlrm_pytorch_onnx_file)
        # check the onnx model
        onnx.checker.check_model(dlrm_pytorch_onnx)
        # '''
        # # run model using onnxruntime
        # import onnxruntime as rt

        # dict_inputs = {}
        # dict_inputs["dense_x"] = X_onnx.numpy().astype(np.float32)
        # if torch.is_tensor(lS_o_onnx):
        #     dict_inputs["offsets"] = lS_o_onnx.numpy().astype(np.int64)
        # else:
        #     for i in range(len(lS_o_onnx)):
        #         dict_inputs["offsets_"+str(i)] = lS_o_onnx[i].numpy().astype(np.int64)
        # if torch.is_tensor(lS_i_onnx):
        #     dict_inputs["indices"] = lS_i_onnx.numpy().astype(np.int64)
        # else:
        #     for i in range(len(lS_i_onnx)):
        #         dict_inputs["indices_"+str(i)] = lS_i_onnx[i].numpy().astype(np.int64)
        # print("dict_inputs", dict_inputs)

        # sess = rt.InferenceSession(dlrm_pytorch_onnx_file, rt.SessionOptions())
        # prediction = sess.run(output_names=["pred"], input_feed=dict_inputs)
        # print("prediction", prediction)
        # '''
        
