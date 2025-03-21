#!/usr/bin/env python
# coding: utf-8

"""
trainer.py: script to train the function.
------------------------------------------
Usage:
    python -m trainer wandb.logging=True wandb.model_name="TaleSumm-ICVT" split_id=[0,1,2,3,4]
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'
import wandb
import yaml
import copy
import torch
import dgl
import hydra
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch import optim
from tqdm import tqdm
from utils.logger import return_logger
from torch.utils.data import DataLoader
from typing import List, Tuple, Dict, Union
from omegaconf import DictConfig, OmegaConf, open_dict
# from thop import profile

from utils.metrics import getScores
from utils.model_config import get_model
from dataloader.multimodal_dataset import MultiModalDataset
from utils.general_utils import (ParseEPS, seed_everything, load_yaml, save_model)
from dataloader.multimodal_dataset import load_edge_dict, load_alignment_reference
from fvcore.nn import FlopCountAnalysis, parameter_count

__author__ = "rodosingh"
__copyright__ = "Copyright 2023, The Story-Summarization Project"
__credits__ = ["Aditya Singh", "Rodo Singh"]
__license__ = "GPL"
__version__ = "0.1"
__email__ = "aditya.si@research.iiit.ac.in"
__status__ = "Development"

logger = return_logger(__name__)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # pt is the probability of the correct class
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

class Trainer(object):
    """
    The trainer class to train the model and prepare data.
    """
    def __init__(self, cfg: DictConfig) -> None:
        r"""
        Train the model with the given specifications and methods and evaluate at the same
        time.
        -----------------------------------------------------------------------------------
        Args:
            - cfg: A dictionary that have extra parameters or args to pass on.
        """
        # Declare device
        if torch.cuda.is_available() and len(cfg['gpus'])>=1:
            self.device = torch.device(f"cuda:{cfg['gpus'][0]}" if torch.cuda.is_available() else 'cpu')
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device('cpu')

        # Import model
        model = get_model(cfg)

        # Set the weights for different series as well as their modality
        modality = cfg['modality']
        # print(f"{cfg['series']}'s Modality = {modality} is selected!\n")

        # Initialize BCE loss function with positive weights
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg[f'{cfg.series}_{modality}']]).to(self.device))
        # self.criterion = FocalLoss()

        # Scheduler and Optimizer
        if cfg['mode'] == 'training':
            # https://www.fast.ai/posts/2018-07-02-adam-weight-decay.html
            self.optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"], amsgrad=cfg["amsgrad"])
            total_steps = int(np.ceil(len(cfg['train'])/cfg['batch_size'])*cfg['epochs'])
            if cfg['lr_scheduler'] == 'onecycle':
                self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=10*cfg['lr'], total_steps=total_steps)
            elif cfg['lr_scheduler'] == 'cyclic':
                self.scheduler = optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=cfg['lr'], max_lr=10*cfg['lr'], step_size_up=total_steps//8, cycle_momentum=False, mode='triangular2')
            else:
                raise ValueError(f"Invalid lr_scheduler (={cfg['lr_scheduler']}).")
        else:
            self.optimizer = None
            self.scheduler = None

        # wandb section
        self.wandb_logging = cfg["wandb"]["logging"] and (cfg['mode'] == 'training')
        if self.wandb_logging and (not cfg["wandb"]["sweeps"]):
            wandb.init(project=cfg["wandb"]["project"], entity=cfg["wandb"]["entity"], config=OmegaConf.to_container(cfg, resolve=True), name=cfg["wandb"]["model_name"])
        if cfg['mode'] == 'training':
            # Whether to evaluate on Test set or not
            self.eval_test = cfg["eval_test"]
            self.mode = ["train", "val", "test"] if self.eval_test else ["train", "val"]
            # All the metrics to be logged
            self.metrics_name = ["AP", "F1"]

            # wandb run name
            self.name = wandb.run.name if cfg["wandb"]["sweeps"] else cfg["wandb"]["model_name"]
            # Save model and Early stopping
            self.model_save_path = cfg["ckpt_path"]
            self.save_best_model = cfg["ES"]["save_best_model"]
            self.early_stopping = cfg["ES"]["early_stopping"]
            self.best_val_AP = float('-inf')
            self.best_test_vid_AP = float('-inf')
            self.best_test_dia_AP = float('-inf')
            self.best_test_all_AP = float('-inf')
            self.best_test_vid_F1 = float('-inf')
            self.best_test_dia_F1 = float('-inf')
            if self.early_stopping or self.save_best_model:
                self.best_val_loss = float('inf')
                if modality == "both":
                    self.best_vid_val_AP = float('-inf')
                    self.best_dia_val_AP = float('-inf')
                self.ctr, self.es = 0, 0
            if self.save_best_model:
                self.model_save_path = os.path.join(self.model_save_path, self.name)
                os.makedirs(self.model_save_path, exist_ok=True)
                with open(f"{self.model_save_path}/{self.name}_config.yaml", "w") as f:
                    f.write(OmegaConf.to_yaml(ParseEPS.convert2Yamlable(copy.deepcopy(cfg)), resolve=True))
                # save_yaml(f"{self.model_save_path}/{self.name}_config.yaml", ParseEPS.convert2Yamlable(cfg.copy()))
                # logger.info(f"Saved config at {self.model_save_path}{self.name}_config.yaml")

        # model section
        self.model = model.to(self.device)
        if (len(cfg["gpus"])>1 and cfg['mode'] == 'training') or \
            (cfg['mode'] == 'inference' and len(cfg["gpus"])>1):
            self.model = nn.DataParallel(self.model, device_ids=cfg["gpus"])

        # other section
        self.cfg = cfg
        self.modality = modality
        self.epochs = cfg["epochs"]
        self.triplet_loss = nn.TripletMarginLoss(margin = 0.05)
        self.inter_weight = 0.00001
        self.intra_weight = 0.00001
        self.model_calc_inter_loss = calc_inter_class()
        self.model_calc_intra_loss = calc_intra_class()
        self.model_calc_loss = calc_loss_class(cfg)
        

    def prepare_data(self, mode:str) -> DataLoader:
        """
        Prepare train and validation (and test too) data loader.
        ------------------------------------------
        Args:
            - mode (str): Whether train, validation, or test data loader. Options: ["train", "val", "test"]

        Returns:
            - dl (Dataloader): A pytorch dataloader object.
        """
        sampling_type = self.cfg['sampling_type']
        if sampling_type == "random" and mode in ["val", "test"]:
            sampling_type = "uniform"

        common_params = {'vary_window_size': self.cfg['vary_window_size'],
                         'scene_boundary_threshold': self.cfg['scene_boundary_threshold'],
                         'window_size': self.cfg['window_size'],
                         'bin_size': self.cfg['bin_size'],
                         'withGROUP': self.cfg['withGROUP'],
                         'normalize_group_labels': self.cfg['normalize_group_labels'],
                         'which_features': self.cfg['which_features'],
                         'modality': self.modality,
                         'vid_label_type': self.cfg['vid_label_type'],
                         'dia_label_type': self.cfg['dia_label_type'],
                         'which_dia_model': self.cfg['which_dia_model'],
                         'get_word_level': self.cfg['enable_dia_encoder'],
                         'max_cap': self.cfg['max_cap'],
                         'concatenation': self.cfg['concatenation']}
        dataset = MultiModalDataset(ep_names=self.cfg[mode],
                                    sampling_type=sampling_type,
                                    **common_params)
        dl = DataLoader(dataset, batch_size=self.cfg['batch_size'], shuffle=False,
                        collate_fn=dataset.collate_fn, num_workers=self.cfg['num-workers'])
        # logger.info(f"{mode.upper()} data loader prepared with {len(dataset)} samples.")
        return dl

    def scoreDict(self, scores: List[np.ndarray],
                  mode: str,
                  combine: bool,
                  prefixes:List[str],
                  suffixes: List[str])->Dict:
        r"""
        Return a dictionary of scores.
        ----------------------------------
        Args:
            - scores (List[np.ndarray]): List of scores.
            - mode (str): Whether 'train', 'val', or 'test'.
            - combine (bool): Whether to combine scores or not.
            - prefixes (List[str]): List of prefixes. Usually, ['vid_', 'dia_']
            - suffixes (List[str]): List of suffixes. Usually, ['AP', 'F1', 'F1_T']
        """
        if combine:
            assert len(prefixes) == 2, "Only two prefixes are allowed for combining scores."
            return {f"{mode}_{suffix}": np.sqrt(scores[prefixes[0][:-1]][i] * scores[prefixes[1][:-1]][i]) for i, suffix in enumerate(suffixes)}
        else:
            if any([len(prefix) == 0 for prefix in prefixes]):
                return {f"{mode}_{suffix}": scores[i] for i, suffix in enumerate(suffixes)}
            else:
                return {f"{prefix}{mode}_{suffix}": scores[prefix[:-1]][i] for prefix in prefixes for i, suffix in enumerate(suffixes)}

    def calc_loss(self, yhat: torch.Tensor, yhat_mask: torch.Tensor,
                  targets: torch.Tensor, target_mask: torch.Tensor
                 )->Tuple[torch.Tensor, List[np.ndarray], List[np.ndarray]]:
        r"""
        Calculate loss for the given yhat and targets.
        ------------------------------------------------
        Args:
            - yhat (torch.Tensor): Predictions from the model.
            - yhat_mask (torch.Tensor): Mask invalid tokens in predictions.
            - targets (torch.Tensor): Ground truth.
            - target_mask (torch.Tensor): Mask invalid tokens in ground truth.
        
        Returns:
            - loss (torch.Tensor): Loss for the given yhat and targets.
            - yhat_lst (List[np.ndarray]): List of predictions.
            - target_lst (List[np.ndarray]): List of ground truth.
        """
        B, _ = yhat.shape
        loss = 0
        yhat_lst, target_lst = [], []
        for i in range(B):
            loss += self.criterion(yhat[i][yhat_mask[i]], targets[i][target_mask[i]])
            yhat_lst.append(torch.sigmoid(yhat[i][yhat_mask[i]]).detach().cpu().numpy())
            target_lst.append(targets[i][target_mask[i]].detach().cpu().numpy())
        return loss/B, yhat_lst, target_lst

    def calc_inter_loss(self, mod, video_name, yhat_lst, target_lst, h, boolean_mask, alignment_ref):
        inter_loss_sum = torch.tensor(0.0).to(h.device)
        B = len(video_name)
        for i in range(B):
            yhat = torch.tensor(yhat_lst[i], device = h.device)
            L = sum(target_lst[i] != 0)
            ht = h[i][boolean_mask[i]]
            refer = torch.tensor(alignment_ref[video_name[i] + '_time_idx.npy'], device = h.device)
            if mod == 'vid':
                ref = refer[:yhat_lst[i].shape[0]]
            else:
                ref = refer[-yhat_lst[i].shape[0]:]

            target = torch.tensor(target_lst[i], device = h.device, dtype= torch.bool)
            x = ht[target]
            xm = torch.mean(x, dim = 0).unsqueeze(0)

            if L > yhat.shape[0]:
                L = yhat.shape[0]

            topk_values, topk_indices = yhat.topk(L)
            inter_ref = ref[topk_indices]
            non_negative_indices = torch.nonzero(inter_ref != -1, as_tuple=False).squeeze(1)
            inter_ref = inter_ref[non_negative_indices]
            xp = ht[inter_ref]
            xpm = torch.mean(xp, dim = 0).unsqueeze(0)

            neg_position = torch.randperm(target.shape[0])[:L].to(h.device)
            xn = ht[neg_position]
            xnm = torch.mean(xn, dim = 0).unsqueeze(0)
            
            inter_loss = self.triplet_loss(xm, xpm, xnm)
            if(inter_loss.isnan()): inter_loss = torch.tensor(0.0).to(h.device)
            inter_loss_sum += inter_loss

        return self.inter_weight * inter_loss_sum / B


    def calc_intra_loss(self, mod, video_name, yhat_lst, target_lst, h, boolean_mask, alignment_ref):
        intra_loss_sum = torch.tensor(0.0).to(h.device)
        B = len(video_name)
        for i in range(B):
            yhat = torch.tensor(yhat_lst[i], device = h.device)
            L = sum(target_lst[i] != 0)
            ht = h[i][boolean_mask[i]]
            refer = torch.tensor(alignment_ref[video_name[i] + '_time_idx.npy'], device = h.device)
            if mod == 'vid':
                ref = refer[-yhat_lst[i].shape[0]:]
            else:
                ref = refer[:yhat_lst[i].shape[0]]

            topk_values, topk_indices = yhat.topk(L)
            inter_ref = ref[topk_indices]
            non_negative_indices = torch.nonzero(inter_ref != -1, as_tuple=False).squeeze(1)
            topk_indices = topk_indices[non_negative_indices]
            inter_ref = inter_ref[non_negative_indices]

            nf = ht[topk_indices]
            pf = ht[inter_ref]
            N = topk_indices.shape[0]

            x = torch.zeros(0, ht.shape[1], device=h.device)
            xp = torch.zeros(0, ht.shape[1], device=h.device)
            xn = torch.zeros(0, ht.shape[1], device=h.device)
            for i in range(nf.shape[0]):
                xt = nf[i].unsqueeze(0).expand(N - 1, -1)
                xpt = pf[i].unsqueeze(0).expand(N - 1, -1)
                xnt = torch.cat((nf[:i], nf[i + 1:]), dim=0)
                x = torch.cat((x, xt), dim=0)
                xp = torch.cat((xp, xpt), dim=0)
                xn = torch.cat((xn, xnt), dim=0)
            
            intra_loss = self.triplet_loss(x, xp, xn)
            if(intra_loss.isnan()): intra_loss = torch.tensor(0.0).to(h.device)
            intra_loss_sum += intra_loss

        return self.intra_weight * intra_loss_sum / B



    def transformANDforward(self, data_batch: Dict, edge_dict, alignment_ref = None, epoch = 0)->Tuple:
        # transform data batch for video and dialogue modality to device
        if self.cfg['withGROUP']:
            feat_dict, video_name, bin_indices, token_type, mask, group_idx, subgroup_len, labels = data_batch
            # convert labels to 1D tensor
            labels = labels.to(self.device)
        else:
            feat_dict, bin_indices, token_type, mask, group_idx, subgroup_len = data_batch

        # Convert everything to device
        bin_indices = bin_indices.to(self.device)
        mask = mask.to(self.device)
        token_type = token_type.to(self.device)
        group_idx = group_idx.to(self.device)
        subgroup_len = subgroup_len.to(self.device)
        if self.modality == "both":
            vid_feat_dict, dia_feat_dict = feat_dict
        elif self.modality == "vid":
            vid_feat_dict = feat_dict
            dia_feat_dict = None
        elif self.modality == "dia":
            dia_feat_dict = feat_dict
            vid_feat_dict = None
        else:
            raise ValueError(f"Invalid modality (={self.modality}).")
        
        if self.modality != "dia":
            vid_feat_dict = {k: v.to(torch.float32).to(self.device) for k, v in vid_feat_dict.items()}
            # extract video ground truth
            if self.cfg['concatenation']:
                vid_boolean_mask = (vid_feat_dict['vid_mask'].sum(dim = -1)>0)
            else:
                if len(self.cfg['which_features']) == 1 and \
                    'mvit' in self.cfg['which_features']:
                    IC_feat = 'mvit'
                else:
                    IC_feat = 'imagenet' if 'imagenet' in self.cfg['which_features'] else 'clip'
                vid_boolean_mask = (vid_feat_dict[f'{IC_feat}_mask'].sum(dim = -1)>0)
            vid_targets = vid_feat_dict['labels']
        if self.modality != "vid":
            dia_feat_dict = {k: v.to(torch.float32).to(self.device) for k, v in dia_feat_dict.items()}
            dia_targets = dia_feat_dict['labels']
            # extract dialogue ground truth
            if self.cfg['enable_dia_encoder']:
                dia_boolean_mask = (dia_feat_dict['word_mask'].sum(dim=-1)>0)
            else:
                dia_boolean_mask = (dia_feat_dict['dia_mask']>0)

        # forward pass
        if self.cfg['ours_model']:


            video_yhat, dia_yhat, hd, hv, hetero_graphs = self.model(vid_feat_dict, dia_feat_dict, video_name, bin_indices,
                                                    token_type, group_idx, mask, subgroup_len, edge_dict,
                                                    dia_targets, dia_boolean_mask, vid_targets, vid_boolean_mask)        


        else:
            yhat = self.model(vid_feat_dict['vid_enc'], vid_feat_dict['vid_mask'])


        if self.modality != "dia":

            if self.cfg['enable_decoder']:

                


                vid_loss, vid_yhat_lst, vid_target_lst = \
                    self.calc_loss(video_yhat, vid_boolean_mask, vid_targets, vid_boolean_mask)
            else:
                vid_loss, vid_yhat_lst, vid_target_lst = \
                    self.calc_loss(yhat, vid_boolean_mask, vid_targets, vid_boolean_mask)



        loss = vid_loss + dia_loss

        if alignment_ref != None:



            vid_inter_loss = self.calc_inter_loss('vid', video_name, dia_yhat_lst, vid_target_lst, hv, vid_boolean_mask, alignment_ref['inter'])
            dia_inter_loss = self.calc_inter_loss('dia', video_name, vid_yhat_lst, dia_target_lst, hd, dia_boolean_mask, alignment_ref['inter'])
            vid_intra_loss = self.calc_intra_loss('vid', video_name, vid_yhat_lst, vid_target_lst, hv, vid_boolean_mask, alignment_ref['intra'])
            dia_intra_loss = self.calc_intra_loss('dia', video_name, dia_yhat_lst, dia_target_lst, hd, dia_boolean_mask, alignment_ref['intra'])
            loss += vid_inter_loss + dia_inter_loss + vid_intra_loss + dia_intra_loss

        # return loss
        if self.modality == "both":
            return loss, vid_yhat_lst, vid_target_lst, dia_yhat_lst, dia_target_lst

        elif self.modality == "vid":
            return vid_loss, vid_yhat_lst, vid_target_lst
        
        elif self.modality == "dia":
            return dia_loss, dia_yhat_lst, dia_target_lst

    def evaluate(self, val_dl: DataLoader, edge_dict, epoch = 0) -> Tuple[float, Union[List[float], Dict[str, List[float]]]]:
        """
        Same as train function, but only difference is that model is freezed
        and no parameters update happen and hence no gradient updates.
        ----------------------------------------------------------------------
        Args:
            - val_dl (DataLoader): Validation data loader.
        """
        self.model.eval()
        eval_loss = 0
        if self.modality == 'both':
            vid_y_true, dia_y_true, vid_y_pred, dia_y_pred = [], [], [], []
        else:
            y_true_epoch, y_pred_epoch = [], []
        with torch.no_grad():
            for _, data_batch in enumerate(tqdm(val_dl, disable=self.cfg["wandb"]["logging"])):
                if self.modality == 'both':
                    loss, vid_yhat, vid_targets, dia_yhat, dia_targets = \
                        self.transformANDforward(data_batch, edge_dict, None, epoch)
                    vid_y_pred.extend(vid_yhat)
                    vid_y_true.extend(vid_targets)
                    dia_y_pred.extend(dia_yhat)
                    dia_y_true.extend(dia_targets)
                else:
                    loss, yhat, targets = self.transformANDforward(data_batch)
                    y_pred_epoch.extend(yhat)
                    y_true_epoch.extend(targets)
                eval_loss += loss.item()
        if self.modality == 'both':
            scores = {'vid': [*getScores(vid_y_true, vid_y_pred)],
                      'dia': [*getScores(dia_y_true, dia_y_pred)]}
        else:
            scores = [*getScores(y_true_epoch, y_pred_epoch)]
        return eval_loss/len(val_dl), scores

    def train(self)->None:
        r"""
        Train the model here.
        """
        # create data
        train_dl = self.prepare_data(mode="train")
        val_dl = self.prepare_data(mode="val")
        # training starts
        edge_dict = load_edge_dict()
        alignment_ref = load_alignment_reference()

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0
            logger.info(f"EPOCH: {epoch+1}/{self.epochs}")
            if self.modality == 'both':
                vid_y_true, vid_y_pred, dia_y_true, dia_y_pred = [], [], [], []
            else:
                y_true_epoch, y_pred_epoch = [], []
            for _, data_batch in enumerate(tqdm(train_dl, disable=self.wandb_logging)):
                self.optimizer.zero_grad()
                if self.modality == 'both':
                    loss, vid_yhat, vid_targets, dia_yhat, dia_targets = \
                        self.transformANDforward(data_batch, edge_dict, alignment_ref, epoch)
                    vid_y_pred.extend(vid_yhat)
                    vid_y_true.extend(vid_targets)
                    dia_y_pred.extend(dia_yhat)
                    dia_y_true.extend(dia_targets)
                    
                else:
                    loss, yhat, targets = self.transformANDforward(data_batch)
                    y_pred_epoch.extend(yhat)
                    y_true_epoch.extend(targets)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                self.scheduler.step() # CyclicLR: called on every batch
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()
                # break  ############ 

            # logger.info(f'Average Graph Density:{res_weight}')

            if self.modality == 'both':
                train_scores = {'vid': [*getScores(vid_y_true, vid_y_pred)],
                                'dia': [*getScores(dia_y_true, dia_y_pred)]}
            else:
                train_scores = [*getScores(y_true_epoch, y_pred_epoch)]
            val_loss, val_scores = self.evaluate(val_dl, edge_dict, epoch)
            # self.scheduler.step(val_loss) # called on every epoch (ReduceLROnPlateau)
            epoch_loss = epoch_loss/len(train_dl)
            # logger.info(f"TRAIN: loss = {epoch_loss} | VAL: loss = {val_loss}\n")
            if self.eval_test:
                test_dl = self.prepare_data(mode="test")
                test_loss, test_scores = self.evaluate(test_dl, edge_dict, epoch)
                # logger.info(f"TEST: loss = {test_loss}\n")

            # ======================= LOG Best Metrics and REST =======================
            if self.modality == 'both':
                val_AP = np.sqrt(val_scores['vid'][0]*val_scores['dia'][0])
            else:
                val_AP = val_scores[0]
            if self.wandb_logging:
                best_happened = True if val_AP > self.best_val_AP else False
                if self.modality == 'both':
                    train_scores_dict = self.scoreDict(train_scores, "train", True, ["vid_", "dia_"], self.metrics_name)

                else:
                    train_scores_dict = self.scoreDict(train_scores, "train", False, [''], self.metrics_name)

            else:
                if self.modality == "both":
                    logger.info(f"TRAIN: Vid_AP = {train_scores['vid'][0]:.3f} | Vid_F1 = {train_scores['vid'][1]:.3f} | Dia_AP = {train_scores['dia'][0]:.3f} | Dia_F1 = {train_scores['dia'][1]:.3f}")
                    logger.info(f"VAL:   Vid_AP = {val_scores['vid'][0]:.3f} | Vid_F1 = {val_scores['vid'][1]:.3f} | Dia_AP = {val_scores['dia'][0]:.3f} | Dia_F1 = {val_scores['dia'][1]:.3f}")
                    if self.eval_test:
                        vid_AP = test_scores['vid'][0]
                        vid_F1 = test_scores['vid'][1]
                        dia_AP = test_scores['dia'][0]
                        dia_F1 = test_scores['dia'][1]
                        logger.info(f"TEST:  Vid_AP = {vid_AP:.3f} | Vid_F1 = {vid_F1:.3f} | Dia_AP = {dia_AP:.3f} | Dia_F1 = {dia_F1:.3f}")
                        
                        if vid_AP > self.best_test_vid_AP:
                            self.best_test_vid_AP = vid_AP
                        if dia_AP > self.best_test_dia_AP:
                            self.best_test_dia_AP = dia_AP
                        if (vid_AP + dia_AP) / 2 > self.best_test_all_AP:
                            self.best_test_all_AP = (vid_AP + dia_AP) / 2
                        if vid_F1 > self.best_test_vid_F1:
                            self.best_test_vid_F1 = vid_F1
                        if dia_F1 > self.best_test_dia_F1:
                            self.best_test_dia_F1 = dia_F1

                else:
                    # logger.info(f"TRAIN: AP = {train_scores[0]:.3f} | F1 = {train_scores[1]:.3f}")
                    # logger.info(f"VAL:   AP = {val_scores[0]:.3f} | F1 = {val_scores[1]:.3f}")
                    if self.eval_test:
                        logger.info(f"TEST:  AP = {test_scores[0]:.3f} | F1 = {test_scores[1]:.3f}")

        logger.info(f"best test vid AP: {self.best_test_vid_AP}")
        logger.info(f"best test dia AP: {self.best_test_dia_AP}")
        logger.info(f"best test all AP: {self.best_test_all_AP}")
        logger.info(f"best test vid F1: {self.best_test_vid_F1}")
        logger.info(f"best test dia F1: {self.best_test_dia_F1}")
        
        logger.info("TRAINING ENDS !!!\n\n")


def main(config: DictConfig):

    # seed everything
    seed_everything(config['seed'], harsh=True)
    # =================================== TRAINER CONFIG ===================================
    if isinstance(config['hidden_sizes'], str) and config['hidden_sizes'] == 'd_model':
        config['hidden_sizes'] = [config['d_model']]
    # ======================== parse the episode config ========================
    split_type_path = os.path.join(config["split_dir"], config["split_type"])
    # logger.info(f"Split type: {config['split_type']} at {split_type_path}")
    if os.path.isfile(split_type_path):    
        episode_config = load_yaml(split_type_path)
        series_lst = ['24', 'prison-break'] if config['series'] == 'all' else config['series']
        split_dict = ParseEPS(episode_config, series=series_lst).dct
        with open_dict(config):
            config.update(split_dict)

    # See the config set...
    # print(OmegaConf.to_yaml(config, resolve=True))

    if not os.path.isfile(split_type_path):
        if config['wandb']['sweeps']:
            orig_name = wandb.run.name
        else:
            orig_name = config['wandb']['model_name']
        for idx in config['split_id']:
            # logger.info(f"Split {idx+1} out of {len(config['split_id'])}")
            eps_config = load_yaml(os.path.join(split_type_path, f"split{idx+1}.yaml"))
            split_dict = ParseEPS(eps_config, series=config['series']).dct
            with open_dict(config):
                config.update(split_dict)
            # logger.info(f"Train Samples: {len(config['train'])} | Val Samples: {len(config['val'])}")
            if config['eval_test']:
                x = 1
                # logger.info(f"Test Samples: {len(config['test'])}")
            if config['wandb']['sweeps']:
                wandb.run.name = orig_name + f"|S{idx+1}"
            else:
                config['wandb']['model_name'] = orig_name + f"|S{idx+1}"
            trainer = Trainer(config)
            trainer.train()
            del trainer
    else:
        trainer = Trainer(config)
        trainer.train()
        del trainer



class calc_loss_class(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        modality = cfg['modality']
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg[f'{cfg.series}_{modality}']]).to('cuda'))


    def forward(self, yhat: torch.Tensor, yhat_mask: torch.Tensor,
                  targets: torch.Tensor, target_mask: torch.Tensor
                 )->Tuple[torch.Tensor, List[np.ndarray], List[np.ndarray]]:
        r"""
        Calculate loss for the given yhat and targets.
        ------------------------------------------------
        Args:
            - yhat (torch.Tensor): Predictions from the model.
            - yhat_mask (torch.Tensor): Mask invalid tokens in predictions.
            - targets (torch.Tensor): Ground truth.
            - target_mask (torch.Tensor): Mask invalid tokens in ground truth.
        
        Returns:
            - loss (torch.Tensor): Loss for the given yhat and targets.
            - yhat_lst (List[np.ndarray]): List of predictions.
            - target_lst (List[np.ndarray]): List of ground truth.
        """
        B, _ = yhat.shape
        loss = 0
        for i in range(B):
            loss += self.criterion(yhat[i][yhat_mask[i]], targets[i][target_mask[i]])
        return loss



  
                        
class calc_inter_class(nn.Module):
    def __init__(self):
        super().__init__()
        self.inter_weight = 0
        self.triplet_loss = nn.TripletMarginLoss(margin = 0.05)


    def forward(self, mod, video_name, yhat_lst, target_lst, h, boolean_mask, alignment_ref):
        inter_loss_sum = torch.tensor(0.0).to(h.device)
        B = len(video_name)
        for i in range(B):
            yhat = torch.tensor(yhat_lst[i], device = h.device)
            L = sum(target_lst[i] != 0)
            ht = h[i][boolean_mask[i]]
            refer = torch.tensor(alignment_ref[video_name[i] + '_time_idx.npy'], device = h.device)
            if mod == 'vid':
                ref = refer[:yhat_lst[i].shape[0]]
            else:
                ref = refer[-yhat_lst[i].shape[0]:]

            target = torch.tensor(target_lst[i], device = h.device, dtype= torch.bool)
            x = ht[target]
            xm = torch.mean(x, dim = 0).unsqueeze(0)

            if L > yhat.shape[0]:
                L = yhat.shape[0]

            topk_values, topk_indices = yhat.topk(L)
            inter_ref = ref[topk_indices]
            non_negative_indices = torch.nonzero(inter_ref != -1, as_tuple=False).squeeze(1)
            inter_ref = inter_ref[non_negative_indices]
            xp = ht[inter_ref]
            xpm = torch.mean(xp, dim = 0).unsqueeze(0)

            neg_position = torch.randperm(target.shape[0])[:L].to(h.device)
            xn = ht[neg_position]
            xnm = torch.mean(xn, dim = 0).unsqueeze(0)
            
            inter_loss = self.triplet_loss(xm, xpm, xnm)
            if(inter_loss.isnan()): inter_loss = torch.tensor(0.0).to(h.device)
            inter_loss_sum += inter_loss

        return self.inter_weight * inter_loss_sum / B


                                       
                        
                        
class calc_intra_class(nn.Module):
    def __init__(self):
        super().__init__()
        self.intra_weight = 1
        self.triplet_loss = nn.TripletMarginLoss(margin = 0.05)


    def forward(self, mod, video_name, yhat_lst, target_lst, h, boolean_mask, alignment_ref):
        intra_loss_sum = torch.tensor(0.0).to(h.device)
        B = len(video_name)
        for i in range(B):
            yhat = torch.tensor(yhat_lst[i], device = h.device)
            L = sum(target_lst[i] != 0)
            ht = h[i][boolean_mask[i]]
            refer = torch.tensor(alignment_ref[video_name[i] + '_time_idx.npy'], device = h.device)
            if mod == 'vid':
                ref = refer[-yhat_lst[i].shape[0]:]
            else:
                ref = refer[:yhat_lst[i].shape[0]]

            topk_values, topk_indices = yhat.topk(L)
            inter_ref = ref[topk_indices]
            non_negative_indices = torch.nonzero(inter_ref != -1, as_tuple=False).squeeze(1)
            topk_indices = topk_indices[non_negative_indices]
            inter_ref = inter_ref[non_negative_indices]

            nf = ht[topk_indices]
            pf = ht[inter_ref]
            N = topk_indices.shape[0]

            x = torch.zeros(0, ht.shape[1], device=h.device)
            xp = torch.zeros(0, ht.shape[1], device=h.device)
            xn = torch.zeros(0, ht.shape[1], device=h.device)
            for i in range(nf.shape[0]):
                xt = nf[i].unsqueeze(0).expand(N - 1, -1)
                xpt = pf[i].unsqueeze(0).expand(N - 1, -1)
                xnt = torch.cat((nf[:i], nf[i + 1:]), dim=0)
                x = torch.cat((x, xt), dim=0)
                xp = torch.cat((xp, xpt), dim=0)
                xn = torch.cat((xn, xnt), dim=0)
            
            intra_loss = self.triplet_loss(x, xp, xn)
            if(intra_loss.isnan()): intra_loss = torch.tensor(0.0).to(h.device)
            intra_loss_sum += intra_loss

        return self.intra_weight * intra_loss_sum / B









def sweep_agent_manager():
    r"""
    Sweep agent manager to run the sweep.
    """
    wandb.init()
    config = wandb.config
    wd = config['weight_decay']
    ams = config['amsgrad']
    lr = config['lr']
    lrs = config['lr_scheduler']
    feat_fusion_style = config['feat_fusion_style']
    epochs = config['epochs']
    wandb.run.name = (f"SWP26|E{epochs}|WD{wd}|AMS{ams}|LR{lr}|LRS{lrs}|{feat_fusion_style}")
    main(dict(config))

@hydra.main(config_path="./configs", config_name="trainer_config", version_base='1.3')
def driver(cfg: DictConfig):
    if cfg.wandb.sweeps:
        wandb.agent(sweep_id=cfg["wandb"]["sweep_id"],
                    function=sweep_agent_manager,
                    count=cfg["wandb"]["sweep_agent_run_count"])
    else:
        main(cfg)

if __name__ == '__main__':
    driver()
