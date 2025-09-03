import json
import math
import os
from statistics import mean, stdev
import torch
from sklearn.metrics import mean_squared_error, balanced_accuracy_score
from torch import nn
from torch.nn import functional as F
import utils
from utils import prediction2label
from scipy.stats import kendalltau
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


class ordinal_loss(nn.Module):
    """Ordinal regression with encoding as in https://arxiv.org/pdf/0704.1028.pdf"""

    def __init__(self, weight_class=False):
        super(ordinal_loss, self).__init__()
        self.weights = weight_class

    def forward(self, predictions, targets):
        modified_target = torch.zeros_like(predictions)
        for i, target in enumerate(targets):
            modified_target[i, 0:target + 1] = 1

        # if torch tensor is empty, return 0
        if predictions.shape[0] == 0:
            return 0
        # loss
        if self.weights is not None:

            return torch.sum((self.weights * F.mse_loss(predictions, modified_target, reduction="none")).mean(axis=1))
        else:
            return torch.sum((F.mse_loss(predictions, modified_target, reduction="none")).mean(axis=1))


class ContextAttention(nn.Module):
    def __init__(self, size, num_head):
        super(ContextAttention, self).__init__()
        self.attention_net = nn.Linear(size, size)
        self.num_head = num_head

        if size % num_head != 0:
            raise ValueError("size must be dividable by num_head", size, num_head)
        self.head_size = int(size / num_head)
        self.context_vector = torch.nn.Parameter(torch.Tensor(num_head, self.head_size, 1))
        nn.init.uniform_(self.context_vector, a=-1, b=1)

    def get_attention(self, x):
        attention = self.attention_net(x)
        attention_tanh = torch.tanh(attention)
        attention_split = torch.stack(attention_tanh.split(split_size=self.head_size, dim=2), dim=0)
        similarity = torch.bmm(attention_split.view(self.num_head, -1, self.head_size), self.context_vector)
        similarity = similarity.view(self.num_head, x.shape[0], -1).permute(1, 2, 0)
        return similarity

    def forward(self, x):
        attention = self.attention_net(x)
        attention_tanh = torch.tanh(attention)
        if self.head_size != 1:
            attention_split = torch.stack(attention_tanh.split(split_size=self.head_size, dim=2), dim=0)
            similarity = torch.bmm(attention_split.view(self.num_head, -1, self.head_size), self.context_vector)
            similarity = similarity.view(self.num_head, x.shape[0], -1).permute(1, 2, 0)
            similarity[x.sum(-1) == 0] = -1e4  # mask out zero padded_ones
            softmax_weight = torch.softmax(similarity, dim=1)

            x_split = torch.stack(x.split(split_size=self.head_size, dim=2), dim=2)
            weighted_x = x_split * softmax_weight.unsqueeze(-1).repeat(1, 1, 1, x_split.shape[-1])
            attention = weighted_x.view(x_split.shape[0], x_split.shape[1], x.shape[-1])
        else:
            softmax_weight = torch.softmax(attention, dim=1)
            attention = softmax_weight * x

        sum_attention = torch.sum(attention, dim=1)
        return sum_attention


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  
        out = self.relu(out)
        return out


def get_conv_layer(rep_name):
    if "pianoroll" in rep_name:
        in_channels = 2
        kernel_width = (3, 4, 4)  # 88
    elif "mel" in rep_name:
        in_channels = 1
        kernel_width = (3, 4, 4)  # 64
    elif "cqt" in rep_name:
        in_channels = 1
        kernel_width = (3, 4, 4)  # 88
    else:
        raise ValueError("Representation not implemented")

    if "5" in rep_name:
        kernel_height = (3, 4, 4)
    elif "10" in rep_name:
        kernel_height = (4, 5, 5)
    elif "20" in rep_name:
        kernel_height = (4, 6, 6)
    else:
        raise ValueError("Representation not implemented")

    convs = nn.Sequential(
        ResidualBlock(in_channels, 64, 3, 1, 1),
        nn.MaxPool2d((kernel_height[0], kernel_width[0])),  # Adjusted pooling to handle increased length
        nn.Dropout(0.1),
        ResidualBlock(64, 128, 3, 1, 1),
        nn.MaxPool2d((kernel_height[1], kernel_width[1])),  # Adjusted pooling
        nn.Dropout(0.1),
        ResidualBlock(128, 256, 3, 1, 1),
        nn.MaxPool2d((kernel_height[2], kernel_width[2])),  # Adjusted pooling
        nn.Dropout(0.1)
    )
    return convs


class multimodal_cnns(nn.Module):

    def __init__(self, modality_dropout, only_cqt=False, only_pr=False):
        super().__init__()

        self.midi_branch = get_conv_layer("pianoroll5")
        self.audio_branch = get_conv_layer("cqt5")
        self.modality_dropout = modality_dropout
        self.only_cqt = only_cqt
        self.only_pr = only_pr

    def forward(self, x):
        x_midi, x_audio = x
        x_midi = self.midi_branch(x_midi).squeeze(-1)
        x_audio = self.audio_branch(x_audio).squeeze(-1)
        # do a modality dropout
        if self.only_cqt:
            x_midi = torch.zeros_like(x_midi, device=x_midi.device)
        elif self.only_pr:
            x_audio = torch.zeros_like(x_audio, device=x_audio.device)
        x_midi_trimmed = x_midi[:, :, :x_audio.size(2)]

        cnns_out = torch.cat((x_midi_trimmed, x_audio), 1)
        return cnns_out


class AudioModel(nn.Module):
    def __init__(self, num_classes, rep, modality_dropout, only_cqt=False, only_pr=False):
        super(AudioModel, self).__init__()

        # All Convolutional Layers in a Sequential Block
        if "pianoroll" in rep:
            conv = get_conv_layer(rep)
        elif "cqt" in rep:
            conv = get_conv_layer(rep)
        elif "mel" in rep:
            conv = get_conv_layer(rep)
        elif "multi" in rep:
            conv = multimodal_cnns(modality_dropout, only_cqt, only_pr)
        self.conv_layers = conv

        # Calculate the size of GRU input feature
        self.gru_input_size = 512 if "multi" in rep else 256

        # GRU Layer
        self.gru = nn.GRU(input_size=self.gru_input_size, hidden_size=128, num_layers=2,
                          batch_first=True, bidirectional=True)

        self.context_attention = ContextAttention(size=256, num_head=4)
        self.non_linearity = nn.ReLU()

        # Fully connected layer
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x1, kk):
        # Applying Convolutional Block
        x = self.conv_layers(x1)
        # Reshape for GRU input
        x = x.squeeze().transpose(0, 1).unsqueeze(0)
        x, _ = self.gru(x)
        x = self.context_attention(x)
        # classiffier
        x = self.non_linearity(x)
        x = self.fc(x)
        return x
        
    def extract_dual_embeddings(self, x1):
        with torch.no_grad():
            x = self.conv_layers(x1) 

            if x.size(-2) == 1:       
                x = x.squeeze(-2).transpose(1, 2)   
            elif x.size(-1) == 1:   
                x = x.squeeze(-1).transpose(1, 2)   
            else:
        
                if x.size(-2) <= x.size(-1):
                    x = x.mean(dim=-2)             
                    x = x.transpose(1, 2)        
                else:
                    x = x.mean(dim=-1)            
                    x = x.transpose(1, 2)          
           
            x, _ = self.gru(x)

            #Pre-attention embedding (mean+std over time)
            x_np = x.squeeze(0).cpu().numpy()    
            mean_emb = x_np.mean(axis=0)
            std_emb  = x_np.std(axis=0)
            before = np.concatenate([mean_emb, std_emb])

            #Post-attention
            after = self.context_attention(x).squeeze(0).cpu().numpy()
            return before, after

    def extract_dual_embeddings_multi(self, x_pr, x_cqt):
        """
        x_pr:  [B, 2, 88, T]  (onsets, frames)
        x_cqt: [B, 1, 88, T]
        returns: before (512,), after (256,)
        """
        with torch.no_grad():
            x = self.conv_layers((x_pr, x_cqt))

            if x.dim() == 4:
                if x.size(-2) == 1:         
                    x = x.squeeze(-2).transpose(1, 2) 
                elif x.size(-1) == 1:           
                    x = x.squeeze(-1).transpose(1, 2)  
                else:
             
                    if x.size(-2) <= x.size(-1):
                        x = x.mean(dim=-2).transpose(1, 2) 
                    else:
                        x = x.mean(dim=-1).transpose(1, 2)  
            elif x.dim() == 3:
         
                if x.size(1) > x.size(2):
                    x = x.transpose(1, 2)               
            else:
                raise ValueError(f"Unexpected multimodal conv output rank {x.dim()}")

            x, _ = self.gru(x)                           

            #Pre-attention (mean+std over time)
            x_np = x.squeeze(0).cpu().numpy()
            before = np.concatenate([x_np.mean(axis=0), x_np.std(axis=0)])  #512

            #Post-attention (global)
            after = self.context_attention(x).squeeze(0).cpu().numpy()      #256

            return before, after


    def extract_triple_embeddings(self, x1):
        with torch.no_grad():
            x = self.conv_layers(x1)                      
            x = x.squeeze(-2).transpose(1, 2)           
            x, _ = self.gru(x)                        

            #pre-attention (time-aware)
            pre = x.mean(dim=1)
            std = x.std(dim=1)
            pre_vec = torch.cat([pre, std], dim=1).squeeze(0).cpu().numpy()

            #post-attention (global 256d)
            z = self.context_attention(x)               
            post_vec = z.squeeze(0).cpu().numpy()

            #post-ReLU (task-tuned 256d)
            z_relu = self.non_linearity(z)
            tuned_vec = z_relu.squeeze(0).cpu().numpy()

            return pre_vec, post_vec, tuned_vec

def load_json(name_file):
    with open(name_file, 'r') as fp:
        data = json.load(fp)
    return data




