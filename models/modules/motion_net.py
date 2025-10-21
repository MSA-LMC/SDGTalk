import math
import numpy as np
import pickle
import torch
import torch.nn.functional as F
import torchvision
import torch.nn as nn
from pos_encoding import get_encoder


# Audio Net ===============================================================================================
class AudioNet_MLP(nn.Module):
    """
    For Hubert/Wav2Vec2
    """
    def __init__(self, dim_in=1024, dim_aud=64):
        super(AudioNet_MLP, self).__init__()
        self.dim_in = dim_in
        self.dim_aud = dim_aud
        self.encoder_mlp = nn.Sequential(
            nn.Linear(self.dim_in, 512, bias=True),  # [batch_size,16,dim_in 768/1024]    -> [batch_size,16,512]
            nn.LeakyReLU(0.02, True),
            nn.Linear(512, 256, bias=True),          # [batch_size,16,512]         -> [batch_size,16,256]
            nn.LeakyReLU(0.02, True),
            nn.Linear(256, 128, bias=True),           # [batch_size,16,256]         -> [batch_size,16, 128]
            nn.LeakyReLU(0.02, True),
            nn.Linear(128, 64, bias=True),            # [batch_size,16,128]          -> [batch_size,16,64]
            nn.LeakyReLU(0.02, True),
            nn.Linear(64, self.dim_aud, bias=True),  # [batch_size,16,64]          -> [batch_size,16,dim_aud]
        )

    def forward(self, x):
        """
        AudioNet: [batch_size, window_size 16,emb_dim 768/1024] -> [batch_size, window_size 16, emb_dim 64]
        """
        x = self.encoder_mlp(x) # [batch_size, 16, 768/1024] -> [batch_size, 16, dim_aud]
        return x

class AudioAttNet_MLP(nn.Module):
    """
    For Hubert/Wav2Vec2
    """
    def __init__(self, dim_aud=32, win_size=16):
        super(AudioAttNet_MLP, self).__init__()
        self.dim_aud = dim_aud
        self.win_size = win_size
        self.attention_mlp = nn.Sequential(
            nn.Linear(self.dim_aud, 32, bias=True), # [batch_size, 16, 64/32] -> [batch_size, 16, 32]
            nn.LeakyReLU(0.02, True),
            nn.Linear(32, 16, bias=True), # [batch_size, 16, 32] -> [batch_size, 16, 16]
            nn.LeakyReLU(0.02, True),
            nn.Linear(16, 8, bias=True), # [batch_size, 16, 16] -> [batch_size, 16, 8]
            nn.LeakyReLU(0.02, True),
            nn.Linear(8, 4, bias=True),  # [batch_size, 16, 8] -> [batch_size, 16, 4]
            nn.LeakyReLU(0.02, True),
            nn.Linear(4, 1, bias=True),  # [batch_size, 16, 4] -> [batch_size, 16, 1]
            nn.LeakyReLU(0.02, True)
        )
        self.attention_fc = nn.Sequential(
            nn.Linear(in_features=self.win_size, out_features=self.win_size, bias=True),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        """
        AudioAttNet_MLP: [batch_size, win_size 16, dim_aud 32] -> [batch_size, dim_aud 32]
        """
        # x: [batch_size, 16, 32/64]
        y = self.attention_mlp(x) # [batch_size, 16, 32/64] -> [batch_size, 16, 1]
        y = self.attention_fc(y.view(-1, self.win_size)).view(-1, self.win_size, 1) # [batch_size, 16, 1] -> [batch_size, 16, 1]
        return torch.sum(y * x, dim=1) # [batch_size, dim_aud]
# =========================================================================================================


# Eye Net ===============================================================================================
class EyeNet_MLP(nn.Module):
    """
    For Eye Landmark
    """
    def __init__(self, dim_in=6, dim_eye=12):
        super(EyeNet_MLP, self).__init__()
        self.dim_in = dim_in
        self.dim_eye = dim_eye
        self.encoder_mlp = nn.Sequential(
            nn.Linear(self.dim_in, 12, bias=True),  # [batch_size,dim_in 12]    -> [batch_size, 12]
            nn.LeakyReLU(0.02, True),
            nn.Linear(12, 12, bias=True),          # [batch_size,12]         -> [batch_size,12]
            nn.LeakyReLU(0.02, True),
            nn.Linear(12, self.dim_eye, bias=True),          # [batch_size,12]         -> [batch_size,self.dim_eye]
        )

    def forward(self, x):
        """
        EyeNetNet: [batch_size,emb_dim 6] -> [batch_size, emb_dim 12]
        """
        x = self.encoder_mlp(x) # [batch_size, 6] -> [batch_size, 12]
        return x
# =========================================================================================================



class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden, self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=False))

        self.net = nn.ModuleList(net)
    
    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
                # x = F.dropout(x, p=0.1, training=self.training)
                
        return x
    
# ==========================================================================================================

class MotionNetworkMLP(nn.Module):
    def __init__(self,
                 audio_extractor = 'hubert',
                 ):
        super(MotionNetworkMLP, self).__init__()

        self.bound = 1.15

        # DYNAMIC PART
        self.num_levels = 12
        self.level_dim = 1
        self.encoder_xy, self.in_dim_xy = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=16, log2_hashmap_size=17, desired_resolution=512 * self.bound)
        self.encoder_yz, self.in_dim_yz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=16, log2_hashmap_size=17, desired_resolution=512 * self.bound)
        self.encoder_xz, self.in_dim_xz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=16, log2_hashmap_size=17, desired_resolution=512 * self.bound)

        self.in_dim = self.in_dim_xy + self.in_dim_yz + self.in_dim_xz

        self.win_size = 16

        # audio network
        if audio_extractor == 'hubert':
            self.audio_in_dim = 1024
            self.audio_dim = 64
            self.audio_net = AudioNet_MLP(self.audio_in_dim, self.audio_dim)
        elif audio_extractor == 'wav2vec2':
            self.audio_in_dim = 768
            self.audio_dim = 64
            self.audio_net = AudioNet_MLP(self.audio_in_dim, self.audio_dim)
        else:
            raise NotImplementedError
        self.audio_att_net = AudioAttNet_MLP(self.audio_dim,self.win_size)
        self.aud_ch_att_net = MLP(self.in_dim, self.audio_dim, 32, 2)

        
        # eye network
        self.eye_in_dim = 6
        self.eye_out_dim = 6
        self.eye_net = EyeNet_MLP(self.eye_in_dim, self.eye_out_dim)
        self.eye_att_net = MLP(self.in_dim, self.eye_out_dim, 16, 2)



        # Motion Field
        # d_xyz:3  d_rot:4  d_scale:3 opacities:1 (3+4+3+1=11)
        self.motion_num_layers = 3       
        self.motion_hidden_dim = 64
        self.motion_out_dim = 11
        self.motion_net = MLP(self.in_dim + self.audio_dim + self.eye_out_dim, self.motion_out_dim, self.motion_hidden_dim, self.motion_num_layers)

        self._initialize_weights()


    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
                    
    @staticmethod
    @torch.jit.script
    def split_xyz(x):
        xy, yz, xz = x[:, :, :-1], x[:, :, 1:], torch.cat([x[:, :,:1], x[:, :,-1:]], dim=-1)
        return xy, yz, xz


    def encode_x(self, xyz, bound):
        # x: [B, N, 3], in [-bound, bound]
        xy, yz, xz = self.split_xyz(xyz)
        feat_xy = self.encoder_xy(xy, bound=bound)
        feat_yz = self.encoder_yz(yz, bound=bound)
        feat_xz = self.encoder_xz(xz, bound=bound)
        
        return torch.cat([feat_xy, feat_yz, feat_xz], dim=-1)
    
    def encode_audio(self, a):
        # a: [batch_size, 16, 29/768/1024], audio features from deepspeech/hubert/wav2vec2
        enc_a = self.audio_net(a) 
        enc_a = self.audio_att_net(enc_a) 
        return enc_a
    
    def encode_eye(self, eye):
        # eye: [batch_size, 6], eye features
        enc_eye = self.eye_net(eye) 
        return enc_eye
    

    def forward(self, xyz, audio_feat, eye_feat):
        enc_x = self.encode_x(xyz, bound=self.bound) # enc_x: [B, N, 12*3]

        # audio =============================================
        # audio_feat: [B,time_window 16, emb_dim 29/768/1024]
        enc_a = self.encode_audio(audio_feat) # enc_a.shape = [B, 32]
        enc_a = enc_a.unsqueeze(1).repeat(1, enc_x.shape[1], 1)  # enc_a.shape : [B, 32] -> [B, 1, 32] -> [B, N, 32] 
        aud_ch_att = self.aud_ch_att_net(enc_x) # [B, N, 12*3] -> [B, N, 32]
        enc_audio = enc_a * aud_ch_att  # [B, N, 32] * [B, N, 32] = [B, N, 32]
        # ===================================================

        # eye ================================================
        # eye: [B, 6]
        enc_eye = self.encode_eye(eye_feat)
        enc_eye = enc_eye.unsqueeze(1).repeat(1, enc_x.shape[1], 1)
        eye_ch_att = self.eye_att_net(enc_x) # [B, N, 12*3] -> [B, N, 12]
        enc_blink = enc_eye * eye_ch_att  # [B, N, 12] * [B, N, 12] = [B, N, 12]
        # ===================================================


        motion_feat = torch.cat([enc_x, enc_audio, enc_blink], dim=-1) # h.shape = [B, N, 12*3+64+6] = [B, N, 106] 
        motion_pred = self.motion_net(motion_feat) # h.shape = [B, N, 11]


        d_xyz = motion_pred[..., :3] # 3
        d_rotations = motion_pred[..., 3:7] # 4
        d_scales = motion_pred[..., 7:10] # 3
        d_opacities = motion_pred[..., 10:11] # 1
        
        results = {
            'd_xyz': d_xyz,
            'd_rotations': d_rotations,
            'd_scales': d_scales,
            'd_opacities': d_opacities,
        }
        return results
