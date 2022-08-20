import torch
from torch import nn
import numpy as np


class Tacotron2Loss(nn.Module):
    def __init__(self, mel_weight=1, gate_weight=0.005, vel_weight=1, pos_weight=0.1, add_l1_loss=False):
        super(Tacotron2Loss, self).__init__()
        self.w_mel = mel_weight
        self.w_gate = gate_weight
        self.w_vel_mel = vel_weight
        self.w_pos = pos_weight
        self.add_l1_loss = add_l1_loss

    def forward(self, model_output, targets):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        vel_mel_target = mel_target[..., 1:] - mel_target[..., :-1]
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, _ = model_output
        gate_out = gate_out.view(-1, 1)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)

        if mel_target.shape[1] == 78:
            mel_loss = nn.MSELoss()(mel_out[:, :-3], mel_target[:, :-3]) + \
                       nn.MSELoss()(mel_out_postnet[:, :-3], mel_target[:, :-3])
            vel_mel_loss = nn.MSELoss()(mel_out[:, :-3, 1:] - mel_out[:, :-3, :-1], vel_mel_target[:, :-3]) + \
                           nn.MSELoss()(mel_out_postnet[:, :-3, 1:] - mel_out_postnet[:, :-3, :-1], vel_mel_target[:, :-3])

            pos_loss = nn.MSELoss()(mel_out[:, -3:], mel_target[:, -3:]) + \
                       nn.MSELoss()(mel_out_postnet[:, -3:], mel_target[:, -3:])
            vel_pos_loss = nn.MSELoss()(mel_out[:, -3:, 1:] - mel_out[:, -3:, :-1], vel_mel_target[:, -3:]) + \
                           nn.MSELoss()(mel_out_postnet[:, -3:, 1:] - mel_out_postnet[:, -3:, :-1], vel_mel_target[:, -3:])
            if self.add_l1_loss:
                mel_l1loss = nn.L1Loss()(mel_out[:, :-3], mel_target[:, :-3]) + \
                             nn.L1Loss()(mel_out_postnet[:, :-3], mel_target[:, :-3])
                vel_mel_l1loss = nn.L1Loss()(mel_out[:, :-3, 1:] - mel_out[:, :-3, :-1], vel_mel_target[:, :-3]) + \
                                 nn.L1Loss()(mel_out_postnet[:, :-3, 1:] - mel_out_postnet[:, :-3, :-1],
                                             vel_mel_target[:, :-3])
                pos_l1loss = nn.L1Loss()(mel_out[:, -3:], mel_target[:, -3:]) + \
                             nn.L1Loss()(mel_out_postnet[:, -3:], mel_target[:, -3:])
                vel_pos_l1loss = nn.L1Loss()(mel_out[:, -3:, 1:] - mel_out[:, -3:, :-1], vel_mel_target[:, -3:]) + \
                                 nn.L1Loss()(mel_out_postnet[:, -3:, 1:] - mel_out_postnet[:, -3:, :-1],
                                            vel_mel_target[:, -3:])
                return self.w_gate * gate_loss + \
                       self.w_mel * (mel_loss + mel_l1loss) + \
                       self.w_vel_mel * (vel_mel_loss + vel_mel_l1loss) + \
                       self.w_pos * (self.w_mel * (pos_loss + pos_l1loss) + self.w_vel_mel * (vel_pos_loss + vel_pos_l1loss))
            else:
                return self.w_mel * mel_loss + self.w_gate * gate_loss + self.w_vel_mel * vel_mel_loss + \
                       self.w_pos * (self.w_mel * pos_loss + self.w_vel_mel * vel_pos_loss)
        else:
            mel_loss = nn.MSELoss()(mel_out, mel_target) + nn.MSELoss()(mel_out_postnet, mel_target)
            vel_mel_loss = nn.MSELoss()(mel_out[..., 1:] - mel_out[..., :-1], vel_mel_target) + \
                           nn.MSELoss()(mel_out_postnet[..., 1:] - mel_out_postnet[..., :-1], vel_mel_target)
            if self.add_l1_loss:
                mel_l1loss = nn.L1Loss()(mel_out, mel_target) + nn.L1Loss()(mel_out_postnet, mel_target)
                vel_mel_l1loss = nn.L1Loss()(mel_out[..., 1:] - mel_out[..., :-1], vel_mel_target) + \
                                 nn.L1Loss()(mel_out_postnet[..., 1:] - mel_out_postnet[..., :-1], vel_mel_target)
                return self.w_mel * (mel_loss + mel_l1loss) + self.w_vel_mel * (vel_mel_loss + vel_mel_l1loss)
            else:
                return self.w_gate * gate_loss + self.w_mel * mel_loss + self.w_vel_mel * vel_mel_loss

