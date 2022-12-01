import torch.nn as nn
import torch
import numpy as np
import math
import torch.nn.functional as F

class Relation_module(nn.Module):
    def __init__(self, in_channel, out_channel, Nr):
        super(Relation_module, self).__init__()

        self.out_channel = out_channel

        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=(1,1), groups=Nr)

        self.fc1 = nn.Linear(64, Nr)
        self.query_fc = nn.Linear(out_channel, out_channel)
        self.key_fc = nn.Linear(out_channel, out_channel)

        self.relu_1 = nn.ReLU(inplace=True)
        self.relu_2 = nn.ReLU(inplace=True)

    def extract_position_matrix(self, bbox):

        n_batches = bbox.size(0)

        # extract position matrix
       
        xmin, ymin, xmax, ymax = torch.split(bbox, [1, 1, 1, 1], dim=2)

        bbox_width = xmax - xmin + 1.
        bbox_height = ymax - ymin + 1.
        center_x = 0.5 * (xmin + xmax)
        center_y = 0.5 * (ymin + ymax)

        delta_x = center_x - torch.transpose(center_x, 1, 2)
        delta_x = torch.div(delta_x, bbox_width)
        delta_x = torch.log(torch.max(torch.abs(delta_x), (torch.full((n_batches, 100, 100), 1e-3)).cuda()))
        delta_y = center_y - torch.transpose(center_y, 1, 2)
        delta_y = torch.div(delta_y, bbox_height)
        delta_y = torch.log(torch.max(torch.abs(delta_y), (torch.full((n_batches, 100, 100), 1e-3)).cuda()))
        delta_width = torch.div(bbox_width, torch.transpose(bbox_width, 1, 2))
        delta_width = torch.log(delta_width)
        delta_height = torch.div(bbox_height, torch.transpose(bbox_height, 1, 2))
        delta_height = torch.log(delta_height)

        concat_list = [delta_x, delta_y, delta_width, delta_height]

        for idx, sym in enumerate(concat_list):
            concat_list[idx] = sym.unsqueeze(3)

        position_matrix = torch.cat((concat_list), 3)

        return position_matrix

    def extract_position_embedding(self, position_mat, feat_dim=64, wave_length=1000):

        n_batches = position_mat.size(0)

        feat_range = torch.arange(0, feat_dim / 8)
        dim_mat = np.power(wave_length, (8. / feat_dim) * feat_range.float())
        dim_mat = torch.reshape(dim_mat, (1, 1, 1, 1, -1))
        position_mat = (100.0 * position_mat).unsqueeze(4)
        div_mat = torch.div(position_mat, dim_mat.cuda())
        sin_mat = torch.sin(div_mat)
        cos_mat = torch.cos(div_mat)

        embedding = torch.cat((sin_mat, cos_mat), 4)
        embedding = torch.reshape(embedding, (n_batches, 100, 100, feat_dim))

        return embedding

    def attention_module_multi_head(self, feat, position_embedding, feat_mask, fc_dim, group, feat_dim, dim, nongt_dim=100):

        n_batches = feat.size(0)

        dim_group = (dim[0] / group, dim[1] / group, dim[2] / group)

        position_embedding_reshape = torch.reshape(position_embedding, (n_batches, 10000, 64))
        # WG
        position_feat_1 = self.fc1(position_embedding_reshape)
        position_feat_1_relu = self.relu_1(position_feat_1)
        aff_weight = torch.reshape(position_feat_1_relu, (n_batches, -1, nongt_dim, fc_dim))
        aff_weight = torch.transpose(aff_weight, 2, 3)

        # WQ
        q_data = self.query_fc(feat)
        q_data_batch = torch.reshape(q_data, (n_batches, -1, group, int(dim_group[0])))
        q_data_batch = torch.transpose(q_data_batch, 1, 2)

        # WK
        k_data = self.key_fc(feat)
        k_data_batch = torch.reshape(k_data, (n_batches, -1, group, int(dim_group[1])))
        k_data_batch = torch.transpose(k_data_batch, 1, 2)

        v_data = feat

        # WA
        aff = torch.matmul(q_data_batch, torch.transpose(k_data_batch, 2, 3))
        aff_scale = (1.0 / math.sqrt(float(dim_group[1]))) * aff
        aff_scale = torch.transpose(aff_scale, 1, 2)

        # weighted_aff
        weighted_aff = torch.log(torch.max(aff_weight, torch.full((n_batches, 100, fc_dim, nongt_dim), 1e-6).cuda())) + aff_scale
        weighted_aff = weighted_aff.masked_fill(feat_mask, -1e9)
        aff_softmax = F.softmax(weighted_aff, dim=3)
        aff_softmax_reshape = torch.reshape(aff_softmax, (n_batches, 100 * fc_dim, nongt_dim))

        # W * FA
        output_t = torch.matmul(aff_softmax_reshape, v_data)
        output_t = torch.reshape(output_t, (n_batches * 100, fc_dim * feat_dim, 1, 1))
        linear_out = self.conv(output_t)
        linear_out = torch.reshape(linear_out, (n_batches, 100, feat_dim, 1, 1))
        output = torch.reshape(linear_out, (n_batches, 100, dim[2]))

        return output

    def forward(self, feat, bbox, feat_mask):

        position_matrix = self.extract_position_matrix(bbox)
        position_embedding = self.extract_position_embedding(position_matrix)

        attention_1 = self.attention_module_multi_head(feat, position_embedding, feat_mask, 32, 32, self.out_channel, dim=(self.out_channel, self.out_channel, self.out_channel))
        new_feat = feat + attention_1
        new_feat_relu = self.relu_2(new_feat)

        return new_feat_relu