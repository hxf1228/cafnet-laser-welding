# Created by Xufeng Huang on 10/02/2022 GMT+08:00
# Email: xufenghuang1228@gmail.com


import numpy as np
from sklearn.preprocessing import minmax_scale, normalize
from sklearn.utils import shuffle
from scipy import signal


def preprocess_img(input_data, shift_step, sample_len):
    seq_indices = list(range(0, input_data.shape[0] - sample_len, shift_step))

    output_data = []
    # input_data = normalize(input_data, norm='l2', axis=0)
    input_data_1 = input_data[:, 0]
    input_data_2 = input_data[:, 1]
    input_data_n_1 = minmax_scale(input_data_1, feature_range=(0, 255))
    input_data_n_2 = minmax_scale(input_data_2, feature_range=(0, 255))

    for iCut in range(len(seq_indices)):
        cut_index = seq_indices[iCut]
        data_cut_1 = input_data_n_1[cut_index:cut_index + sample_len]
        data_cut_2 = input_data_n_2[cut_index:cut_index + sample_len]

        reshape_num = int(sample_len ** 0.5)
        img_1_reshape = data_cut_1.reshape((reshape_num, reshape_num), order="F")
        img_2_reshape = data_cut_2.reshape((reshape_num, reshape_num), order="F")
        img = np.stack([img_1_reshape, img_2_reshape], axis=2)
        #img = img_2_reshape  # without resize
        output_data.append(img)
    data = output_data
    return data


def preprocess_raw_multi(input_data, shift_step, sample_len):
    seq_indices = list(range(0, input_data.shape[0] - sample_len, shift_step))
    output_data_pre = []

    input_data = normalize(input_data, norm='l2', axis=0)

    for iCut in range(len(seq_indices)):
        cut_index = seq_indices[iCut]
        data_cut = input_data[cut_index:cut_index + sample_len, :]
        #data_cut = normalize(data_cut, norm='l2', axis=0)
        output_data_pre.append(data_cut)

    output_data = output_data_pre

    return output_data


def preprocess_raw_single(input_data, shift_step, sample_len):
    seq_indices = list(range(0, input_data.shape[0] - sample_len, shift_step))
    output_data_pre = []

    input_data = normalize(input_data, norm='l2', axis=0)

    for iCut in range(len(seq_indices)):
        cut_index = seq_indices[iCut]
        data_cut = input_data[cut_index:cut_index + sample_len, 1]
        output_data_pre.append(data_cut)

    output_data = np.expand_dims(output_data_pre, axis=1)

    return output_data

