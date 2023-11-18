import h5py
import numpy as np
from scipy import signal

from src.datasets.weld_path import All
from src.datasets.data_process import preprocess_raw_single, preprocess_img, preprocess_raw_multi


class DataWeld:
    def __init__(self, cfg):
        self._cfg = cfg
        self.class_name = All
        self.preprocess_switch = {'rgb': preprocess_img,
                                  'raw': preprocess_raw_single,
                                  'multi': preprocess_raw_multi}

    def get_data(self):
        data_h5 = h5py.File(self._cfg.path.process_path, 'r')
        # name = data_h5.keys()
        data_files = self.class_name
        n_class = len(data_files)  # the num of categories

        n_each_class = []
        label_set = []
        data_set = []
        preprocess_kwargs = {"shift_step": self._cfg.data.shift_step,
                             "sample_len": self._cfg.data.sample_len}

        if self._cfg.model_name == "rescnn":
            preprocess_func = self.preprocess_switch['rgb']
        elif self._cfg.model_name == "multirescnn":
            preprocess_func = self.preprocess_switch['rgb']  # rgb
        elif self._cfg.model_name == "multicnn" or self._cfg.model_name == "eamulticnn":
            preprocess_func = self.preprocess_switch['multi']
        else:
            preprocess_func = self.preprocess_switch['raw']

        for iClass in range(n_class):
            acoustic_group, photodiode_group = data_files[iClass]
            data_i_set = []
            for iCondi in range(len(acoustic_group)):
                data_raw_acoustic = data_h5[acoustic_group[iCondi]][()]
                data_raw_photodiode = data_h5[photodiode_group[iCondi]][()]
                up_multiple = int(data_raw_acoustic.size / data_raw_photodiode.size)
                data_poly_down_acoustic = signal.resample_poly(data_raw_acoustic,
                                                               up=1, down=up_multiple)
                # data_poly_up_photodiode = signal.resample_poly(data_raw_photodiode,
                #                                                up=up_multiple, down=1)  # up-sampling
                # data_part_acoustic = data_raw_acoustic[0:data_poly_up_photodiode.size]
                data_part_acoustic = data_poly_down_acoustic[0:data_raw_photodiode.size]

                # data_fusion = np.hstack((data_part_acoustic, data_poly_up_photodiode))  # up-sampling
                data_fusion = np.hstack((data_part_acoustic, data_raw_photodiode))  # down-sampling

                cut_data = preprocess_func(data_fusion, **preprocess_kwargs)
                # cut_data = preprocess_raw(data_fusion, **preprocess_kwargs)
                data_i_set.append(cut_data)
                # plt.plot(data_raw_photodiode)
                # plt.show()
            n_each_class.append(len(data_i_set))
            data_set.extend(data_i_set)
        data_h5.close()

        n_each_class = np.asarray(n_each_class)
        label = np.arange(n_class, dtype=np.int32).reshape(n_class, 1)
        label = np.repeat(label, n_each_class, axis=0).reshape(-1)
        data_set = np.asarray(data_set, dtype=np.float32)

        ds_n_each_class = np.min(n_each_class)

        # down-sampling
        # vals, idx_start, count = np.unique(label, return_counts=True, return_index=True)
        # selects = []
        # for i in range(len(vals)):
        #     index = list(range(count[i])) + idx_start[i]
        #     select = index[np.random.choice(len(index), size=ds_n_each_class, replace=False)]
        #     selects.append(select)
        # selects = np.array(selects).reshape(-1, )
        #
        # data_set = data_set[selects, :, :]
        # label = label[selects]
        #
        # vals_1, idx_start_1, count_1 = np.unique(label, return_counts=True, return_index=True)

        return data_set, label  # [Nc,2,1024], [Nc,]
