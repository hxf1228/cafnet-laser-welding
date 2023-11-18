# Created by Xufeng Huang on 2022-05-16
# Email: xufenghuang1228@gmail.com
# Description: convert raw data to specified format


import os
from multiprocessing.dummy import Pool as ThreadPool
import h5py
import pandas as pd
import numpy as np

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf


class DataHdf5(object):
    """
    Load .csv data, convert to hdf5
    """

    def __init__(self, cfg):
        self.root_dir = cfg.path.raw_path
        self.root_dir_split = self.root_dir.split(os.path.sep)
        self.file_list, self.category_list = self.get_file_list()
        self.output_name = "process.h5"
        self.output_path = os.path.join(cfg.path.process_path, self.output_name)

    def get_file_list(self):
        file_list = []
        category_list = []
        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            category_name = dirpath.split(os.path.sep)[-1]
            for filename in filenames:
                data_index = filename.split(".")[0]
                category_list.append(category_name + "_" + data_index)
                file_list.append(os.path.join(dirpath, filename))
        return file_list, category_list

    @staticmethod
    def read_csv_raw(file_list):
        data_raw_pd = pd.read_csv(file_list, header=None)
        data_raw = data_raw_pd.values
        return data_raw

    def random_serial_generator(self, input_data):
        random_num = np.random.uniform(low=1,
                                       high=input_data.shape[0] - self.cfg.DATASETS.DATA_POINTS,
                                       size=self.cfg.DATASETS.SAMPLES)
        random_serial = np.round(random_num).astype(np.int32)
        return random_serial

    def convert_hdf5(self, input_data):
        if not os.path.exists(self.output_path):
            h5f = h5py.File(self.output_path, 'w')
        else:
            h5f = h5py.File(self.output_path, 'r+')

        for i, iFile in enumerate(input_data):
            data_object = h5f.create_dataset(name=self.category_list[i],
                                             shape=iFile.shape,
                                             dtype=float)
            data_object[...] = iFile

        h5f.flush()
        h5f.close()

    def run(self):
        # Instead of using "for" for file traversal, which is time-consuming
        # Multiprocess: speed up ~10x
        # max_thread = multiprocessing.cpu_count()
        pool = ThreadPool()  # If processes is None then the number returned by os.cpu_count() is used.
        # read .csv
        data_raw = pool.map(self.read_csv_raw, self.file_list)
        pool.close()
        pool.join()

        self.convert_hdf5(data_raw)

