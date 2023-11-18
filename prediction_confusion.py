import warnings

import os
import hydra
import numpy as np
import torch
import torch.utils.data
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from src.utils.plot_cm import plot_confusion_matrix

from src.datasets.dataloader import DataWeld
from src.models.get_model import get_model
from src.utils.reproduce import set_seed

warnings.filterwarnings('ignore')


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def main(cfg):
    OmegaConf.set_struct(cfg, False)
    if cfg.pretty_print:
        print(OmegaConf.to_yaml(cfg))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    gen_train = DataWeld(cfg)

    # seed = int(np.random.randint(0, 2 ** 16, 1))
    acc_list = []
    # seed_list = [21895, 16604, 2403, 35593, 32332, 24879, 32009, 50810, 56351, 59628]
    seed_list = [56351, ]
    for iSeed in seed_list:
        seed = iSeed
        set_seed(iSeed)

        x_data, y_label = gen_train.get_data()

        x_tr = torch.tensor(x_data)
        y_tr = torch.LongTensor(y_label)
        dataset = torch.utils.data.TensorDataset(x_tr, y_tr)

        dataset_size = len(dataset)
        shuffle_dataset = True
        train_ratio = 0.8
        test_ratio = 1 - train_ratio
        val_ratio = test_ratio
        train_num = int(np.floor(train_ratio * dataset_size))
        val_num = int(np.floor(val_ratio * dataset_size))
        test_num = int(np.floor(test_ratio * dataset_size))
        indices = list(range(dataset_size))
        if shuffle_dataset:
            np.random.shuffle(indices)
        train_indices = indices[0:train_num - val_num]
        val_indices = indices[train_num - val_num:train_num]
        test_indices = indices[train_num:]

        # Creating data samplers and loaders:
        train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
        test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

        train_loader = DataLoader(dataset,
                                  batch_size=cfg.params.batch_size,
                                  sampler=train_sampler, )

        val_loader = DataLoader(dataset,
                                batch_size=cfg.params.batch_size,
                                sampler=val_sampler,
                                )

        test_loader = DataLoader(dataset,
                                 batch_size=cfg.params.test_batch_size,
                                 sampler=test_sampler,
                                 shuffle=False,
                                 )

        classifier = torch.nn.DataParallel(get_model(cfg.model_name).to(device))
        #checkpoint = torch.load(f'checkpoints/hyperpara/{cfg.model_name}_0.9_best_16604.pt')
        checkpoint = torch.load(f'checkpoints/comparision/{cfg.model_name}-p_best_56351.pt')
        classifier.load_state_dict(checkpoint)

        classifier.eval()

        with torch.no_grad():
            test_correct_num = 0
            total = 0
            labels_test_all = []
            out_labels_test_all = []
            for iTest, (inputs_test, labels_test) in enumerate(test_loader):
                labels_test_all.extend(labels_test)
                inputs_test, labels_test = inputs_test.to(device), labels_test.to(device)
                if cfg.model_name == "multicnn" or cfg.model_name == "eamulticnn":
                    input1_test = inputs_test[:, :, 0].unsqueeze(1)
                    input2_test = inputs_test[:, :, 1].unsqueeze(1)
                    outputs_test = classifier(input1_test, input2_test)
                elif cfg.model_name == "multirescnn":
                    input1_test = inputs_test[:, :, :, 0].unsqueeze(1)
                    input2_test = inputs_test[:, :, :, 1].unsqueeze(1)
                    outputs_test = classifier(input1_test, input2_test)
                else:
                    outputs_test = classifier(inputs_test)
                _, pred_test = torch.max(outputs_test, 1)
                total += labels_test.size(0)
                out_labels_test_all.extend(pred_test.cpu().numpy())
                test_correct_num += (pred_test == labels_test).sum().item()

        labels_test_all = np.stack(labels_test_all, axis=0)
        out_labels_test_all = np.stack(out_labels_test_all, axis=0)
        vals, idx_start, count = np.unique(labels_test_all, return_counts=True, return_index=True)
        print(count)
        # print(f"Predict time per sample: {pred_time_per * 1e3:.2f} ms")
        print('Seed: {}, Test Acc: {:.2f} %'.format(seed,
                                                    100 * test_correct_num / total))
        acc_i = 100 * test_correct_num / total
        acc_list.append(acc_i)

        # confusion matrix
        CLASSES_NAME_WELD = {0: 'EP', 1: 'NFP', 2: 'IP', }

        # 设置画图属性
        plt.rc('font', family='Times New Roman')  # 修改画图的字体为TImes New Roman
        plt.rcParams['xtick.direction'] = 'in'  # 设置xtick和ytick的方向：in、out、inout

        plt.rcParams['ytick.direction'] = 'in'
        # Compute confusion matrix
        cnf_matrix = confusion_matrix(labels_test_all, out_labels_test_all)
        np.set_printoptions(precision=2)
        plt.figure(figsize=(5, 4))  # 1inch=2.5cm  8.4cm=3.36inch
        plt.rcParams.update({'font.size': 16})
        shrink_value = 0.9
        plot_confusion_matrix(cnf_matrix,
                              classes=CLASSES_NAME_WELD,
                              normalize=True,
                              shrink=shrink_value)
        tick_marks = np.array(range(len(CLASSES_NAME_WELD))) + 0.5

        pic_name = cfg.model_name + '_cm.tif'
        pic_path = os.path.join("results", "figures", pic_name)

        # 设置刻度线样式
        # offset the tick
        plt.clim(0, 100)
        plt.gca().set_xticks(tick_marks, minor=True)
        plt.gca().set_yticks(tick_marks, minor=True)
        plt.gca().xaxis.set_ticks_position('none')  # 把刻度线弄掉
        plt.gca().yaxis.set_ticks_position('none')
        plt.grid(True, which='minor', linewidth=0.8, linestyle='-')
        plt.subplots_adjust(top=0.95, bottom=0.12, right=0.95, left=0.18, hspace=0, wspace=0)  # 调整图像边缘
        plt.margins(0, 0)
        plt.savefig(pic_path, dpi=300, pil_kwargs={"compression": "tiff_lzw"})
        plt.show()

    print('Mean: {:.2f}, Std: {:.2f}'.format(np.mean(acc_list), np.std(acc_list)))

    GlobalHydra.get_state().clear()
    return 0


if __name__ == '__main__':
    main()
