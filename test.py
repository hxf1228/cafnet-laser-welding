import warnings

import hydra
import numpy as np
import torch
import torch.utils.data
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import pandas as pd
from timeit import default_timer as timer
import matplotlib.pyplot as plt

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
    seed_list = [21895, 16604, 2403, 35593, 32332, 24879, 32009, 50810, 56351, 59628]
    # seed_list = [16604, ]
    for iSeed in seed_list:
        seed = iSeed
        set_seed(seed)

        x_data, y_label = gen_train.get_data()

        x_tr = torch.tensor(x_data)
        y_tr = torch.LongTensor(y_label)
        dataset = torch.utils.data.TensorDataset(x_tr, y_tr)

        dataset_size = len(dataset)
        shuffle_dataset = True
        train_ratio = 0.2
        test_ratio = 1 - train_ratio
        val_ratio = test_ratio

        train_num = int(np.floor(train_ratio * dataset_size))
        val_num = int(np.floor(val_ratio * dataset_size))
        test_num = int(np.floor(test_ratio * dataset_size))
        indices = list(range(dataset_size))
        if shuffle_dataset:
            np.random.shuffle(indices)
        train_indices = indices[0:train_num]
        val_indices = indices[train_num:]
        test_indices = val_indices
        # train_indices = indices[0:train_num - val_num]
        # val_indices = indices[train_num - val_num:train_num]
        # test_indices = indices[train_num:]

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
        checkpoint = torch.load(f'checkpoints/hyperpara/{cfg.model_name}_{train_ratio}_best.pt')
        # checkpoint = torch.load(f'checkpoints/{cfg.model_name}_best_{seed}.pt')
        classifier.load_state_dict(checkpoint)

        classifier.eval()

        with torch.no_grad():
            test_correct_num = 0
            total = 0
            inputs_test_all = []
            outputs_test_all = []
            last_feature_all = []
            labels_test_all = []
            conv2_feature_all = []
            atten_feature_all = []
            for iTest, (inputs_test, labels_test) in enumerate(test_loader):
                inputs_test_all.extend(inputs_test)
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
                test_correct_num += (pred_test == labels_test).sum().item()
                outputs_test_all.extend(outputs_test.cpu().numpy())
                last_feature_all.extend(classifier.module.last_feature.cpu().numpy())
                if cfg.model_name == "eamulticnn":
                    conv2_feature_all.extend(classifier.module.conv2_feature.cpu().numpy())
                    atten_feature_all.extend(classifier.module.att_feature.cpu().numpy())

            # predicted_times = []
            # for iTime in range(10):
            #     start_time1 = timer()
            #     outputs_test = classifier(input1_test, input2_test)
            #     #outputs_test = classifier(inputs_test)
            #     end_time1 = timer()
            #     once_time = end_time1 - start_time1
            #     predicted_times.append(once_time)
            # predicted_time = np.mean(predicted_times)
            #
            # pred_time_per = predicted_time
            inputs_test_all = np.stack(inputs_test_all, axis=0)
            outputs_test_all = np.stack(outputs_test_all, axis=0)
            last_feature_all = np.stack(last_feature_all, axis=0)
            labels_test_all = np.stack(labels_test_all, axis=0)
            vals, idx_start, count = np.unique(labels_test_all, return_counts=True, return_index=True)

            # # t-SNE
            # tsne = TSNE(n_components=2,
            #             perplexity=75,
            #             learning_rate="auto",
            #             n_iter=2000,
            #             init="random",
            #             random_state=iSeed)
            #
            # if cfg.model_name == "eamulticnn":
            #     conv2_feature_all = np.stack(conv2_feature_all, axis=0)
            #     conv2_feature_all_r = conv2_feature_all.reshape(total, -1)
            #     atten_feature_all = np.stack(atten_feature_all, axis=0)
            #     atten_feature_all_r = atten_feature_all.reshape(total, -1)
            #
            # inputs_test_reshape = inputs_test_all.reshape(total, -1)
            # atten_tsne = tsne.fit_transform(atten_feature_all_r)
            # conv2_tsne = tsne.fit_transform(conv2_feature_all_r)
            # last_tsne = tsne.fit_transform(last_feature_all)
            # input_tsne = tsne.fit_transform(inputs_test_reshape)
            # labels_test = np.expand_dims(labels_test_all, axis=1)
            # last_tsne_total = np.hstack((last_tsne, labels_test))
            # input_tsne_total = np.hstack((input_tsne, labels_test))
            # conv2_tsne_total = np.hstack((conv2_tsne, labels_test))
            # atten_tsne_total = np.hstack((atten_tsne, labels_test))
            #
            # # save results for plot
            # np.savetxt(f"results/{cfg.model_name}_last_tsne.csv",
            #            last_tsne_total,
            #            delimiter=",")
            # np.savetxt(f"results/{cfg.model_name}_conv2_tsne.csv",
            #            conv2_tsne_total,
            #            delimiter=",")
            # np.savetxt(f"results/{cfg.model_name}_input_tsne.csv",
            #            input_tsne_total,
            #            delimiter=",")
            # np.savetxt(f"results/{cfg.model_name}_atten_tsne.csv",
            #            atten_tsne_total,
            #            delimiter=",")
            #
            # plt.figure(figsize=(12, 5))
            # plt.subplot(141)
            # plt.scatter(input_tsne[:, 0], input_tsne[:, 1], c=labels_test_all)
            # plt.subplot(142)
            # plt.scatter(conv2_tsne[:, 0], conv2_tsne[:, 1], c=labels_test_all)
            # plt.subplot(143)
            # plt.scatter(atten_tsne[:, 0], atten_tsne[:, 1], c=labels_test_all)
            # plt.subplot(144)
            # plt.scatter(last_tsne[:, 0], last_tsne[:, 1], c=labels_test_all)
            # plt.margins(0, 0)
            # plt.show()

            # print(f"Predict time per sample: {pred_time_per * 1e3:.2f} ms")
            print('Seed: {}, Test Acc: {:.2f} %'.format(seed,
                                                        100 * test_correct_num / total))
            acc_i = 100 * test_correct_num / total
            acc_list.append(acc_i)
    print('Mean: {:.2f}, Std: {:.2f}'.format(np.mean(acc_list), np.std(acc_list)))

    GlobalHydra.get_state().clear()
    return 0


if __name__ == '__main__':
    main()
