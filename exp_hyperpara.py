import copy

import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader
import numpy as np

import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from src.datasets.dataloader import DataWeld
from src.models.one_cnn import CNN1D
from src.models.rescnn import ResCNN
from src.utils.reproduce import set_seed
from src.models.get_model import get_model


@hydra.main(config_path="conf", config_name="exp_hyperpara", version_base="1.2")
def main(cfg):
    OmegaConf.set_struct(cfg, False)
    if cfg.pretty_print:
        print(OmegaConf.to_yaml(cfg))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    seed_list = [21895, 16604, 2403, 35593, 32332, 24879, 32009, 50810, 56351, 59628]
    seed = seed_list[0]
    set_seed(seed)
    gen_train = DataWeld(cfg)

    x_data, y_label = gen_train.get_data()

    x_tr = torch.tensor(x_data)
    y_tr = torch.LongTensor(y_label)
    dataset = torch.utils.data.TensorDataset(x_tr, y_tr)

    dataset_size = len(dataset)
    shuffle_dataset = True
    # train_ratio = 0.8
    # test_ratio = (1 - train_ratio)/ 2
    # val_ratio = test_ratio

    train_ratio = 0.8
    test_ratio = 1 - train_ratio
    val_ratio = test_ratio

    train_num = int(np.floor(train_ratio * dataset_size))
    val_num = int(np.floor(val_ratio * dataset_size))
    test_num = int(np.floor(test_ratio * dataset_size))
    indices = list(range(dataset_size))
    if shuffle_dataset:
        set_seed(seed)
        np.random.shuffle(indices)
    train_indices = indices[0:train_num]
    val_indices = indices[train_num:]
    # train_indices = indices[0:train_num - val_num]
    # val_indices = indices[train_num - val_num:train_num]
    # test_indices = indices[train_num:]
    test_indices = val_indices

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
                             batch_size=cfg.params.batch_size,
                             sampler=test_sampler,
                             )

    criteria = nn.CrossEntropyLoss()
    classifier = torch.nn.DataParallel(get_model(cfg.model_name).cuda())

    num_epochs = 200
    min_loss = 10000
    best_epoch = 1
    optimizer = torch.optim.SGD(classifier.parameters(), lr=3e-2)
    #optimizer = torch.optim.Adam(classifier.parameters(), lr=3e-3)  # rescnn
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
    #                                             step_size=10,
    #                                             gamma=0.5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=10, eta_min=3e-5)

    classifier.to(device)

    print('model_parameter...............')
    num_params = 0
    for param in classifier.parameters():
        num_params += param.numel()
    print(num_params / 1e6, 'M')  # unit: M

    best_model = None
    val_loss_epoch = []
    train_loss_epoch = []

    for iEpoch in range(num_epochs):
        losses = []
        val_losses = []
        test_losses = []
        # Train process
        classifier.train()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            if cfg.model_name == "multicnn" or cfg.model_name == "eamulticnn":
                input1, input2 = inputs[:, :, 0].unsqueeze(1), inputs[:, :, 1].unsqueeze(1)
                outputs = classifier(input1, input2)
            elif cfg.model_name == "multirescnn":
                input1, input2 = inputs[:, :, :, 0].unsqueeze(1), inputs[:, :, :, 1].unsqueeze(1)
                outputs = classifier(input1, input2)
            else:
                outputs = classifier(inputs)

            loss = criteria(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            lr = scheduler.get_last_lr()
            losses.append(loss.cpu().item())  # Train losses (total)
        train_loss = sum(losses) / (len(train_indices))
        train_loss_epoch.append(train_loss)
        # Validation process
        classifier.eval()
        with torch.no_grad():
            for iVal, (inputs_val, labels_val) in enumerate(val_loader):
                inputs_val, labels_val = inputs_val.to(device), labels_val.to(device)
                if cfg.model_name == "multicnn" or cfg.model_name == "eamulticnn":
                    input1_val = inputs_val[:, :, 0].unsqueeze(1)
                    input2_val = inputs_val[:, :, 1].unsqueeze(1)
                    outputs_val = classifier(input1_val, input2_val)
                elif cfg.model_name == "multirescnn":
                    input1_val = inputs_val[:, :, :, 0].unsqueeze(1)
                    input2_val = inputs_val[:, :, :, 1].unsqueeze(1)
                    outputs_val = classifier(input1_val, input2_val)
                else:
                    outputs_val = classifier(inputs_val)

                loss_val = criteria(outputs_val, labels_val)
                val_losses.append(loss_val.cpu().item())
            val_loss = sum(val_losses) / (len(val_indices))
            val_loss_epoch.append(val_loss)
            if val_loss < min_loss:
                min_loss = val_loss
                best_epoch = iEpoch
                best_model = copy.deepcopy(classifier)
                # torch.save(best_model, f'checkpoints/{cfg.model_name}_best.pt')
                save_path = f'checkpoints/{cfg.exp_name}/{cfg.model_name}-p_{train_ratio}_best.pt'
                torch.save(best_model.state_dict(), save_path)
            print(
                '[epoch %d] %s loss: %f min loss: %f at epoch %d ' %
                (iEpoch, 'val', val_loss, min_loss, best_epoch))
        print('[epoch %d] train loss: %f ' % (iEpoch, train_loss))
    train_loss_epoch = np.expand_dims(train_loss_epoch, axis=1)
    val_loss_epoch = np.expand_dims(val_loss_epoch, axis=1)
    loss_total = np.hstack((train_loss_epoch, val_loss_epoch))

    np.savetxt(f"results/{cfg.exp_name}/{cfg.model_name}_loss.csv",
               loss_total,
               fmt='%.6f',
               delimiter=",")
    # Test
    best_model.eval()
    with torch.no_grad():
        test_correct_num = 0
        total = 0
        for iTest, (inputs_test, labels_test) in enumerate(test_loader):
            inputs_test, labels_test = inputs_test.to(device), labels_test.to(device)
            if cfg.model_name == "multicnn" or cfg.model_name == "eamulticnn":
                input1_test = inputs_test[:, :, 0].unsqueeze(1)
                input2_test = inputs_test[:, :, 1].unsqueeze(1)
                outputs_test = best_model(input1_test, input2_test)
            elif cfg.model_name == "multirescnn":
                input1_test = inputs_test[:, :, :, 0].unsqueeze(1)
                input2_test = inputs_test[:, :, :, 1].unsqueeze(1)
                outputs_test = best_model(input1_test, input2_test)
            else:
                outputs_test = best_model(inputs_test)

            _, pred_test = torch.max(outputs_test, 1)
            total += labels_test.size(0)
            test_correct_num += (pred_test == labels_test).sum().item()

        print('Seed: {}, Test Acc: {:.2f} %'.format(seed,
                                                    100 * test_correct_num / total))

    GlobalHydra.get_state().clear()
    return 0


if __name__ == '__main__':
    main()
