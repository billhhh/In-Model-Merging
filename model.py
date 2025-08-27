from __future__ import print_function, division

# pytorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# image imports
from skimage import io, transform
from PIL import Image

# general imports
import os
import time
from shutil import copyfile
from shutil import rmtree

# data science imports
import pandas as pd
import numpy as np
import csv

import cxr_dataset as CXR
import eval_model as E
import random
import torch.nn.functional as F

use_gpu = torch.cuda.is_available()
gpu_count = torch.cuda.device_count()
print("Available GPU count:" + str(gpu_count))

shallow_layers = 10
prob = 0.3
similarity_threshold = 0.3
inm_epochs = 10
backbone = True
load_ckpt = 'results/In_Model_Merging_backbone_training_l10_e30/checkpoint.pth'
is_train = True


def merge_similar_kernels_layerwise(model, shallow_layers=10, prob=0.1, similarity_threshold=0.2):

    def compute_cosine_similarity(kernel1, kernel2):
        flat_kernel1 = kernel1.view(-1)
        flat_kernel2 = kernel2.view(-1)
        cosine_similarity = F.cosine_similarity(flat_kernel1.unsqueeze(0), flat_kernel2.unsqueeze(0))
        return cosine_similarity.item()

    conv_layer_count = 0
    for layer in model.features:
        if isinstance(layer, nn.Conv2d):
            conv_layer_count += 1

            if conv_layer_count <= shallow_layers:
                continue

            weight = layer.weight.data  # [out_channels, in_channels, kernel_h, kernel_w]
            out_channels = weight.size(0)

            if out_channels > 1:
                new_weight = weight.clone()
                for i in range(out_channels):
                    if random.random() < prob:
                        other_idx = random.choice([j for j in range(out_channels) if j != i])
                        similarity = compute_cosine_similarity(weight[i], weight[other_idx])
                        if similarity > similarity_threshold:
                            new_weight[i] = 0.8 * weight[i] + 0.2 * weight[other_idx]

                layer.weight.data.copy_(new_weight)


def merge_similar_kernels_resnet(model, shallow_layers=10, prob=0.1, similarity_threshold=0.2):

    def compute_cosine_similarity(kernel1, kernel2):
        flat_kernel1 = kernel1.view(-1)
        flat_kernel2 = kernel2.view(-1)
        cosine_similarity = F.cosine_similarity(flat_kernel1.unsqueeze(0), flat_kernel2.unsqueeze(0))
        return cosine_similarity.item()

    block_count = 0

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
            continue

        if isinstance(module, (nn.Sequential, nn.ModuleList)):
            for submodule in module:
                if isinstance(submodule, (nn.Conv2d, nn.BatchNorm2d)):
                    continue

                if isinstance(submodule, (nn.Sequential, nn.ModuleList)):
                    for block in submodule:
                        if isinstance(block, (nn.Conv2d, nn.BatchNorm2d)):
                            continue

                        if hasattr(block, 'conv2'):
                            block_count += 1

                            if block_count <= shallow_layers:
                                continue

                            conv_layer = block.conv2
                            weight = conv_layer.weight.data  # [out_channels, in_channels, kernel_h, kernel_w]
                            out_channels = weight.size(0)

                            if out_channels > 1:
                                new_weight = weight.clone()
                                for i in range(out_channels):
                                    if random.random() < prob:
                                        other_idx = random.choice([j for j in range(out_channels) if j != i])
                                        similarity = compute_cosine_similarity(weight[i], weight[other_idx])

                                        if similarity > similarity_threshold:
                                            new_weight[i] = 0.8 * weight[i] + 0.2 * weight[other_idx]

                                conv_layer.weight.data.copy_(new_weight)


def checkpoint(model, best_loss, epoch, LR, folder_name, optimizer):
    """
    Saves checkpoint of torchvision model during training.

    Args:
        model: torchvision model to be saved
        best_loss: best val loss achieved so far in training
        epoch: current epoch of training
        LR: current learning rate in training
    Returns:
        None
    """

    print('saving')
    state = {
        'model': model,
        'best_loss': best_loss,
        'epoch': epoch,
        'rng_state': torch.get_rng_state(),
        'LR': LR,
        'optimizer': optimizer
    }

    save_path = 'results/' + folder_name + '/checkpoint.pth'
    torch.save(state, save_path)


def train_model(
        args,
        model,
        criterion,
        optimizer,
        LR,
        num_epochs,
        dataloaders,
        dataset_sizes,
        weight_decay):
    """
    Finetune torchvision model to NIH CXR data.

    Args:
        model: torchvision model to be finetuned (densenet-121 in this case)
        criterion: loss criterion (binary cross entropy loss, BCELoss)
        optimizer: optimizer to use in training (SGD)
        LR: learning rate
        num_epochs: continue training up to this many epochs
        dataloaders: pytorch train and val dataloaders
        dataset_sizes: length of train and val datasets
        weight_decay: weight decay parameter we use in SGD with momentum
    Returns:
        model: trained torchvision model
        best_epoch: epoch on which best model val loss was obtained

    """
    since = time.time()

    start_epoch = 1
    best_loss = 999999
    best_epoch = -1
    last_train_loss = -1

    # iterate over epochs
    for epoch in range(start_epoch, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # set model to train or eval mode based on whether we are in train or
        # val; necessary to get correct predictions given batchnorm
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0

            i = 0
            total_done = 0
            # iterate over all data in train/val dataloader:
            for data in dataloaders[phase]:
                i += 1
                inputs, labels, _ = data
                batch_size = inputs.shape[0]
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda()).float()

                if not backbone:
                    if isinstance(model, models.ResNet):
                        merge_similar_kernels_resnet(model, shallow_layers=shallow_layers, prob=prob,
                                                     similarity_threshold=similarity_threshold)
                    else:
                        merge_similar_kernels_layerwise(model, shallow_layers=shallow_layers, prob=prob,
                                                        similarity_threshold=similarity_threshold)

                outputs = model(inputs)

                # calculate gradient and update parameters in train phase
                optimizer.zero_grad()
                loss = criterion(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data * batch_size

            epoch_loss = running_loss / dataset_sizes[phase]

            if phase == 'train':
                last_train_loss = epoch_loss

            print(phase + ' epoch {}:loss {:.4f} with data size {}'.format(
                epoch, epoch_loss, dataset_sizes[phase]))

            # decay learning rate if no val loss improvement in this epoch
            if phase == 'val' and epoch_loss > best_loss and ((epoch - best_epoch) >= args.tolerance):
                print("decay loss from " + str(LR) + " to " +
                      str(LR / 10) + " as not seeing improvement in val loss")
                LR = LR / 10
                # create new optimizer with lower learning rate
                optimizer = optim.SGD(
                    filter(
                        lambda p: p.requires_grad,
                        model.parameters()),
                    lr=LR,
                    momentum=0.9,
                    weight_decay=weight_decay)
                print("created new optimizer with LR " + str(LR))

            # checkpoint model if has best val loss yet
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_epoch = epoch
                checkpoint(model, best_loss, epoch, LR, args.name, optimizer)

            # log training and validation loss over each epoch
            if phase == 'val':
                with open('results/' + args.name + '/log_train.log', 'a') as logfile:
                    logwriter = csv.writer(logfile, delimiter=',')
                    if(epoch == 1):
                        logwriter.writerow(["epoch", "train_loss", "val_loss"])
                    logwriter.writerow([epoch, last_train_loss, epoch_loss])

        total_done += batch_size
        if(total_done % (100 * batch_size) == 0):
            print("completed " + str(total_done) + " so far in epoch")

        # break if no val loss improvement in 3 epochs
        # if ((epoch - best_epoch) >= 3):
        #     print("no improvement in 3 epochs, break")
        #     break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights to return
    checkpoint_best = torch.load('results/' + args.name + '/checkpoint.pth')
    model = checkpoint_best['model']

    return model, best_epoch


def train_cnn(args, PATH_TO_IMAGES, LR, WEIGHT_DECAY, NUM_EPOCHS, BATCH_SIZE):
    """
    Train torchvision model to NIH data given high level hyperparameters.

    Args:
        PATH_TO_IMAGES: path to NIH images
        LR: learning rate
        WEIGHT_DECAY: weight decay parameter for SGD

    Returns:
        preds: torchvision model predictions on test fold with ground truth for comparison
        aucs: AUCs for each train,test tuple

    """

    # try:
    #     rmtree('results/')
    # except BaseException:
    #     pass  # directory doesn't yet exist, no need to clear it
    # os.makedirs("results/")
    os.makedirs('results/' + args.name, exist_ok=True)

    # use imagenet mean,std for normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    N_LABELS = 14  # we are predicting 14 labels

    # load labels
    df = pd.read_csv("nih_labels.csv", index_col=0)

    # define torchvision transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }

    # create train/val dataloaders
    transformed_datasets = {}
    transformed_datasets['train'] = CXR.CXRDataset(
        path_to_images=PATH_TO_IMAGES,
        fold='train',
        transform=data_transforms['train'])
    transformed_datasets['val'] = CXR.CXRDataset(
        path_to_images=PATH_TO_IMAGES,
        fold='val',
        transform=data_transforms['val'])

    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(
        transformed_datasets['train'],
        batch_size=BATCH_SIZE,
        shuffle=True,
        # num_workers=8)
        num_workers=32)
    dataloaders['val'] = torch.utils.data.DataLoader(
        transformed_datasets['val'],
        batch_size=1,
        shuffle=True,
        # num_workers=8)
        num_workers=1)

    # please do not attempt to train without GPU as will take excessively long
    if not use_gpu:
        raise ValueError("Error, requires GPU")

    if args.model == 'vgg19':
        model = torchvision.models.vgg19(pretrained=True)
        model.classifier[6] = nn.Sequential(nn.Linear(4096, N_LABELS), nn.Sigmoid())
    elif args.model == 'vgg19_bn':
        model = torchvision.models.vgg19_bn(pretrained=True)
        model.classifier[6] = nn.Sequential(nn.Linear(4096, N_LABELS), nn.Sigmoid())
    elif args.model == 'vgg16':
        model = torchvision.models.vgg16(pretrained=True)
        model.classifier[6] = nn.Sequential(nn.Linear(4096, N_LABELS), nn.Sigmoid())
    elif args.model == 'vgg16_bn':
        model = torchvision.models.vgg16_bn(pretrained=True)
        model.classifier[6] = nn.Sequential(nn.Linear(4096, N_LABELS), nn.Sigmoid())
    elif args.model == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = nn.Sequential(nn.Linear(model.fc.in_features, N_LABELS), nn.Sigmoid())
    elif args.model == 'resnet34':
        model = torchvision.models.resnet34(pretrained=True)
        model.fc = nn.Sequential(nn.Linear(model.fc.in_features, N_LABELS), nn.Sigmoid())
    elif args.model == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
        model.fc = nn.Sequential(nn.Linear(model.fc.in_features, N_LABELS), nn.Sigmoid())
    elif args.model == 'densenet121':
        model = models.densenet121(weights='DEFAULT')
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, N_LABELS), nn.Sigmoid())

    if (not backbone) and (load_ckpt is not None):
        loaded_state = torch.load(load_ckpt)
        model = loaded_state['model']
        LR = loaded_state['LR']
        NUM_EPOCHS = inm_epochs

    # put model on GPU
    model = model.cuda()

    # define criterion, optimizer for training
    criterion = nn.BCELoss()
    optimizer = optim.SGD(
        filter(
            lambda p: p.requires_grad,
            model.parameters()),
        lr=LR,
        momentum=0.9,
        weight_decay=WEIGHT_DECAY)
    dataset_sizes = {x: len(transformed_datasets[x]) for x in ['train', 'val']}

    # debugging
    # preds, aucs = E.make_pred_multilabel(
    #     args, data_transforms, model, PATH_TO_IMAGES)

    # train model
    if is_train:
        model, best_epoch = train_model(args, model, criterion, optimizer, LR, num_epochs=NUM_EPOCHS,
                                        dataloaders=dataloaders, dataset_sizes=dataset_sizes, weight_decay=WEIGHT_DECAY)

    # get preds and AUCs on test fold
    preds, aucs = E.make_pred_multilabel(
        args, data_transforms, model, PATH_TO_IMAGES)

    return preds, aucs
