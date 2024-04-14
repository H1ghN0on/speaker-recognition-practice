import torch.nn as nn
import os

from utils import *

# Select hyperparameters

# Acoustic features
n_mels            = 40                                   # number of mel filters in bank filters
log_input         = True                                 # logarithm of features by level

# Neural network archtecture
layers            = [3, 4, 6, 3]                         # number of ResNet blocks in different level of frame level
activation        = nn.ReLU                              # activation function used in ResNet blocks
num_filters       = [32, 64, 128, 256]                   # number of filters of ResNet blocks in different level of frame level
encoder_type      = 'SP'                                 # type of statistic pooling layer ('SP'  – classical statistic pooling 
                                                         # layer and 'ASP' – attentive statistic pooling)
nOut              = 512                                  # embedding size

# Loss function for angular losses
margin            = 0.35                                 # margin parameter
scale             = 32.0                                 # scale parameter

# Train dataloader
max_frames_train  = 200                                  # number of frame to train
train_path        = '../data/voxceleb1_dev/wav'          # path to train wav files
batch_size_train  = 128                                   # batch size to train
pin_memory        = False                                # pin memory
num_workers_train = 5                                    # number of workers to train
shuffle           = True                                 # shuffling of training examples

# Validation dataloader
max_frames_val    = 1000                                 # number of frame to validate
val_path          = '../data/voxceleb1_dev/wav'          # path to val wav files
batch_size_val    = 128                                   # batch size to validate
num_workers_val   = 5                                    # number of workers to validate

# Test dataloader
max_frames_test   = 1000                                 # number of frame to test
test_path         = '../data/voxceleb1_dev/wav'          # path to val wav files
batch_size_test   = 128                                   # batch size to test
num_workers_test  = 5                                    # number of workers to test

# Optimizer
lr                = 2.5                                  # learning rate value
weight_decay      = 0                                    # weight decay value

# Scheduler32
val_interval      = 5                                    # frequency of validation step
max_epoch         = 40                                   # number of epoches

# Augmentation
musan_path        = '../data/musan_split'                # path to splitted SLR17 dataset
rir_path          = '../data/RIRS_NOISES/simulated_rirs' # path to SLR28 dataset

def separate_train_data():
    train_list = [] # данные для обучения
    val_list = [] # данные для валидации
    test_list = [] # данные для тестирования

    # Файл с разделением данных для обучения, валидации и тестирования
    with open('../data/voxceleb1_test/iden_split.txt', 'r') as f:
        lines = f.readlines()

    black_list = os.listdir('../data/voxceleb1_test/wav')
    num_train_spk = []

    for line in lines:
        line = line.strip().split(' ')
        spk_id = line[1].split('/')[0]

        if not (spk_id in black_list):
            num_train_spk.append(spk_id)
        else:
            continue
        
        if line[0] == '1':
            train_list.append(' '.join([spk_id, line[1]]))

        elif line[0] == '2':
            val_list.append(' '.join([spk_id, line[1]]))                                  

        elif line[0] == '3':
            test_list.append(' '.join([spk_id, line[1]]))                   

    return train_list, val_list, test_list, len(set(num_train_spk))

def init_model_and_helpers(train_list, val_list, num_train_spk):
    model = ResNet(
        BasicBlock, \
        layers=layers, \
        activation=activation, \
        num_filters=num_filters, \
        nOut=nOut, \
        encoder_type=encoder_type, \
        n_mels=n_mels, \
        log_input=log_input)
    
    trainfunc = AAMSoftmaxLoss(
        nOut=nOut, \
        nClasses=num_train_spk, \
        margin=margin)
    
    main_model = MainModel(model, trainfunc).cuda()

    train_dataset = train_dataset_loader(
        train_list=train_list, \
        max_frames=max_frames_train, \
        train_path=train_path, \
        augment=True, 
        musan_path=musan_path, 
        rir_path=rir_path)
    
    train_loader = DataLoader(
        train_dataset, \
        batch_size=batch_size_train, \
        pin_memory=pin_memory, \
        num_workers=num_workers_train, \
        shuffle=shuffle)
    
    val_dataset = test_dataset_loader(
        test_list=val_list, \
        max_frames=max_frames_val, \
        test_path=val_path)

    val_loader = DataLoader(
        val_dataset, \
        batch_size=batch_size_val, \
        num_workers=num_workers_val)
    
    # Initialize optimizer and scheduler
    optimizer = SGDOptimizer(
        main_model.parameters(), \
        lr=lr, \
        weight_decay=weight_decay)
    
    scheduler = OneCycleLRScheduler(
        optimizer, \
        pct_start=0.30, \
        cycle_momentum=False, \
        max_lr=lr, \
        div_factor=20, \
        final_div_factor=10000, \
        total_steps= \
        max_epoch * len(train_loader))
    
    return main_model, train_loader, val_loader, optimizer, scheduler

def train(main_model, train_loader, val_loader, optimizer, scheduler):
    start_epoch = 0
    checkpoint_flag = True

    if checkpoint_flag:
        start_epoch = loadParameters(main_model, optimizer, scheduler, path='../data/lab3_models/lab3_model_0003.pth')
        start_epoch = start_epoch + 1

    # Train model
    for num_epoch in range(start_epoch, max_epoch):
        train_loss, train_top1 = train_network(
            train_loader, \
            main_model, \
            optimizer, \
            scheduler, \
            num_epoch, \
            verbose=True)
        if (num_epoch + 1) % val_interval == 0:
            _, val_top1 = test_network(val_loader, main_model)
            
        saveParameters(main_model, optimizer, scheduler, num_epoch, path='../data/lab3_models')

def main():
    train_list, val_list, test_list, num_train_spk = separate_train_data()

    main_model, train_loader, val_loader, optimizer, scheduler = init_model_and_helpers(train_list, val_list, num_train_spk)
    train(main_model, train_loader, val_loader, optimizer, scheduler)

    

if __name__ ==  '__main__':
    main()