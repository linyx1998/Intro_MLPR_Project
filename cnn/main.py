from functions import *
from dim_reduction import *
import torch.utils.data
from torch import nn
from torchvision import transforms, models, datasets
import time
import sys

TRAIN_RATIO = 0.49
BATCH_SIZE = 256
LEARNING_RATE = 0.01
EPOCH_NUM = 200

if __name__ == '__main__':
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5,), (0.5,), (0.5,))
    # ])

    transform_train_shape = transforms.Compose([
        transforms.RandomHorizontalFlip(0.1),
        transforms.RandomVerticalFlip(0.1),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomRotation(0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,), (0.5,))
    ])

    transform_train_color = transforms.Compose([
        transforms.ColorJitter(),
        transforms.RandomGrayscale(),
        transforms.RandomInvert(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,), (0.5,))
    ])

    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    transform_optim = transforms.Compose([
        transforms.ToTensor(),   
        transforms.Normalize(norm_mean, norm_std),
        transforms.RandomHorizontalFlip(),
        transforms.RandomErasing(scale=(0.04, 0.2), ratio=(0.5, 2)),
        transforms.RandomCrop(32, padding=4)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,), (0.5,))
    ])

    train_dataset = datasets.CIFAR10('./cifar10_data', train=True, download=True, 
                    transform=transform_optim)
    test_dataset = datasets.CIFAR10('./cifar10_data', train=False, download=True, 
                    transform=transform_optim)

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(
                    range(0, (int)(len(train_dataset)*TRAIN_RATIO)))
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(
                    range((int)(len(train_dataset)*TRAIN_RATIO), len(train_dataset)))

    train_data_loader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler,
                        batch_size=BATCH_SIZE, num_workers=4)
    valid_data_loader = torch.utils.data.DataLoader(train_dataset, sampler=valid_sampler,
                        batch_size=BATCH_SIZE, num_workers=4)
    test_data_loader = torch.utils.data.DataLoader(test_dataset,
                        batch_size=BATCH_SIZE, num_workers=4)

    model_name = sys.argv[1]

    # choose model
    if model_name == 'vgg11':
        classifier = models.vgg11(pretrained=False)
        classifier.load_state_dict(torch.load('./pretrain_models/vgg11-bbd30ac9.pth'))
    elif model_name == 'vgg13':
        classifier = models.vgg13(pretrained=False)
        classifier.load_state_dict(torch.load('./pretrain_models/vgg13-c768596a.pth'))
    elif model_name == 'vgg16':
        classifier = models.vgg16(pretrained=False)
        classifier.load_state_dict(torch.load('./pretrain_models/vgg16-397923af.pth'))
    elif model_name == 'vgg19':
        classifier = models.vgg19(pretrained=False)
        classifier.load_state_dict(torch.load('./pretrain_models/vgg19-dcbb9e9d.pth'))
    elif model_name == 'resnet50':
        classifier = models.resnet50(pretrained=False)
        classifier.load_state_dict(torch.load('./pretrain_models/resnet50-19c8e357.pth'))
    elif model_name == 'resnet34':
        classifier = models.resnet34(pretrained=False)
        classifier.load_state_dict(torch.load('./pretrain_models/resnet34-333f7ec4.pth'))
    elif model_name == 'resnet18':
        classifier = models.resnet18(pretrained=False)
        classifier.load_state_dict(torch.load('./pretrain_models/resnet18-5c106cde.pth'))
    else:
        print("Wrong Model Selection")
        sys.exit()
    
    # adjust the output num from 1000 to 10
    if model_name.startswith('vgg'):
        classifier.classifier[6] = nn.Linear(4096, 10)
    elif model_name.startswith('resnet'):
        features_num = classifier.fc.in_features
        classifier.fc = nn.Linear(features_num, 10)
    else:
        print("Wrong Model Selection")
        sys.exit()
    
    print("lr:", LEARNING_RATE)
    print("epoches:", EPOCH_NUM)

    train_begin = time.time()
    if sys.argv[2] == 'optim':
        optim_train(classifier, train_data_loader, valid_data_loader, EPOCH_NUM, LEARNING_RATE, model_name)
    else:
        train(classifier, train_data_loader, valid_data_loader, EPOCH_NUM, LEARNING_RATE, model_name)
    train_end = time.time()
    print("trainging time:", train_end - train_begin, "s")

    test(classifier, test_data_loader)

