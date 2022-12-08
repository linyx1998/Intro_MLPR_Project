from functions import *
from dim_reduction import *
import torch.utils.data
from torch import nn
from torchvision import transforms, models, datasets
import time
import sys

BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCH_NUM = 50

if __name__ == '__main__':
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

    feature_remain_width = (int)(sys.argv[3])
    feature_remain = ((feature_remain_width)**2)*3

    # for r in range(feature_remain.size):
    X_train, y_train, X_test, y_test = read_data(True)
    if sys.argv[2] == 'PCA':
        X_train, y_train, X_test, y_test = pca(feature_remain_width, feature_remain)
    elif sys.argv[2] == 'KernelPCA':
        X_train, y_train, X_test, y_test = kernel_pca(feature_remain_width, feature_remain)
    elif sys.argv[2] == 'LLE':
        X_train, y_train, X_test, y_test = lle(feature_remain_width, feature_remain)
        
    # train_dataset = CustomTensorDataset([X_train, y_train])
    # test_dataset = CustomTensorDataset([X_test, y_test])

    # X_train, y_train, X_test, y_test = read_data(True)

    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(), \
        torch.from_numpy(y_train).long().squeeze())
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_test).float(), \
        torch.from_numpy(y_test).long().squeeze())

    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                        batch_size=BATCH_SIZE, num_workers=4)
    valid_data_loader = torch.utils.data.DataLoader(train_dataset,
                        batch_size=BATCH_SIZE, num_workers=4)
    test_data_loader = torch.utils.data.DataLoader(test_dataset,
                        batch_size=BATCH_SIZE, num_workers=4)

    train_begin = time.time()
    # if sys.argv[2] == 'auto':
    #     auto_train(classifier, train_data_loader, valid_data_loader, EPOCH_NUM, LEARNING_RATE, model_name)
    # else:
    train(classifier, train_data_loader, valid_data_loader, EPOCH_NUM, LEARNING_RATE, model_name)
    train_end = time.time()
    print("trainging time:", train_end - train_begin, "s")

    test(classifier, test_data_loader)

