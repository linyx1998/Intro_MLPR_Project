from torch import nn, optim
import torch.utils.data
from torch.optim.lr_scheduler import StepLR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, train_data_loader, valid_data_loader, epoch_num, lr, model_name):
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
    model.to(device)

    for epoch in range(epoch_num):
        # train process
        model.train()
        train_acc = 0
        train_loss = 0
        for step, (x, y_true) in enumerate(train_data_loader):
            x = x.to(device)
            y_true = y_true.to(device)
            optimizer.zero_grad()

            y_pred = model(x)
            # print(y_pred.shape)
            train_acc += (y_pred.max(1)[1] == y_true).float().mean().item()
            loss = criterion(y_pred, y_true)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()
        train_acc = train_acc/len(train_data_loader)
        train_loss = train_loss/len(train_data_loader)

        # eval process
        model.eval()
        eval_acc = 0.0
        eval_loss = 0.0
        for step, (x, y_true) in enumerate(valid_data_loader):
            x = x.to(device)
            y_true = y_true.to(device)
            y_pred = model(x)

            eval_acc += (y_pred.max(1)[1] == y_true).float().mean().item()
            loss = criterion(y_pred, y_true)
            eval_loss += loss.item()
        eval_acc = eval_acc/len(valid_data_loader)
        eval_loss = eval_loss/len(valid_data_loader)

        print("epoch "+str(epoch), "train acc = "+str(train_acc), "train loss = "+str(train_loss))
        print("epoch "+str(epoch), "eval acc = "+str(eval_acc), "eval loss = "+str(eval_loss))
        print()

    torch.save(model, './trained_models/'+model_name+'_'+str(epoch_num)+'.pth')


def auto_train(model, train_data_loader, valid_data_loader, epoch_num, lr, model_name):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    model.to(device)
    best_weights = model.state_dict()
    best_eval_acc = 0.0

    for epoch in range(epoch_num):
        # train process
        model.train()
        train_acc = 0
        train_loss = 0
        for step, (x, y_true) in enumerate(train_data_loader):
            x = x.to(device)
            y_true = y_true.to(device)
            optimizer.zero_grad()

            y_pred = model(x)
            train_acc += (y_pred.max(1)[1] == y_true).float().mean().item()
            loss = criterion(y_pred, y_true)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()
        train_acc = train_acc/len(train_data_loader)
        train_loss = train_loss/len(train_data_loader)

        # eval process
        model.eval()
        eval_acc = 0.0
        eval_loss = 0.0
        for step, (x, y_true) in enumerate(valid_data_loader):
            x = x.to(device)
            y_true = y_true.to(device)
            y_pred = model(x)

            eval_acc += (y_pred.max(1)[1] == y_true).float().mean().item()
            loss = criterion(y_pred, y_true)
            eval_loss += loss.item()
        eval_acc = eval_acc/len(valid_data_loader)
        eval_loss = eval_loss/len(valid_data_loader)

        print("epoch "+str(epoch), "train acc = "+str(train_acc), "train loss = "+str(train_loss))
        print("epoch "+str(epoch), "eval acc = "+str(eval_acc), "eval loss = "+str(eval_loss))
        print()

        if eval_acc >= best_eval_acc:
            best_weights = model.state_dict()

    model.load_state_dict(best_weights)
    torch.save(model, '../trained_models/'+model_name+'_auto.pth')


def test(model, test_data_loader):
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    # test process
    model.eval()
    test_acc = 0.0
    test_loss = 0.0
    for step, (x, y_true) in enumerate(test_data_loader):
        x = x.to(device)
        y_true = y_true.to(device)
        y_pred = model(x)

        test_acc += (y_pred.max(1)[1] == y_true).float().mean().item()
        loss = criterion(y_pred, y_true)
        test_loss += loss.item()
    test_acc = test_acc/len(test_data_loader)
    test_loss = test_loss/len(test_data_loader)

    print("test acc = "+str(test_acc), "test loss = "+str(test_loss))
