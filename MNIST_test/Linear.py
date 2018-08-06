import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class SimpleNet(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(SimpleNet, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, out_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class ActivationNet(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(ActivationNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(n_hidden_2, out_dim)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class BatchNet(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(BatchNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1),
            nn.BatchNorm1d(n_hidden_1),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.BatchNorm1d(n_hidden_2),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(n_hidden_2, out_dim)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class Normalize(object):
    def __call__(self, sample):
        img_tensor = sample
        img_mean, img_std = [], []

        if img_tensor.size(2) == 3:
            for i in range(3):
                img_mean.append(torch.mean(img_tensor[i]))
                img_std.append(torch.std(img_tensor[i]) + 1e-5)
        else:
            img_mean.append(torch.mean(img_tensor))
            img_std.append(torch.std(img_tensor) + 1e-5)
        img_tensor = transforms.Normalize(img_mean, img_std)(img_tensor)
        return img_tensor


def modify_learning(base_lr, epoch, optimizer):
    lr = base_lr * (0.1 ** (epoch // 2))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def main():
    # Hyperparameters
    batch_size = 128
    learning_rate = 1e-2
    num_eoches = 5

    norm = Normalize()
    data_tf = transforms.Compose(
        [transforms.ToTensor(),
         norm])

    train_dataset = datasets.MNIST(root='./data', train=True, transform=data_tf, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=data_tf)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = BatchNet(28 * 28, 300, 100, 10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    for epoch in range(num_eoches):
        for step, (data, target) in enumerate(train_loader):

            new_lr = modify_learning(learning_rate, epoch, optimizer)

            input_val = Variable(data)
            target_val = Variable(target)

            output = model(input_val)
            loss = criterion(output, target_val)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                print('epoch : {} , step : {}, lr : {} , loss : {}'.format(epoch, step, new_lr, loss.data[0]))

    print('##################开始测试网络#####################')
    model.eval()
    eval_loss = 0
    eval_acc = 0
    for data, target in test_loader:
        data = Variable(data, volatile=True)
        target = Variable(target, volatile=True)

        predict = model(data)
        loss = criterion(predict, target)
        eval_loss += loss.data[0] * target.size(0)
        _, pred = torch.max(predict, 1)
        num_correct = (pred == target).sum()
        eval_acc += num_correct.data[0]
    print('Test Loss: {:.6f}, Acc : {:.6f}'.format(eval_loss / len(test_dataset), eval_acc / len(test_dataset)))


if __name__ == '__main__':
    main()
