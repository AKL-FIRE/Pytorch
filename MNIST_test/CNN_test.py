from torch import nn
import torch.nn.init as init


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        layer1 = nn.Sequential()
        layer1.add_module('conv1', nn.Conv2d(3, 32, 3, 1, padding=1))

        layer1.add_module('relu1', nn.ReLU(True))
        layer1.add_module('pool1', nn.MaxPool2d(2, 2))
        self.layer1 = layer1

        layer2 = nn.Sequential()
        layer2.add_module('conv2', nn.Conv2d(32, 64, 3, 1, padding=1))
        layer2.add_module('relu2', nn.ReLU(True))
        layer2.add_module('pool2', nn.MaxPool2d(2, 2))
        self.layer2 = layer2

        layer3 = nn.Sequential()
        layer3.add_module('conv3', nn.Conv2d(64, 128, 3, 1, padding=1))
        layer3.add_module('relu3', nn.ReLU(True))
        layer3.add_module('pool3', nn.MaxPool2d(2, 2))
        self.layer3 = layer3

        layer4 = nn.Sequential()
        layer4.add_module('fc1', nn.Linear(2048, 512))
        layer4.add_module('fc_relu1', nn.ReLU(True))
        layer4.add_module('fc2', nn.Linear(512, 64))
        layer4.add_module('fc_relu2', nn.ReLU(True))
        layer4.add_module('fc3', nn.Linear(64, 10))
        self.layer4 = layer4

    def forward(self, x):
        conv1 = self.layer1(x)
        conv2 = self.layer2(conv1)
        conv3 = self.layer3(conv2)
        fc_input = conv3.view(conv3.size(0), -1)
        fc_out = self.layer4(fc_input)
        return fc_out


model = SimpleCNN()
#print(model)

# 提取出网络前两层
new_model = nn.Sequential(*list(model.children())[:2])
print(new_model)

# 提取出网络中的卷积部分
conv_model = nn.Sequential()
for layer in model.named_modules():
    if isinstance(layer[1], nn.Conv2d):
        conv_model.add_module(layer[0], layer[1])
print(conv_model)

# 提取出网络的参数部分 name[0]名字   name[1]具体的参数，Variable
for name in model.named_parameters():
    print(name[0])

# 对权重进行初始化
for m in model.modules():
    if isinstance(m, nn.Conv2d):
        init.normal(m.weight.data)
        init.xavier_normal(m.weight.data)
        init.kaiming_normal(m.weight.data)
        m.bias.data.fill_(0)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_()
