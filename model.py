import torch.nn as nn


# Defining your CNN model
# We have defined the baseline model
class baseline_Net(nn.Module):

    def __init__(self, classes):
        super(baseline_Net, self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(128, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.b4 = nn.Sequential(
            nn.MaxPool2d((3, 3)),
            nn.Conv2d(128, 256, 3, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Sequential(
            nn.Linear(256, 1024),
            nn.Dropout(),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, classes),
        )

    def forward(self, x):
        out1 = self.b2(self.b1(x))
        out2 = self.b4(self.b3(out1))
        out_avg = self.avg_pool(out2)
        out_flat = out_avg.view(-1, 256)
        out4 = self.fc2(self.fc1(out_flat))

        return out4


# Custom CNN model
class custom_Net(nn.Module):

    def __init__(self, classes):
        super(custom_Net, self).__init__()

        self.b1 = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True)
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(128, 256, 3),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.avg_pool1 = nn.AdaptiveAvgPool2d((107, 107))
        self.b4 = nn.Sequential(
            nn.Conv2d(256, 512, 3),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.b5 = nn.Sequential(
            nn.MaxPool2d((3, 3)),
            nn.Conv2d(512, 1024, 3),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(inplace=True)
        )
        self.b6 = nn.Sequential(
            nn.MaxPool2d((3, 3)),
            nn.Conv2d(1024, 256, 3),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.avg_pool2 = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Sequential(
            nn.Linear(256, 1024),
            nn.Dropout(),
            nn.LeakyReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Dropout(),
            nn.LeakyReLU(inplace=True)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(512, 256),
        )
        self.fc4 = nn.Sequential(
            nn.Linear(256, classes),
        )

    def forward(self, x):
        out = self.b1(x)
        out = self.b2(out)
        out = self.b3(out)
        out = self.avg_pool1(out)
        out = self.b4(out)
        out = self.b5(out)
        out = self.b6(out)
        out = self.avg_pool2(out)
        out = out.view(-1, 256)

        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)

        return out

