import paddle.nn as nn
import paddle.nn.functional as F


class Mb_Tiny(nn.Layer):

    def __init__(self, num_classes=2):
        super(Mb_Tiny, self).__init__()
        self.base_channel = 8 * 2

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2D(inp, oup, 3, stride, 1, bias_attr=None),
                nn.BatchNorm2D(oup),
                nn.ReLU()
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2D(inp, inp, 3, stride, 1, groups=inp, bias_attr=None),
                nn.BatchNorm2D(inp),
                nn.ReLU(),

                nn.Conv2D(inp, oup, 1, 1, 0, bias_attr=None),
                nn.BatchNorm2D(oup),
                nn.ReLU(),
            )

        self.model = nn.Sequential(
            conv_bn(3, self.base_channel, 2),  # 160*120
            conv_dw(self.base_channel, self.base_channel * 2, 1),
            conv_dw(self.base_channel * 2, self.base_channel * 2, 2),  # 80*60
            conv_dw(self.base_channel * 2, self.base_channel * 2, 1),
            conv_dw(self.base_channel * 2, self.base_channel * 4, 2),  # 40*30
            conv_dw(self.base_channel * 4, self.base_channel * 4, 1),
            conv_dw(self.base_channel * 4, self.base_channel * 4, 1),
            conv_dw(self.base_channel * 4, self.base_channel * 4, 1),
            conv_dw(self.base_channel * 4, self.base_channel * 8, 2),  # 20*15
            conv_dw(self.base_channel * 8, self.base_channel * 8, 1),
            conv_dw(self.base_channel * 8, self.base_channel * 8, 1),
            conv_dw(self.base_channel * 8, self.base_channel * 16, 2),  # 10*8
            conv_dw(self.base_channel * 16, self.base_channel * 16, 1)
        )
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = F.avg_pool2d(x, 7)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x
