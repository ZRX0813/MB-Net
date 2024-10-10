import torch.nn as nn
import torch.utils.data


# Fusion mechanism
class DotProductFusion(nn.Module):
    def __init__(self, input_channels):
        super(DotProductFusion, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x1, x2, x3):
        dot_product1 = self.conv1(x1)
        dot_product2 = self.conv2(x2)
        dot_product3 = self.conv3(x3)
        dot_product_features = dot_product1 + dot_product2 + dot_product3 + x1 + x2 + x3
        return dot_product_features


class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class MB_Net(nn.Module):

    def __init__(self, in_ch=3, out_ch=1):
        super(MB_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.wsum1 = DotProductFusion(64)
        self.wsum2 = DotProductFusion(128)
        self.wsum3 = DotProductFusion(256)
        self.wsum4 = DotProductFusion(512)
        self.wsum5 = DotProductFusion(1024)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, channels_1, channels_2, channels_3):
        # channels_1
        channels_1_e1 = self.Conv1(channels_1)
        channels_1_e2 = self.Maxpool1(channels_1_e1)
        channels_1_e2 = self.Conv2(channels_1_e2)
        channels_1_e3 = self.Maxpool2(channels_1_e2)
        channels_1_e3 = self.Conv3(channels_1_e3)
        channels_1_e4 = self.Maxpool3(channels_1_e3)
        channels_1_e4 = self.Conv4(channels_1_e4)
        channels_1_e5 = self.Maxpool4(channels_1_e4)
        channels_1_e5 = self.Conv5(channels_1_e5)
        # channels_2
        channels_2_e1 = self.Conv1(channels_2)
        channels_2_e2 = self.Maxpool1(channels_2_e1)
        channels_2_e2 = self.Conv2(channels_2_e2)
        channels_2_e3 = self.Maxpool2(channels_2_e2)
        channels_2_e3 = self.Conv3(channels_2_e3)
        channels_2_e4 = self.Maxpool3(channels_2_e3)
        channels_2_e4 = self.Conv4(channels_2_e4)
        channels_2_e5 = self.Maxpool4(channels_2_e4)
        channels_2_e5 = self.Conv5(channels_2_e5)
        # channels_3
        channels_3_e1 = self.Conv1(channels_3)
        channels_3_e2 = self.Maxpool1(channels_3_e1)
        channels_3_e2 = self.Conv2(channels_3_e2)
        channels_3_e3 = self.Maxpool2(channels_3_e2)
        channels_3_e3 = self.Conv3(channels_3_e3)
        channels_3_e4 = self.Maxpool3(channels_3_e3)
        channels_3_e4 = self.Conv4(channels_3_e4)
        channels_3_e5 = self.Maxpool4(channels_3_e4)
        channels_3_e5 = self.Conv5(channels_3_e5)

        add_layer1 = self.wsum1(channels_1_e1, channels_2_e1, channels_3_e1)
        add_layer2 = self.wsum2(channels_1_e2, channels_2_e2, channels_3_e2)
        add_layer3 = self.wsum3(channels_1_e3, channels_2_e3, channels_3_e3)
        add_layer4 = self.wsum4(channels_1_e4, channels_2_e4, channels_3_e4)
        add_layer5 = self.wsum5(channels_1_e5, channels_2_e5, channels_3_e5)

        d5 = self.Up5(add_layer5)
        d5 = torch.cat((add_layer4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((add_layer3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((add_layer2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((add_layer1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        return out


if __name__ == "__main__":
    import torch as t

    channels_1 = t.randn(1, 3, 128, 128)
    channels_2 = t.randn(1, 3, 128, 128)
    channels_3 = t.randn(1, 3, 128, 128)

    net = MB_Net(3, 2)
    out = net(channels_1, channels_2, channels_3)
    print(out.shape)
