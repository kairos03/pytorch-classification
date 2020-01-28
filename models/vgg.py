"""Impelementation of Very Deep Convolutional Networks for Large-Scale Image Recognition(https://arxiv.org/abs/1409.1556) a.k.a VGG Net"""

import torch
import torch.nn as nn

# VGG 모듈
class VGG(nn.modules):
    def __init__(self, features, num_classes=1000, init_weights=True):
        """
        VGG Model
        """
        super(VGG, self).__init__()
        self.features = features                            # make_layers 함수로 부터 featrue 모듈을 생성하여 입력 받는다.
        self.avg_pool = nn.AdaptiveAvgPool2d((7, 7))        # Conv모듈의 마지막 avg pool은 spartial 크기를 7x7로 맞춰야 하기 때문에 adaptive pool을 이용한다.
        self.classifier = nn.Sequential(                    # VGG에서 공통으로 사용되는 classifier를 정의한다.
            nn.Linear(7 * 7 * 512, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        if init_weights:                                    # 모델의 weights를 _initialize_weights 함수로 초기화 한다.
            self._initialize_weights()
    
    def forward(self, x):
        x = self.features(x)                                # 입력 이미지로 features 먼저 추출
        x = self.avg_pool(x)                                # spatial size 7x7로 맞추기 위해 adaptive pool 통과
        x = torch.flatten(x, 1)                             # FC에 넣기 위해 flatten
        x = self.classifier(x)                              # classifier로 최종 통과
        return x
    
    def _initialize_weights(self):
        """
        생성된 모델의 weights를 모듈 별로 초기화
        """
        for m in self.modules():                            # nn.modules에서 제공하는 현 class안에 있는 모든 모듈을 불러오는 함수
            if isinstance(m, nn.Conv2d):                    # convolution은 kaiming_normal을 이용하여 초기화
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):                  # linear는 mean=0, std=0.01로 초기화
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.append(nn.Conv2d(in_channels, v, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(inplace=True))
            in_channels = v
    return nn.Sequential(*layers)

# VGG논문에서 제안하는 feature extractor의 구성
cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, cfg, pretrained_path=None, **kwargs):
    """
    config를 입력 받아 전체 VGG네트워크 생성
    """
    if pretrained_path:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg]), **kwargs)
    if pretrained_path:
        model.load_state_dict(torch.load(pretrained_path))    # pretrained_path에 있는 모델 로딩
    return model


def vgg11(**kwargs):
    return _vgg('vgg11', 'A', None, **kwargs)


def vgg13(**kwargs):
    return _vgg('vgg13', 'B', None, **kwargs)


def vgg16(**kwargs):
    return _vgg('vgg16', 'D', None, **kwargs)


def vgg19(**kwargs):
    return _vgg('vgg19', 'E', None, **kwargs)