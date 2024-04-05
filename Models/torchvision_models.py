import torch
import torch.nn as nn
import torchvision.models as models
import Training.model_utils as model_utils


def get_pretrained_dlv3(num_classes: int,
                        in_channels: int, num_layers: int = 50) -> nn.Module:
    if num_layers == 101:
        model = models.segmentation.deeplabv3_resnet101(pretrained=models.segmentation.
                                                        DeepLabV3_ResNet101_Weights.DEFAULT)
    else:
        model = models.segmentation.deeplabv3_resnet50(weights=models.segmentation.
                                                       DeepLabV3_ResNet50_Weights.DEFAULT)

    if in_channels == 1:
        conv1 = model.backbone.conv1
        layer_1_weights = conv1.weight.data
        layer_1_weights_ave = torch.mean(layer_1_weights, dim=1, keepdim=True)
        # layer_1_weights_ave = torch.max(layer_1_weights, dim=1)
        conv1.weight = torch.nn.Parameter(layer_1_weights_ave)
        conv1.in_channels = 1

    elif in_channels != 3:
        conv1 = torch.nn.Conv2d(in_channels, 64, bias=False, kernel_size=7)
        model.backbone.conv1 = conv1

    # new classifier
    new_dlv3_head = models.segmentation.deeplabv3.DeepLabHead(in_channels=2048, num_classes=num_classes)
    model.classifier = new_dlv3_head

    # new auxiliary head
    new_aux_head = models.segmentation.deeplabv3.FCNHead(1024, num_classes)
    model.aux_classifier = new_aux_head

    model_utils.set_model_all_trainable(model)

    return model


def get_pretrained_LRASPP(num_classes: int,
                          in_channels: int) -> nn.Module:
    model = models.segmentation.lraspp_mobilenet_v3_large(weights=models.segmentation.
                                                          LRASPP_MobileNet_V3_Large_Weights.DEFAULT)

    if in_channels == 1:
        l1_conv = model.backbone._modules['0']._modules['0']
        l1_conv_weights = l1_conv.weight.data
        l1_ave_conv_weights = torch.mean(l1_conv_weights, dim=1,
                                         keepdim=True)
        l1_conv.weight = torch.nn.Parameter(l1_ave_conv_weights)
        l1_conv.in_channels = 1

    elif in_channels != 3:
        l1_conv = model.backbone._modules['0']._modules['0']
        new_conv = nn.Conv2d(in_channels, l1_conv.out_channels,
                                                               kernel_size=l1_conv.kernel_size,
                                                               padding=l1_conv.padding,
                                                               stride=l1_conv.stride)
        model.backbone._modules['0']._modules['0'] = new_conv
    low_classifer = model.classifier.low_classifier
    low_in_features = low_classifer.in_channels
    new_low_classifier = nn.Conv2d(low_in_features, num_classes, kernel_size=(1, 1),
                                   stride=(1, 1), bias=False)
    model.classifier.low_classifier = new_low_classifier

    high_classifer = model.classifier.high_classifier
    high_in_features = high_classifer.in_channels
    new_high_classifier = nn.Conv2d(high_in_features, num_classes, kernel_size=(1, 1),
                                    stride=(1, 1), bias=False)
    model.classifier.high_classifier = new_high_classifier
    return model
