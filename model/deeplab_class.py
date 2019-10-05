import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch
import numpy as np
affine_par = True

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def outS(i):
    i = int(i)
    i = (i+1)/2
    i = int(np.ceil((i+1)/2.0))
    i = (i+1)/2
    return i

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine = affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine = affine_par)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
        self.bn1 = nn.BatchNorm2d(planes,affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                               padding=padding, bias=False, dilation = dilation)
        self.bn2 = nn.BatchNorm2d(planes,affine = affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine = affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Classifier_Module(nn.Module):

    def __init__(self, dilation_series, padding_series, num_classes, mode=0, latent_vars=5, latent_vars_list=(3, 5, 7, 10), cond_latent_vars=3):
        super(Classifier_Module, self).__init__()
        self.num_scales = len(dilation_series)
        self.mode = mode
        self.latent_vars = latent_vars 
        self.cond_latent_vars = cond_latent_vars 
        self.latent_vars_list = latent_vars_list
        self.num_latent_vars = len(latent_vars_list) 

        if self.mode == 1:
            self.conv2d_list_cls = nn.ModuleList()
            for dilation, padding in zip(dilation_series, padding_series):
                self.conv2d_list_cls.append(nn.Conv2d(2048, num_classes-1, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias = True))

            for m in self.conv2d_list_cls:
                m.weight.data.normal_(0, 0.01)
        elif self.mode == 3:
            self.conv2d_list_lv = nn.ModuleList()
            for dilation, padding in zip(dilation_series, padding_series):
                self.conv2d_list_lv.append(nn.Conv2d(2048, self.latent_vars+1, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias = True))

            for m in self.conv2d_list_lv:
                m.weight.data.normal_(0, 0.01)
        #TODO: find out if ModuleList of ModuleList is possible and rewrite the code in that case
        elif self.mode == 4:
            self.conv2d_list_lv = nn.ModuleList()
            for lv in self.latent_vars_list:
                for dilation, padding in zip(dilation_series, padding_series):
                    self.conv2d_list_lv.append(nn.Conv2d(2048, lv+1, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias = True))

            for m in self.conv2d_list_lv:
                m.weight.data.normal_(0, 0.01)
        elif self.mode == 5:
            self.conv2d_list_lv = nn.ModuleList()
            for dilation, padding in zip(dilation_series, padding_series):
                self.conv2d_list_lv.append(nn.Conv2d(2048, self.latent_vars+1, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias = True))

            for m in self.conv2d_list_lv:
                m.weight.data.normal_(0, 0.01)

            self.conv2d_list_cond_lv = nn.ModuleList()
            for dilation, padding in zip(dilation_series, padding_series):
                self.conv2d_list_cond_lv.append(nn.Conv2d(2048, (self.latent_vars+1)*self.cond_latent_vars, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias = True))

            for m in self.conv2d_list_cond_lv:
                m.weight.data.normal_(0, 0.01)

        if self.mode == 6:
            self.cls_layer = nn.Linear(2048, num_classes-1, bias=True)
            self.cls_layer.weight.data.normal_(0, 0.01)
        else:
            self.conv2d_list_seg = nn.ModuleList()
            for dilation, padding in zip(dilation_series, padding_series):
                self.conv2d_list_seg.append(nn.Conv2d(2048, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias = True))

            for m in self.conv2d_list_seg:
                m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        #print('shape of x in forward')
        #print(x.size())
        #print('called forward')
        if self.mode == 6:
            avg_x = F.max_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0)
            #print('shape of x after avg pool')
            #print(avg_x.size())
            return self.cls_layer(torch.squeeze(torch.squeeze(avg_x, 3), 2))
        else:
            seg_act = self.conv2d_list_seg[0](x)
            for i in range(1, len(self.conv2d_list_seg)):
                seg_act += self.conv2d_list_seg[i](x)
        if self.mode == 0:
            return seg_act
        elif self.mode == 1:
            cls_act = self.conv2d_list_cls[0](x)
            for i in range(1, len(self.conv2d_list_cls)):
                cls_act += self.conv2d_list_cls[i](x)
            cls_act = F.max_pool2d(cls_act, kernel_size=(cls_act.size(2), cls_act.size(3)), padding=0)
            return seg_act, cls_act
        elif self.mode == 2:
            cls_act = F.max_pool2d(seg_act[:,1:,:,:], kernel_size=(seg_act.size(2), seg_act.size(3)), padding=0)
            return seg_act, cls_act
        elif self.mode == 3:
            lv_act = self.conv2d_list_lv[0](x)
            for i in range(1, len(self.conv2d_list_lv)):
                lv_act += self.conv2d_list_lv[i](x)
            return seg_act, lv_act
        elif self.mode == 4:
            lv_act_list = []
            #print(self.num_latent_vars)
            for lv_idx in range(self.num_latent_vars):
                lv_act = 0
                for i in range(self.num_scales):
                    lv_act += self.conv2d_list_lv[lv_idx * self.num_latent_vars + i](x)
                #print('appending to lv_act_list')
                lv_act_list.append(lv_act)
            return seg_act, lv_act_list
        elif self.mode == 5:
            lv_act = self.conv2d_list_lv[0](x)
            for i in range(1, len(self.conv2d_list_lv)):
                lv_act += self.conv2d_list_lv[i](x)
            cond_lv_act = self.conv2d_list_cond_lv[0](x)
            for i in range(1, len(self.conv2d_list_cond_lv)):
                cond_lv_act += self.conv2d_list_cond_lv[i](x)
            return seg_act, lv_act, cond_lv_act



class ResNet(nn.Module):
    #TODO: list as default argument is boilerplate, consider exchanging for non mutable type
    def __init__(self, block, layers, num_classes, mode=0, latent_vars=5, latent_vars_list=(3, 5, 7, 10), cond_latent_vars=3):
        self.latent_vars = latent_vars
        self.cond_latent_vars = cond_latent_vars
        self.latent_vars_list = latent_vars_list
	self.mode = mode
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.layer5 = self._make_pred_layer(Classifier_Module, [6,12,18,24],[6,12,18,24],num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #        for i in m.parameters():
        #            i.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)
    def _make_pred_layer(self,block, dilation_series, padding_series,num_classes):
        return block(dilation_series, padding_series, num_classes, self.mode, self.latent_vars, self.latent_vars_list, self.cond_latent_vars)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.layer5(x)
        #if self.mode == 0:
        #    x = self.layer5(x)
        #    return x
        #elif self.mode in [1, 2, 3, 4]:
        #    x, y = self.layer5(x)
        #    return x, y
        #elif self.mode in [5]:
        #    x, y, z = self.layer5(x)
        #    return x, y, z

    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for 
        the last classification layer. Note that for each batchnorm layer, 
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return 
        any batchnorm parameter
        """
        b = []

        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

    
        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj+=1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.layer5.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i
            


    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': args.learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10*args.learning_rate}] 


def Res_Deeplab(num_classes=21, mode=0, latent_vars=5, latent_vars_list=(3, 5, 7, 10), cond_latent_vars=3):
    model = ResNet(Bottleneck,[3, 4, 23, 3], num_classes, mode, latent_vars, latent_vars_list, cond_latent_vars)
    return model

class Net(nn.Module):
    def __init__(self, fc6_dilation = 1):
        super(Net, self).__init__()

        self.conv1_1 = nn.Conv2d(3,64,3,padding = 1)
        self.conv1_2 = nn.Conv2d(64,64,3,padding = 1)
        self.pool1 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding=1)
        self.conv2_1 = nn.Conv2d(64,128,3,padding = 1)
        self.conv2_2 = nn.Conv2d(128,128,3,padding = 1)
        self.pool2 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding=1)
        self.conv3_1 = nn.Conv2d(128,256,3,padding = 1)
        self.conv3_2 = nn.Conv2d(256,256,3,padding = 1)
        self.conv3_3 = nn.Conv2d(256,256,3,padding = 1)
        self.pool3 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding=1)
        self.conv4_1 = nn.Conv2d(256,512,3,padding = 1)
        self.conv4_2 = nn.Conv2d(512,512,3,padding = 1)
        self.conv4_3 = nn.Conv2d(512,512,3,padding = 1)
        self.pool4 = nn.MaxPool2d(kernel_size = 3, stride = 1, padding=1)
        self.conv5_1 = nn.Conv2d(512,512,3,padding = 2, dilation = 2)
        self.conv5_2 = nn.Conv2d(512,512,3,padding = 2, dilation = 2)
        self.conv5_3 = nn.Conv2d(512,512,3,padding = 2, dilation = 2)
        self.pool5 = nn.MaxPool2d(kernel_size = 3, stride = 1, padding=1)
        self.pool5a = nn.AvgPool2d(kernel_size = 3, stride = 1, padding=1)

        self.fc6 = nn.Conv2d(512,1024, 3, padding = fc6_dilation, dilation = fc6_dilation)

        self.drop6 = nn.Dropout2d(p=0.5)
        self.fc7 = nn.Conv2d(1024,1024,1)

        self.normalize = Normalize()

        return

    def forward(self, x):
        return self.forward_as_dict(x)['conv5fc']

    def forward(self, x):

        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.pool3(x)

        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))

        x = self.pool4(x)

        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))

        x = F.relu(self.fc6(x))
        x = self.drop6(x)
        x = F.relu(self.fc7(x))

        return x

    def train(self, mode=True):

        super().train(mode)

        for layer in self.not_training:

            if isinstance(layer, torch.nn.Conv2d):

                layer.weight.requires_grad = False
                layer.bias.requires_grad = False

def VGG_Deeplab(num_classes=21, mode=0, latent_vars=5, latent_vars_list=(3, 5, 7, 10), cond_latent_vars=3):
    model = VGG(Bottleneck,[3, 4, 23, 3], num_classes, mode, latent_vars, latent_vars_list, cond_latent_vars)
    return model
