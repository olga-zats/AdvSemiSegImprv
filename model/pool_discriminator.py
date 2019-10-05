import torch.nn as nn
import torch.nn.functional as F
import torch

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False


class FCDiscriminator(nn.Module):

	def __init__(self, num_classes, kernel_size, ndf = 64):
		super(FCDiscriminator, self).__init__()

                self.pool = nn.AvgPool2d(kernel_size)
		self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
		self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
		self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
		self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
		self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		#self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
		#self.sigmoid = nn.Sigmoid()


	def forward(self, x):
                x = self.pool(x)
                _, _, h, w = x.size()
                
                # upsampling for pooled maps
                if h == 1 and w == 1:
                    x = x.repeat(1, 1, 32, 32)

                if h == 3 and w == 3:
                    x = x.repeat(1, 1, 11, 11)

		x = self.conv1(x)
		x = self.leaky_relu(x)
		x = self.conv2(x)
		x = self.leaky_relu(x)
		x = self.conv3(x)
		x = self.leaky_relu(x)
		x = self.conv4(x)
		x = self.leaky_relu(x)
		x = self.classifier(x)
		#x = self.up_sample(x)
		#x = self.sigmoid(x) 

		return x
