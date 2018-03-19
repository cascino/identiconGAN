import torch
import torchvision
from PIL import Image
from torch.autograd import Variable
from model import G

use_cuda = torch.cuda.is_available()
batch_size = 64

generator = torch.load('gen.pt').cpu()
test_noise = torch.Tensor(batch_size, 100, 1, 1).normal_(0, 1)

if use_cuda:
    generator.cuda()
    test_noise = test_noise.cuda()

sample = generator(Variable(test_noise))
torchvision.utils.save_image(sample.cpu().data, 'fake1.jpg', normalize=True)

