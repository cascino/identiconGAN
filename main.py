import torch
import os
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torch.autograd import Variable
from torchvision import transforms
from model import D, G

class IdenticonDataset(Dataset):
    def __init__(self, root_dir):
        self.root = root_dir
        self.tform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(os.listdir(self.root))

    def __getitem__(self, idx):
        file = os.path.dirname(__file__)
        working_dir = os.path.join(file, self.root)
        imname = 'id' + str(idx).zfill(3) + '.jpg'
        impath = os.path.join(working_dir, imname)
        return self.tform(Image.open(impath))

# hyperparameters
epochs = 20
torch.manual_seed(1)
batch_size = 64
use_cuda = torch.cuda.is_available()

dataset = IdenticonDataset('./data')
dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=2)

discriminator = D()
generator = G()

print(generator)
print(discriminator)

# loss(o, t) = - 1/n \sum_i (t[i] log(o[i]) + (1 - t[i]) log(1 - o[i]))
loss = nn.BCELoss(size_average=True)

if use_cuda:
    discriminator.cuda()
    generator.cuda()
    loss.cuda()

optimD = optim.Adam(discriminator.parameters(), lr=0.001, betas=(0.5, 0.999))
optimG = optim.Adam(generator.parameters(), lr=0.001, betas=(0.5, 0.999))

test_noise = torch.Tensor(batch_size, 100, 1, 1).normal_(0, 1)

if use_cuda:
    test_noise = test_noise.cuda()

test_noiseV = Variable(test_noise)

for i in range(epochs):
    for j, data in enumerate(dataloader):
        latent = torch.Tensor(data.size(0), 100, 1, 1)
        label = torch.Tensor(data.size(0), 1, 1, 1)

        if use_cuda:
            latent = latent.cuda()
            label = label.cuda()
            data = data.cuda()
        # input an image, 0|1 if fake|real

        # update discriminator        
        # train on real
        optimD.zero_grad()
        real_label = Variable(label.fill_(1), requires_grad=False)
        real_im = Variable(data, requires_grad=False)

        out = discriminator(real_im)
        loss_real = loss(out, real_label)
        loss_real.backward()

        # train on fake
        noise = Variable(latent.normal_(0, 1), requires_grad=False)
        fake_label = Variable(label.fill_(0), requires_grad=False)

        fake = generator(noise)
        out = discriminator(fake.detach())
        loss_fake = loss(out, fake_label)
        loss_fake.backward()
        optimD.step()

        # train generator
        fake_real_label = Variable(label.fill_(1), requires_grad=False)       

        optimG.zero_grad()
        out = discriminator(fake)
        loss_gen = loss(out, fake_real_label)
        loss_gen.backward()
        optimG.step()

        print('epoch [{}]/[{}]    batch {}    lossD {:.5f}    lossG {:.5f}'.format(
                i, epochs, j, (loss_real.cpu().data[0] + loss_fake.cpu().data[0]), 
                loss_gen.cpu().data[0]))

        if j % 50 == 0:
            out = generator(test_noiseV).cpu().data
            torchvision.utils.save_image(out, './fake.jpg', normalize=True)
            torch.save(discriminator, 'dis.pt')
            torch.save(generator, 'gen.pt')

torch.save(discriminator, 'dis.pt')
torch.save(generator, 'gen.pt')
