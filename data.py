import os
import numpy as np 
from PIL import Image 

num_examples = 5000

def generate_identicon(f):
    # generate random github-like identicons 
    # random matrices are used instead of hashing
    px_color = np.random.randint(0, 256, size=(3))
    bg_color = 240

    pattern = np.random.randint(0, 2, size=(5, 5))
    buffer = np.zeros(shape=(5, 5, 3))
    pattern[:,4] = pattern[:,0]
    pattern[:,3] = pattern[:,1]
    im = [pattern for i in range(3)]
    im = np.stack(im, axis=2)
    im *= px_color
    im[im < 1] = bg_color

    im = Image.fromarray(im.astype('uint8'), mode='RGB')
    im = im.resize((64, 64), Image.NEAREST)
    im.save(f + '.jpg')

if not os.path.exists('./data'):
    os.mkdir('./data')

for i in range(num_examples):
    generate_identicon('./data/id' + str(i).zfill(3))

