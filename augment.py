import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html
from util.util import save_image
import tqdm
import time

opt = TestOptions().parse()
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt, phase='generate', pairList = opt.genPairLst, dset='train')
dataset = data_loader.load_data()
model = create_model(opt)

print(opt.how_many)
print(len(dataset))

model = model.eval()

# test
for i, data in enumerate(dataset):
    model.set_input(data)
    model.test()
    model.save_fake()
