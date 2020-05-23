import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html
import time

opt = TestOptions().parse()
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt, phase = 'generate', pairList = opt.genPairLst, dset = 'test')
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
# create website

print(len(dataset))

model = model.eval()
print(model.training)

opt.how_many = 999999
# test
for i, data in enumerate(dataset):
    print(' process %d/%d img ..'%(i,opt.how_many))
    if i >= opt.how_many:
        break
    model.set_input(data)
    startTime = time.time()
    model.test()
    endTime = time.time()
    print(endTime-startTime)
    model.save_visuals()

visualizer.close()



