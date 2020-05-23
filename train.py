import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer

opt = TrainOptions().parse()

train_data_loader = CreateDataLoader(opt, phase = 'train', pairList = opt.trainPairLst, dset = 'train')
train_dataset = train_data_loader.load_data()
train_dataset_size = len(train_data_loader)
print('#training images = %d' % train_dataset_size)

test_data_loader = CreateDataLoader(opt, phase = 'train', pairList = opt.testPairLst, dset = 'test')
test_dataset = test_data_loader.load_data()
test_dataset_size = len(test_data_loader)
print('#test images = %d' % test_dataset_size)

nThreads_save = opt.nThreads
serial_batches_save = opt.serial_batches
no_flip_save = opt.no_flip

opt.nThreads = 1
opt.serial_batches = True
opt.no_flip = True

generate_loader = CreateDataLoader(opt, phase = 'generate', pairList = opt.genPairLst, dset = 'test')
generate_dataset = generate_loader.load_data()
generate_dataset_size = len(generate_loader)
print('#generate images = %d' % generate_dataset_size)

opt.nThreads = nThreads_save
opt.serial_batches = serial_batches_save
opt.no_flip = no_flip_save

model = create_model(opt)
visualizer = Visualizer(opt)

phases = ['train', 'validate']
datasets = {}
datasets[phases[0]] = train_dataset
datasets[phases[1]] = test_dataset
epoch_iter = {phases[0]:0, phases[1]:0}
total_steps = {phases[0]:0, phases[1]:0}

print('displaying initial visuals before training')
model = model.eval()
for _, data in enumerate(generate_dataset):
    model.set_input(data)
    model.test()
    model.save_visuals(0)

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()

    for phase in phases:

        epoch_iter[phase] = 0
        if phase == 'train':
            model.train()
        else:
            model.eval()

        for batch, data in enumerate(datasets[phase]):
            if batch == 2:
                break
            iter_start_time = time.time()
            visualizer.reset()
            total_steps[phase] += opt.batchSize
            epoch_iter[phase] += opt.batchSize
            model.set_input(data)
            model.optimize_parameters(phase)

            errors = model.get_current_errors()

            if total_steps[phase] % opt.print_freq == 0:
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(phase ,epoch, epoch_iter[phase], errors, t)

    if epoch % opt.save_epoch_freq == 0 or epoch == 1:
        print('saving the model at the end of epoch %d, iters %d' %
            (epoch, total_steps[phase]))
        model.save(epoch)

    if epoch % opt.display_epoch_freq == 0 or epoch == 1:
        print('displaying visuals at the end of epoch %d' % epoch)
        model = model.eval()
        for i, data in enumerate(generate_dataset):
            model.set_input(data)
            model.test()
            model.save_visuals(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    model.update_learning_rate()

visualizer.close()
