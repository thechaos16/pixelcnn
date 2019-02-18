
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim, backends
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from torchvision.datasets.mnist import read_image_file, read_label_file
from models import PixelCNN, GatedPixelCNN


backends.cudnn.benchmark = True


def one_hot_encoder(batch, classes):
    y_vec = np.zeros((len(batch), classes), dtype=np.float)
    for i, label in enumerate(batch):
        y_vec[i, batch[i]] = 1.0
    return torch.Tensor(y_vec)


# supported by dataset owner
def data_loader(root_path, train=True):
    if train:
        data_dict = {
            'train': {
                'data': read_image_file(os.path.join(root_path, 'train', 'train-images-idx3-ubyte')),
                'label': read_label_file(os.path.join(root_path, 'train', 'train-labels-idx1-ubyte'))},
            'test': {
                'data': read_image_file(os.path.join(root_path, 'test', 't10k-images-idx3-ubyte')),
                'label': read_label_file(os.path.join(root_path, 'test', 't10k-labels-idx1-ubyte'))

            }
        }
    else:
        data_dict = {
            'data': read_image_file(os.path.join(root_path, 'test', 't10k-images-idx3-ubyte')),
            'label': None
        }
    return data_dict


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--model', type=str, default='./models')
    args.add_argument('--epochs', type=int, default=100)
    args.add_argument('--batch', type=int, default=64)
    args.add_argument('--sample', type=int, default=14)
    args.add_argument('--output', type=str, default='./outputs')
    args.add_argument('--lr', type=float, default=0.01)
    args.add_argument('--layer', type=int, default=5)
    args.add_argument('--size', type=int, default=16)
    args.add_argument('--load', type=int, default=0)
    args.add_argument('--gate', type=int, default=1)
    args.add_argument('--conditional', type=int, default=0)
    args.add_argument('--pause', type=int, default=0)
    config = args.parse_args()

    num_classes = 10  # FIXME: hardcode
    is_gpu = torch.cuda.is_available()
    base_tr = transforms.Compose([transforms.ToTensor()])
    train_dataset = DataLoader(
        datasets.MNIST('./data', train=True, download=True,  transform=base_tr),
        batch_size=config.batch, shuffle=True, num_workers=4
    )
    test_dataset = DataLoader(
        datasets.MNIST('./data', train=False, transform=base_tr),
        batch_size=config.batch, shuffle=False, num_workers=4
    )

    if config.gate:
        model = GatedPixelCNN(
            config.size, config.layer, conditional=config.conditional, num_classes=num_classes
        )
    else:
        model = PixelCNN(config.size, config.layer)
    if is_gpu:
        model.cuda()
    optimizer = optim.Adam([*model.parameters()], lr=config.lr)
    os.makedirs(config.model, exist_ok=True)

    if config.mode == 'train':
        for epoch in range(config.epochs):
            model.train()
            err_tr, err_te = 0, 0
            batch_iter_tr, batch_iter_te = 0, 0
            batch_size = len(train_dataset)
            for batch_idx, train in enumerate(train_dataset):
                train_label = one_hot_encoder(train[1], num_classes)
                train_data = train[0]
                if is_gpu:
                    train_label = train_label.cuda()
                    train_data = train_data.cuda()
                train_label = Variable(train_label)
                train = Variable(train_data)
                target = Variable((train.data[:, 0] * 255).long())
                train_pred = model(train, train_label)
                
                loss = F.cross_entropy(train_pred, target)
                err_tr += loss.item()
                batch_iter_tr += 1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            model.eval()
            for test in test_dataset:
                test_label = one_hot_encoder(test[1], num_classes)
                test_data = test[0]
                if is_gpu:
                    test_label = test_label.cuda()
                    test_data = test_data.cuda()
                test_label = Variable(test_label)
                test = Variable(test_data)
                target = Variable((test.data[:, 0] * 255).long())
                loss = F.cross_entropy(model(test, test_label), target)
                err_te += loss.item()
                batch_iter_te += 1
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(state, os.path.join(config.model, '{}.torch'.format(epoch)))
            print('epoch: {0}, train_err: {1}, test_err: {2}'.format(
                epoch, err_tr / batch_iter_tr, err_te / batch_iter_te
            ))
            # sampling
            nrows = 12
            sample = torch.Tensor(nrows*nrows, 1, 28, 28)
            random_num = np.random.randint(0, num_classes)
            print('random_number: {}'.format(random_num))
            random_condition = one_hot_encoder([random_num for i in range(nrows*nrows)], num_classes)
            if is_gpu:
                sample = sample.cuda()
                random_condition = random_condition.cuda()
            sample.fill_(0)
            for w in range(28):
                for h in range(28):
                    out = model(Variable(sample, volatile=True), Variable(random_condition))
                    probs = F.softmax(out[:, :, w, h]).data
                    sample[:, :, w, h] = torch.multinomial(probs, 1).float() / 255.  # normalization
            utils.save_image(sample, 'sample_{:02d}.png'.format(epoch), nrow=nrows, padding=0)
    else:
        file_path = os.path.join(config.model, '{}.torch'.format(config.load))
        if not os.path.isfile(file_path):
            raise FileNotFoundError()
        load_state = torch.load(file_path)
        model.load_state_dict(load_state['model'])
        model.eval()
        nrows = 12
        sample = torch.Tensor(nrows * nrows, 1, 28, 28)
        row_condition = [np.random.randint(0, num_classes) for i in range(nrows*nrows)]
        print(row_condition)
        random_condition = one_hot_encoder(row_condition, num_classes)
        if is_gpu:
            sample = sample.cuda()
            random_condition = random_condition.cuda()
        sample.fill_(0)
        for w in range(28):
            for h in range(28):
                out = model(Variable(sample, volatile=True), Variable(random_condition))
                probs = F.softmax(out[:, :, w, h]).data
                sample[:, :, w, h] = torch.multinomial(probs, 1).float() / 255.  # normalization        
        utils.save_image(sample, 'sample_{:02d}.png'.format(config.load), nrow=nrows, padding=0)
