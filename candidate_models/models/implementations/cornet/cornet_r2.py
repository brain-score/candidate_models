import argparse
import base64
import io
import json
import math
import os
import pickle
import pprint
import shlex
import subprocess
import time

import fire
import numpy as np
import pandas
import torch
import torch.backends.cudnn
import torch.distributed
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets
import torchvision.models
import torchvision.transforms
import tqdm
from PIL import Image
from sklearn.decomposition import PCA
from torch.nn import Module

from candidate_models.models.implementations import Defaults
# from streams.envs import objectome
# from streams.metrics.behav_cons import objectome_cons
from candidate_models.models.implementations.cornet import TemporalPytorchModel

np.random.seed(0)
torch.manual_seed(0)

host = os.uname()[1]
if host.startswith('braintree'):
    DATA_PATH = '/braintree/data2/active/common/imagenet_raw/'
    # DATA_PATH = '/braintree/data2/active/users/qbilius/datasets/imagenet_lmdb/'
    # DATA_PATH ‘= '/scratch/Mon/imagenet_raw/'
elif host.startswith('node'):
    DATA_PATH = '/om/user/qbilius/imagenet_raw/'

OUTPUT = os.environ.get('MEMO_DIR')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=['resnet18', 'resnet50'],
                    help=('model architecture (default: resnet18)'))
parser.add_argument('-j', '--workers', default=40, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
# parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
#                     help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# parser.add_argument('--print-freq', '-p', default=100, type=int,
# metavar='N', help='print frequency (default: 10)')
# parser.add_argument('--resume', default='', type=str, metavar='PATH',
# help='path to latest checkpoint (default: none)')
# parser.add_argument('--pretrained', dest='pretrained', action='store_true',
#                     help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--ngpus', default=1, type=int)

parser.add_argument('--block_stop', default=2, type=int)
parser.add_argument('--times', default=2, type=int)
parser.add_argument('--region', default='IT', choices=['V4', 'IT'])

FLAGS, _ = parser.parse_known_args()
FLAGS.distributed = FLAGS.world_size > 1
if FLAGS.distributed:
    torch.distributed.init_process_group(backend=FLAGS.dist_backend,
                                         init_method=FLAGS.dist_url,
                                         world_size=FLAGS.world_size)
normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
torch.backends.cudnn.benchmark = True


def set_gpus(n=1):
    gpus = subprocess.run(shlex.split('nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,nounits'),
                          check=True, stdout=subprocess.PIPE).stdout
    gpus = pandas.read_csv(io.BytesIO(gpus), sep=', ', engine='python')
    gpus = gpus[gpus['memory.total [MiB]'] > 10000]  # only above 10 GB
    if os.environ.get('CUDA_VISIBLE_DEVICES') is not None:
        visible = [int(i) for i in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
        gpus = gpus[gpus['index'].isin(visible)]
    gpus['ratio'] = gpus['memory.free [MiB]'] / gpus['memory.total [MiB]']
    gpus = gpus.sort_values(by='ratio', ascending=False)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in gpus['index'].iloc[:n]])
    # return gpus['index'].iloc[:n].tolist()


# set_gpus(FLAGS.ngpus)
# FLAGS.gpus = set_gpus(FLAGS.ngpus)


class CORBlock(nn.Module):

    def __init__(self, in_channels, out_channels, ntimes=1, stride=1, name=None):
        super(CORBlock, self).__init__()

        self.name = name
        self.ntimes = ntimes
        self.stride = stride

        self.relu = nn.ReLU(inplace=True)
        self.conv_first = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.shortcut = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.shortcut_bn = nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels * 4, out_channels * 4, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.conv3 = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1, bias=False)
        self.last_relu = nn.ReLU(inplace=True)

        for n in range(ntimes):
            setattr(self, f'bn1_{n}', nn.BatchNorm2d(out_channels * 4))
            setattr(self, f'bn2_{n}', nn.BatchNorm2d(out_channels * 4))
            setattr(self, f'bn3_{n}', nn.BatchNorm2d(out_channels))

        # self.feats = []

    def forward(self, x):
        # feats = []
        x = self.conv_first(x)
        # times = self.ntimes if self.name != f'b{FLAGS.block_stop}' else FLAGS.times
        # for n in range(self.ntimes):
        for n in range(self.ntimes):
            if n == 0:
                residual = self.shortcut_bn(self.shortcut(x))
            else:
                residual = x

            x = self.conv1(x)
            x = getattr(self, f'bn1_{n}')(x)
            x = self.relu(x)

            if n == 0 and self.stride == 2:
                self.conv2.stride = (2, 2)
            else:
                self.conv2.stride = (1, 1)
            x = self.conv2(x)
            x = getattr(self, f'bn2_{n}')(x)
            x = self.relu(x)

            x = self.conv3(x)
            x = getattr(self, f'bn3_{n}')(x)

            x += residual
            x = self.last_relu(x)
            # self.feats.append(x)
        # self.feats.append(feats)

        return x


class CORNet(nn.Module):

    def __init__(self, ntimes, num_classes=1000):
        super(CORNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.blocks = nn.Sequential(CORBlock(64, 128, ntimes=ntimes[0], stride=2, name='b0'),
                                    CORBlock(128, 256, ntimes=ntimes[1], stride=2, name='b1'),
                                    CORBlock(256, 512, ntimes=ntimes[2], stride=2, name='b2'))

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.blocks(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class CORNet2Wrapper(TemporalPytorchModel):
    def __init__(self, weights=Defaults.weights, batch_size=Defaults.batch_size,
                 image_size=Defaults.image_size):
        super().__init__('cornet', weights, batch_size, image_size)

    def _create_model(self, model_name, weights):
        model = CORNet([2, 4, 2])
        assert weights in [None, 'imagenet']
        if weights == 'imagenet':
            class Wrapper(Module):
                def __init__(self, model):
                    super(Wrapper, self).__init__()
                    self.module = model

            model = Wrapper(model)  # model was wrapped with DataParallel, so weights require `module.` prefix
            checkpoint = torch.load(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..',
                                                 'model-weights', 'cornet2', 'epoch_43.pth.tar'),
                                    map_location=lambda storage, loc: storage)  # map onto cpu
            model.load_state_dict(checkpoint['state_dict'])
            model = model.module  # unwrap
        return model


def train(restore_path=None,
          save_train_epochs=.1,
          save_val_epochs=.5,
          save_model_epochs=4,
          save_model_secs=None
          ):
    model = CORNet([2, 4, 2])

    if not FLAGS.distributed:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)

    train = ImageNetTrain(model)
    vals = [
        ImageNetVal(model),
        I2(model)
    ]

    start_epoch = 0
    if restore_path is not None:
        if os.path.isfile(restore_path):
            print("=> loading checkpoint '{}'".format(restore_path))
            ckpt_data = torch.load(restore_path)
            start_epoch = ckpt_data['epoch']
            # start_step = ckpt_data['step_in_epoch']
            model.load_state_dict(ckpt_data['state_dict'])
            train.optimizer.load_state_dict(ckpt_data['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(restore_path, ckpt_data['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(restore_path))

    recs = []
    recent_time = time.time()

    nsteps = len(train.data_loader)
    if save_train_epochs is not None:
        save_train_steps = (np.arange(0, 1, save_train_epochs) * nsteps).astype(int)
    if save_val_epochs is not None:
        save_val_steps = (np.arange(0, 1, save_val_epochs) * nsteps).astype(int)
    if save_model_epochs is not None:
        save_model_steps = (np.arange(0, 1, save_model_epochs) * nsteps).astype(int)

    results = {'meta': {'step_in_epoch': 0,
                        'epoch': start_epoch,
                        'wall_time': time.time()}
               }
    for epoch in progress(0, FLAGS.epochs + 1, initial=start_epoch, desc='epoch'):
        if FLAGS.distributed:
            train.sampler.set_epoch(epoch)

        st = time.time()
        for step, data in enumerate(progress(train.data_loader, desc=train.name)):
            tt = time.time() - st
            # if tt > .1:
            #     print(tt)
            if save_val_steps is not None:
                if step in save_val_steps:
                    for val in vals:
                        results[val.name] = val()
                    train.model.train()

            if OUTPUT is not None:
                recs.append(results)
                if len(results) > 1:
                    pickle.dump(recs, open(OUTPUT + 'results.pkl', 'wb'))

                ckpt_data = {}
                ckpt_data['flags'] = FLAGS.__dict__.copy()
                ckpt_data['epoch'] = epoch
                ckpt_data['state_dict'] = model.state_dict()
                ckpt_data['optimizer'] = train.optimizer.state_dict()

                if save_model_secs is not None:
                    if time.time() - recent_time > save_model_secs:
                        torch.save(ckpt_data, OUTPUT + 'latest_checkpoint.pth.tar')
                        recent_time = time.time()

                if save_model_steps is not None:
                    if step in save_model_steps:
                        torch.save(ckpt_data, OUTPUT + f'epoch_{epoch:02d}.pth.tar')

            else:
                if len(results) > 1:
                    pprint.pprint(results)

            if epoch < FLAGS.epochs:
                frac_epoch = epoch + (step + 1) / len(train.data_loader)
                rec = train(frac_epoch, *data)
                results = {'meta': {'step_in_epoch': step + 1,
                                    'epoch': frac_epoch,
                                    'wall_time': time.time()}
                           }
                if save_train_steps is not None:
                    if step in save_train_steps:
                        results[train.name] = rec
            st = time.time()


def test(restore_path=None):
    model = CORNet([2, 4, 2])

    if not FLAGS.distributed:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)

    val = ImageNetVal(model)

    if restore_path is not None:
        if os.path.isfile(restore_path):
            print("=> loading checkpoint '{}'".format(restore_path))
            ckpt_data = torch.load(restore_path)
            model.load_state_dict(ckpt_data['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(restore_path, ckpt_data['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(restore_path))

    print(val())


class ImageNetTrain(object):

    def __init__(self, model):
        self.name = 'train'
        self.model = model
        self.data_loader = self.data()
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         FLAGS.lr,
                                         momentum=FLAGS.momentum,
                                         weight_decay=FLAGS.weight_decay)
        self.lr = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20)
        self.loss = nn.CrossEntropyLoss().cuda()

    def data(self):
        dataset = torchvision.datasets.ImageFolder(
            os.path.join(DATA_PATH, 'train'),
            torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                normalize,
            ]))
        if FLAGS.distributed:
            self.sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            self.sampler = None

        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=FLAGS.batch_size,
                                                  shuffle=self.sampler is None,
                                                  num_workers=FLAGS.workers,
                                                  pin_memory=True,
                                                  sampler=self.sampler)
        return data_loader

    def __call__(self, frac_epoch, inp, target):
        start = time.time()

        self.lr.step(epoch=frac_epoch)
        target = target.cuda(non_blocking=True)
        output = self.model(inp)

        rec = {}
        loss = self.loss(output, target)
        rec['loss'] = loss.item()
        rec['top1'], rec['top5'] = accuracy(output, target, topk=(1, 5))
        rec['top1'] /= len(output)
        rec['top5'] /= len(output)
        rec['learning_rate'] = self.lr.get_lr()[0]

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        del loss  # reduce GPU memory

        rec['dur'] = time.time() - start
        return rec


class ImageNetVal(object):
    def __init__(self, model):
        self.name = 'val'
        self.model = model
        self.data_loader = self.data()
        self.loss = nn.CrossEntropyLoss(size_average=False).cuda()

    def data(self):
        dataset = torchvision.datasets.ImageFolder(
            os.path.join(DATA_PATH, 'val_in_folders'),
            torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                normalize,
            ]))
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=FLAGS.batch_size,
                                                  shuffle=False,
                                                  num_workers=FLAGS.workers,
                                                  pin_memory=True)

        return data_loader

    def __call__(self):
        self.model.eval()
        start = time.time()
        rec = {'loss': 0, 'top1': 0, 'top5': 0}
        with torch.no_grad():
            for step, (inp, target) in enumerate(progress(self.data_loader, desc=self.name)):
                target = target.cuda(non_blocking=True)
                output = self.model(inp)

                rec['loss'] += self.loss(output, target).item()
                p1, p5 = accuracy(output, target, topk=(1, 5))
                rec['top1'] += p1
                rec['top5'] += p5

        for key in rec:
            rec[key] /= len(self.data_loader.dataset.samples)
        rec['dur'] = (time.time() - start) / len(self.data_loader)

        return rec


class I2(object):

    def __init__(self, model):
        obj = objectome.Objectome()
        self.name = 'I2'
        self.imids = obj.meta.id
        self.model = model
        self.data_loader = self.data()
        self.layer = self.model._modules.get('module').avgpool

    def _store_feats(self, layer, inp, output):
        id_ = output.device.index
        output = np.reshape(output, (len(output), -1)).numpy()
        self._model_feats[id_] = output

    def data(self):
        dataset = ImageFolder(
            '/braintree/home/qbilius/.streams/objectome/imageset/',
            torchvision.transforms.Compose([
                torchvision.transforms.Resize(224),
                torchvision.transforms.ToTensor(),
                normalize,
            ]))
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=FLAGS.batch_size,
                                                  shuffle=False,
                                                  num_workers=FLAGS.workers,
                                                  pin_memory=True)
        return data_loader

    def __call__(self):
        start = time.time()
        imids = []
        self.model.eval()
        hook = self.layer.register_forward_hook(self._store_feats)
        model_feats = []
        with torch.no_grad():
            for inp, target, imid in progress(self.data_loader, desc=self.name):
                self._model_feats = {}
                self.model(inp)
                for device_id in range(FLAGS.ngpus):
                    model_feats.append(self._model_feats[device_id])
                imids.extend(np.array(imid))
        hook.remove()

        rec = {}
        rec['dur'] = (time.time() - start) / len(self.data_loader)

        order = [imids.index(imid) for imid in self.imids]
        model_feats = np.row_stack(model_feats)[order]
        pca = PCA(n_components=min(model_feats.shape[1], 200))
        try:
            model_feats_pca = pca.fit_transform(model_feats)
        except:
            rec['I2n'] = np.nan
        else:
            i2 = objectome_cons(model_feats_pca, metric='i2n', kind='dprime', cap=5)
            rec['I2n'] = i2.cons.mean()

        return rec


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        _, pred = output.topk(max(topk), dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = [correct[:k].sum().item() for k in topk]
        return res


class Images(torchvision.datasets.ImageFolder):

    def __init__(self, impaths, targets, transform=None, target_transform=None,
                 loader=torchvision.datasets.folder.default_loader):

        # classes, class_to_idx = self._find_classes(root)
        # samples = make_dataset(root, class_to_idx, extensions)
        # if len(samples) == 0:
        #     raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
        #                        "Supported extensions are: " + ",".join(extensions)))

        # self.root = root
        self.loader = loader
        self.extensions = torchvision.datasets.folder.IMG_EXTENSIONS

        self.classes = np.unique(targets)
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.samples = [(s, t) for s, t in zip(impaths, targets)]
        self.targets = targets

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        imid = os.path.splitext(os.path.basename(path))[0]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, imid


class ImageFolder(torchvision.datasets.ImageFolder):

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        imid = os.path.splitext(os.path.basename(path))[0]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, imid


class ProgressFile(object):

    def __init__(self, pbar, fname):
        self.fname = fname
        self.pbar = pbar

    def write(self, message):
        if not repr(message).startswith("'\\x") and OUTPUT is not None:
            message = message.strip('\r\n')
            messages = []
            for instance in self.pbar._instances:
                try:
                    bar_str = repr(instance)
                except:
                    pass
                else:
                    if instance.__hash__() == self.__hash__():
                        messages.append(message)
                    else:
                        messages.append(bar_str)

            messages = '\n'.join(messages)
            if len(message) > 0:
                with open(self.fname, 'w') as f:
                    f.write(tqdm._utils._unicode(messages))

    def flush(self):
        pass


class progress(tqdm.tqdm):

    def __init__(self, start=0, stop=None, step=1, initial=0, total=None,
                 file='progress.out', desc=None, **tqdm_kwargs):
        try:
            start + 1
        except:  # not a number
            try:
                rng = start[initial]
            except:
                rng = start  # initial silently ignored
        else:  # range
            if stop is None:
                rng = np.arange(initial, start, step)
                total = start
            else:
                rng = np.arange(initial, stop, step)
                total = stop - start

        f = ProgressFile(self, OUTPUT + file) if OUTPUT is not None else None
        super(progress, self).__init__(rng, initial=initial, total=total,
                                       file=f, desc=desc, **tqdm_kwargs)


class ImageNet(torch.utils.data.Dataset):
    """
    ImageNet dataset.
    Args:
        root (string): Root directory for the database files.
        kind (string): One of {'train', 'val'}.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        self.env = lmdb.open(self.root, max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']

        # cache_file = '_cache_' + ''.join(c for c in root if c in string.ascii_letters)
        # if os.path.isfile(cache_file):
        #     self.keys = pickle.load(open(cache_file, "rb"))
        # else:
        #     with self.env.begin(write=False) as txn:
        #         self.keys = [key for key, _ in tqdm.tqdm(txn.cursor())]
        #     import ipdb; ipdb.set_trace()
        #     pickle.dump(self.keys, open(cache_file, "wb"))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target) where target is the index of the target category.
        """
        env = self.env
        with env.begin(write=False) as txn:
            data = txn.get(str(index).encode())  # self.keys[index])

        data = json.loads(data.decode())

        imgbuf = io.BytesIO(base64.b64decode(data['image']))
        img = Image.open(imgbuf).convert('RGB')
        target = data['label']

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.root + ')'


if __name__ == '__main__':
    fire.Fire()
