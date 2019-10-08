import argparse
import os
import random
import shutil
import time
import warnings
# import umap
# from scipy.sparse.csgraph import connected_components
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# from sklearn.decomposition import PCA
import numpy as np
# from sklearn import metrics
# from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-vb', '--val-batch-size', default=1000, type=int,
                    metavar='N')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N',
                    help='mini-batch size (default: 1), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or'
                         ' multi node data parallel training')
parser.add_argument('--kfold', default=5, type=int)

best_acc1 = 0


class KFoldSampler(torch.utils.data.Sampler):

    def __init__(self, data_source, k=5, seed=None):
        self.data_source = data_source
        self.good_index = [i for i, (x, y) in enumerate(data_source.imgs) if data_source.classes[int(y)] == "good"]
        self.defective_index = [i for i, (x, y) in enumerate(data_source.imgs) if data_source.classes[int(y)] == "defective"]
        np.random.seed(seed)
        np.random.shuffle(self.good_index)
        np.random.shuffle(self.defective_index)
        self.good_index_split = np.array_split(self.good_index, k)
        self.state = "good_train"
        self.set_k(0)

    def set_k(self, kidx):
        self.train = self.good_index_split[:]
        del self.train[kidx]
        self.train = [i for inter in self.train for i in inter]
        self.val = self.good_index_split[kidx]

    def set_state(self, state):
        self.state = state

    def __iter__(self):
        if self.state == "test":
            return iter(range(len(self.data_source)))
        elif self.state == "good_train":
            return iter(self.train)
        elif self.state == "good_val":
            return iter(self.val)
        elif self.state == "good":
            return iter(self.good_index)
        elif self.state == "defective":
            return iter(self.defective_index)
        else:
            assert True

    def __len__(self):
        if self.state == "test":
            return len(self.data_source)
        elif self.state == "good_train":
            return len(self.train)
        elif self.state == "good_val":
            return len(self.val)
        elif self.state == "good":
            return len(self.good_index)
        elif self.state == "defective":
            return len(self.defective_index)
        else:
            assert True


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend,
                                init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    args.arch = 'mobilenet_v2'
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()
    # mobilenetv2
    model = torch.hub.load('pytorch/vision', 'mobilenet_v2', pretrained=True)
    model.classifier = nn.Sequential(*list(model.classifier.children())[:-2])
    # model.features = nn.Sequential(*list(model.features.children())[:6])
    # print(model)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(
                    (args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to
            # all available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate
        # batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'test')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(traindir, transforms.Compose([
            transforms.Resize(224),
            # transforms.Grayscale(),
            transforms.ToTensor(),
            # transforms.Lambda(lambda gray: torch.cat([gray, gray, gray])),
            normalize
        ]))
    print("Train", train_dataset.classes)

    train_sampler = KFoldSampler(train_dataset, seed=10, k=5)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(224),
            # transforms.Grayscale(),
            transforms.ToTensor(),
            # transforms.Lambda(lambda gray: torch.cat([gray, gray, gray])),
            normalize,
        ]))
    print("Test", val_dataset.classes)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.val_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # しきい値の決定
    thresholds_list = [0.85, 0.9, 0.95, 0.99, 0.999]
    threshold_results = {}
    for test_threshold in thresholds_list:
        d_average = 0
        for k_count in range(args.kfold):
            # Train
            train_sampler.set_k(k_count)
            print("train index", train_sampler.train)
            print("val_index", train_sampler.val)
            d_average += train(train_loader, val_loader, model, criterion, args, k_count, test_threshold)
        threshold_results[test_threshold] = d_average/args.kfold
        print(str(test_threshold), d_average/args.kfold)
    print(threshold_results)
    threshold = min(threshold_results, key=threshold_results.get)
    print(threshold, threshold_results[threshold])

    # 決めたしきい値を用いてもう一度正常部分空間をすべての正常データで作る
    # それに対して以上データを追加していく


def train(train_loader, val_loader, model, criterion, args, k_count, test_threshold):

    # switch to evaluate mode
    model.eval()

    # Train
    # 正常データのみを用いたトレーニング
    with torch.no_grad():
        sub_vec = None
        sub_val = None
        train_loader.sampler.set_state("good_train")
        for i, (images, target) in enumerate(train_loader):  # TODO batch dependancies
            end = time.time()
            print("\n#" + "-"*30 + ' ' + str(i) + 'epoch ' + "-"*30 + "#")
            # Reverse order
            images = torch.from_numpy(images.numpy()[::-1].copy())
            target = torch.from_numpy(target.numpy()[::-1].copy())

            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            output = output.cpu().numpy()
            target = target.cpu().numpy()

            # normlization
            # scaler = MinMaxScaler()
            # print(scaler.fit(output))
            # output = scaler.transform(output)
            # output = output.numpy()
            # output = output/np.linalg.norm(output, axis=1).reshape(-1, 1)

            print("### Calc eigenvalue, eigenvector")
            print("input shape (model output)", output.shape)
            # SVDを用いた正常部分空間作成
            if (i == 0):
                e_vec, e_val = calc_SVD(output)

            # インクリメンタルPCAを用いた更新
            else:
                # incremental
                print("Incremental")
                e_vec, e_val = incremental_PCA(output, sub_vec, sub_val, i)

            print("e_val.shape", e_val.shape)
            print("e_val", e_val)
            print("e_vec.shape", e_vec.shape)
            print("e_vec", e_vec)
            plt.figure()
            plt.plot(e_val)
            plt.savefig('e_val.png')

            # subspace
            # 次元を落として部分空間を作成
            print("### Calc Subspace")
            sub_vec, sub_val = calc_sub_vec(e_vec, e_val, test_threshold)

            print(time.time() - end)

        # # umap
        # # embedding = PCA(random_state=0).fit_transform(output.cpu())
        # embedding = umap.UMAP(n_neighbors=5, random_state=0).fit_transform(output)
        # plt.figure()
        # plt.scatter(embedding[:, 0], embedding[:, 1], c=target, cmap=cm.nipy_spectral)
        # plt.colorbar()
        # plt.savefig('umap.png')

        # ----------------validation------------------------
        # 生成した正常部分空間を正常データのみで評価(寄与率決定用)
        train_loader.sampler.set_state("good")
        outputs_list = []
        targets_list = []
        for i, (images, target) in enumerate(train_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            output = output.cpu()
            target = target.cpu()
            # output = output/np.linalg.norm(output, axis=1).reshape(-1, 1)
            # output = scaler.transform(output)
            outputs_list.append(output)
            targets_list.append(target)

        output = torch.cat(outputs_list).numpy()
        target = torch.cat(targets_list).numpy()

        print("### Calc Error")
        good_d, good_stddev = calc_errorval(output, sub_vec)

        # -------------------test---------------------------------
        # 生成した正常部分空間を以上データを含めて評価(可視化用)
        train_loader.sampler.set_state("test")
        outputs_list = []
        targets_list = []
        for i, (images, target) in enumerate(train_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            output = output.cpu()
            target = target.cpu()
            # output = output/np.linalg.norm(output, axis=1).reshape(-1, 1)
            # output = scaler.transform(output)
            outputs_list.append(output)
            targets_list.append(target)

        output = torch.cat(outputs_list).numpy()
        target = torch.cat(targets_list).numpy()

        print("### Calc Error")
        d, _ = calc_errorval(output, sub_vec)

        plt.figure()
        plt.ylim(0, 1)
        plt.scatter(range(len(d)), d, c=target, cmap=cm.nipy_spectral)
        plt.colorbar()
        plt.savefig('d_{}_{}.png'.format(str(test_threshold), k_count))

        # roc_label = np.where((target == 0) | (target == 1))
        # d = d[roc_label]
        # target = target[roc_label]
        # fpr, tpr, thresholds = metrics.roc_curve(target, (d*(-1)))
        # plt.figure()
        # plt.plot(fpr, tpr)
        # plt.savefig('roc.png')
        # print(metrics.roc_auc_score(target, (d*(-1))))

    return np.mean(good_d) + good_stddev*10


def calc_SVD(features):
    # svd
    print("SVD", len(features))
    U, s, V = np.linalg.svd(features)
    e_val = s**2 / features.shape[0]
    e_vec = V.T
    return e_vec, e_val


def calc_sub_vec(e_vec, e_val, threshold):
    sum_all = np.sum(e_val)
    sum_val = np.array([np.sum(e_val[:i])/sum_all for i in range(1, len(e_val)+1)])
    r = int(min(np.where(sum_val >= threshold)[0])+1)
    sub_vec = e_vec.T[:r].T
    sub_val = e_val[:r]

    plt.figure()
    plt.plot(sum_val)
    plt.savefig('sum_e_val.png')
    print("sub_vec.shape", sub_vec.shape)
    print("sub_vec", sub_vec)
    return sub_vec, sub_val


def incremental_PCA(features, sub_vec, sub_val, n):
    h = np.reshape(features[0] - sub_vec @ sub_vec.T @ features[0], (-1, 1))
    h_norm = np.linalg.norm(h)
    h_hat = h / h_norm if h_norm > 0.1 else np.zeros(h.shape)
    g = np.reshape(sub_vec.T @ features[0], (-1, 1))
    gamma = h_hat.T @ features[0]
    # calc L
    lamd = np.diag(sub_val)
    l1 = np.block([[lamd, np.zeros((lamd.shape[0], 1))],
                  [np.zeros((1, lamd.shape[1])), 0]])
    l2 = np.block([[g @ g.T, gamma * g],
                   [gamma * g.T, gamma**2]])
    ll = (n/(n+1) * l1) + (1/(n+1) * l2)
    # calc rotation matrix
    e_val, rot = np.linalg.eigh(ll)
    e_val = e_val[::-1]
    # update e_vec
    e_vec = np.block([sub_vec, h_hat]) @ rot
    e_vec = e_vec.T[::-1].T
    return e_vec, e_val


def calc_errorval(features, sub_vec):
    # Y_rec
    y = features @ sub_vec @ sub_vec.T

    # ユークリッド距離
    # dist = np.linalg.norm(output-y, axis=1)
    # cos
    dist = [np.inner(oi, yi) / (np.linalg.norm(oi) * np.linalg.norm(yi)) for oi, yi in zip(features, y)]
    # sin
    dist = np.sqrt(1-np.power(dist, 2))
    print(dist)

    # 分散
    # good_variarance = dist.var()
    # 標準偏差
    stddev = dist.std()
    return dist, stddev


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
