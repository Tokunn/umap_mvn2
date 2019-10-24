import argparse
import os
import csv
import shutil
import random
import time
import warnings
import umap
# from scipy.sparse.csgraph import connected_components
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# from sklearn.decomposition import PCA
import numpy as np
from sklearn import metrics
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
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

RESIZE = 50


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
parser.add_argument('--prmc', default=0.5, type=float)
parser.add_argument('--pngdir', default='.', type=str)
parser.add_argument('--delheadvec', default=0, type=int)
parser.add_argument('--uselayer', default=0, type=int)

best_acc1 = 0


def DEBUG(*args):
    # input("[!!] DEBUG, > ")
    if len(args) == 1:
        return args[0]
    return args


def DEBUG_SHOW(images):
    os.makedirs('imgs', exists_ok=True)
    imgs = np.transpose(images, (0, 2, 3, 1))
    for i, img in enumerate(imgs):
        plt.figure()
        plt.imshow(img)
        plt.savefig("imgs/{}.png".format(str(time.time())))
        # plt.show()
        plt.close()


def saveimg(images, args, prefix=""):
    for i, img in enumerate(images):
        plt.figure()
        plt.imshow(np.transpose(img, (1, 2, 0)))
        plt.savefig(os.path.join(args.pngdir, prefix+str(i)+'.png'))
        plt.close()


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
        if self.state == "all":
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
        if self.state == "all":
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


class ImageFolderPath(torchvision.datasets.folder.DatasetFolder):

    def __init__(self, root, transform=None, target_transform=None,
                 loader=torchvision.datasets.folder.default_loader, is_valid_file=None):
        super(ImageFolderPath, self).__init__(
                root, loader,
                torchvision.datasets.folder.IMG_EXTENSIONS if is_valid_file is None else None,
                transform=transform,
                target_transform=target_transform,
                is_valid_file=is_valid_file)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path


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

    if os.path.exists(args.pngdir):
        shutil.rmtree(args.pngdir)
    os.makedirs(args.pngdir, exist_ok=True)

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
    if args.uselayer:
        model.features = nn.Sequential(*list(model.features.children())[:args.uselayer])
        print(model)
        print(len(model.features))

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

    if args.uselayer:
        train_dataset = datasets.ImageFolder(traindir, transforms.Compose([
                transforms.Resize(224),
                # transforms.Grayscale(),
                transforms.ToTensor(),
                # transforms.Lambda(lambda gray: torch.cat([gray, gray, gray])),
                normalize
            ]))
    else:
        train_dataset = datasets.ImageFolder(traindir, transforms.Compose([
                transforms.Resize(RESIZE),  # 画像を表示したい
                # transforms.Grayscale(),
                transforms.ToTensor(),
                # transforms.Lambda(lambda gray: torch.cat([gray, gray, gray])),
            ]))
    print("Train", train_dataset.classes)

    train_sampler = KFoldSampler(train_dataset, seed=1, k=5)

    train_loader1000 = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=1000, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    train_loader1 = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.uselayer:
        val_dataset = ImageFolderPath(valdir, transforms.Compose([
                transforms.Resize(224),
                # transforms.Grayscale(),
                transforms.ToTensor(),
                # transforms.Lambda(lambda gray: torch.cat([gray, gray, gray])),
                normalize,
            ]))
    else:
        val_dataset = ImageFolderPath(valdir, transforms.Compose([
                transforms.Resize(RESIZE),  # 画像を表示したい
                # transforms.Grayscale(),
                transforms.ToTensor(),
                # transforms.Lambda(lambda gray: torch.cat([gray, gray, gray])),
            ]))
    print("Test", val_dataset.classes)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.val_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # しきい値の決定
    thresholds_list = [0.85, 0.9, 0.95, 0.99, 0.999]
    # thresholds_list = DEBUG([0.999])
    threshold_results = {}
    for test_threshold in thresholds_list:
        d_average = 0
        for k_count in range(args.kfold):
            # Train
            train_sampler.set_k(k_count)
            print("train index", train_sampler.train)
            print("val_index", train_sampler.val)
            d_result, (_, _) = train_good(train_loader1000, val_loader,
                                          model, criterion, args,
                                          k_count, test_threshold, "good_train")
            d_average += d_result
        threshold_results[test_threshold] = d_average/args.kfold
        print(str(test_threshold), d_average/args.kfold)
    print(threshold_results)
    threshold = min(threshold_results, key=threshold_results.get)
    print(threshold, threshold_results[threshold])

    # 決めたしきい値を用いて、すべてのデータで正常部分空間を作る
    _, (sub_vec, sub_val) = train_good(train_loader1000, val_loader,
                                       model, criterion, args,
                                       k_count, threshold, "good")

    # それに対して異常データを追加していく
    train_defective(train_loader1, val_loader, model, args, threshold, sub_vec, sub_val)


def train_good(train_loader, val_loader, model,
               criterion, args, k_count, test_threshold, sampler_state):
    start_time = time.time()

    # switch to evaluate mode
    model.eval()

    # Train
    # 正常データのみを用いたトレーニング
    with torch.no_grad():
        sub_vec = None
        sub_val = None
        train_loader.sampler.set_state(sampler_state)
        for i, (images, target) in enumerate(train_loader):  # TODO batch dependancies
            learned_good_image = images.numpy()
            assert i == 0
            # DEBUG_SHOW(images)
            end = time.time()
            print("\n#" + "="*30 + ' train_good SVD ' + str(test_threshold) + '/' + str(k_count) + ' ' + "="*30 + "#")
            # Reverse order
            images = torch.from_numpy(images.numpy()[::-1].copy())
            target = torch.from_numpy(target.numpy()[::-1].copy())

            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            if args.uselayer:
                output = model(images)
            else:
                # 画像を表示したい
                output = images.reshape(images.shape[0], -1)
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
                assert False
                # incremental
                print("Incremental")
                e_vec, e_val = incremental_PCA(output, sub_vec, sub_val, i)

            print("e_val.shape", e_val.shape)
            # print("e_val", e_val)
            print("e_vec.shape", e_vec.shape)
            # print("e_vec", e_vec)
            # plt.figure()
            # plt.plot(e_val)
            # plt.savefig('E_VAL.png')
            # plt.close()

            # subspace
            # 次元を落として部分空間を作成
            print("### Calc Subspace")
            if args.delheadvec:
                # 最初に頭から指定個の基底を落とす
                e_vec = e_vec.T[args.delheadvec:].T
                e_val = e_val[args.delheadvec:]
            # 寄与率に応じて後ろを落とす
            sub_vec, sub_val = calc_sub_vec(e_vec, e_val, test_threshold, args)

            print(time.time() - end)

            if sampler_state == 'good':
                testall(val_loader, model, args, test_threshold, sub_vec, sub_val, learned_good_image, prefix="good")
            elif k_count == 0:
                testall(val_loader, model, args, test_threshold, sub_vec, sub_val, learned_good_image, prefix="good_crossval")

            if not args.uselayer:
                # 画像を表示したい
                for i, vec_img in enumerate(sub_vec.T):
                    vec_img = np.reshape(vec_img, (3, RESIZE, RESIZE))
                    vec_img = np.transpose(vec_img, (1, 2, 0))
                    vec_img = (vec_img-vec_img.min()) / (vec_img.max()-vec_img.min())
                    plt.imshow(vec_img)
                    plt.savefig(os.path.join(args.pngdir, "vec_img_{}".format(i)))
                    plt.close()

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
            if args.uselayer:
                output = model(images)
            else:
                # 画像を表示したい
                output = images.reshape(images.shape[0], -1)
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

        # # -------------------test---------------------------------
        # # 生成した正常部分空間を以上データを含めて評価(可視化用)
        # train_loader.sampler.set_state("test")
        # outputs_list = []
        # targets_list = []
        # for i, (images, target) in enumerate(train_loader):
        #     if args.gpu is not None:
        #         images = images.cuda(args.gpu, non_blocking=True)
        #     target = target.cuda(args.gpu, non_blocking=True)

        #     # compute output
        #     output = model(images)
        #     output = output.cpu()
        #     target = target.cpu()
        #     # output = output/np.linalg.norm(output, axis=1).reshape(-1, 1)
        #     # output = scaler.transform(output)
        #     outputs_list.append(output)
        #     targets_list.append(target)

        # output = torch.cat(outputs_list).numpy()
        # target = torch.cat(targets_list).numpy()

        # print("### Calc Error")
        # d, _ = calc_errorval(output, sub_vec)

        # plt.figure()
        # plt.ylim(0, 1)
        # plt.scatter(range(len(d)), d, c=target, cmap=cm.nipy_spectral)
        # plt.colorbar()
        # plt.savefig('d_{}_{}.png'.format(str(test_threshold), k_count))

    print("Train good Time {}".format(time.time() - start_time))
    return np.mean(good_d) + good_stddev*10, (sub_vec, sub_val)


def train_defective(train_loader, val_loader, model, args, threshold, sub_vec, sub_val):

    # switch to evaluate mode
    model.eval()

    # Train
    # 異常データを追加していく
    with torch.no_grad():
        assert train_loader.batch_size == 1
        # 正常の学習に使用した枚数を取得
        train_loader.sampler.set_state("good")
        n_good = len(train_loader)
        train_loader.sampler.set_state("defective")  # 異常データを１枚ずつ
        # 異常の学習に使用した画像をテスト用に保存
        learned_defect_image = []
        p0 = None
        for i, (images, target) in enumerate(train_loader):  # TODO batch dependancies
            learned_defect_image.append(images.numpy()[0])
            saveimg(learned_defect_image, args, prefix="learned")
            # DEBUG_SHOW(images)
            end = time.time()
            print("\n#" + "-"*30 + ' ' + str(i) + 'train defective ' + "-"*30 + "#")
            # Reverse order
            images = torch.from_numpy(images.numpy()[::-1].copy())
            target = torch.from_numpy(target.numpy()[::-1].copy())

            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            if args.uselayer:
                output = model(images)
            else:
                # 画像を表示したい
                output = images.reshape(images.shape[0], -1)
            output = output.cpu().numpy()
            target = target.cpu().numpy()

            print("### Calc eigenvalue, eigenvector")
            print("input shape (model output)", output.shape)

            # インクリメンタルPCAを用いた更新
            # incremental
            print("Incremental")
            e_vec, e_val = incremental_PCA(output, sub_vec, sub_val, i+n_good, args, state="adddefective")

            print("e_vec.shape", e_vec.shape)
            # print("e_vec", e_vec)
            print("e_val.shape", e_val.shape)
            # print("e_val", e_val)

            # subspace
            # # 次元を落として部分空間を作成
            # # print("### Calc Subspace")
            # sub_vec, sub_val = calc_sub_vec(e_vec, e_val, threshold, args)
            # # 落とさない
            # sub_vec, sub_val = e_vec, e_val
            # 異常である最後を落とす
            sub_vec = e_vec.T[:-1].T
            sub_val = e_val[:-1]

            # 部分空間の更新を確認
            p1 = sub_vec @ sub_vec.T
            if p0 is not None:
                p_e_val, _ = np.linalg.eigh(p0 @ p1 @ p0)
                p_e_val = p_e_val[::-1]
                print(p_e_val[:len(sub_val)])
            p0 = p1

            stime = time.time() - end
            print("Time : ", f"{stime:.3f}")

            testall(val_loader, model, args, threshold, sub_vec, sub_val, learned_defect_image, prefix="test_{}".format(i))


def testall(val_loader, model, args, threshold, sub_vec, sub_val, learned_image, prefix=""):
    start_time = time.time()
    # -------------------test---------------------------------
    # 生成した正常部分空間をテストデータを含めて評価(可視化用)
    outputs_list = []
    targets_list = []
    path_list = []
    for i, (images, target, path) in enumerate(val_loader):
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        if args.uselayer:
            output = model(images)
        else:
            # 画像を表示したい
            output = images.reshape(images.shape[0], -1)
        output = output.cpu()
        target = target.cpu()
        outputs_list.append(output)
        targets_list.append(target)
        path_list.append(path)

    output = torch.cat(outputs_list).numpy()
    target = torch.cat(targets_list).numpy()

    print("### Calc Error")
    d, _ = calc_errorval(output, sub_vec)

    # UMAPの計算
    # calc_umap(output, target, prefix=prefix)

    # ROCとAUCの計算
    labelg = val_loader.dataset.class_to_idx["good"]
    labeld = val_loader.dataset.class_to_idx["defective"]
    roc_label = np.where((target == labelg) | (target == labeld))
    roc_d = d[roc_label]
    roc_targetd = target[roc_label]
    calc_roc(roc_targetd, roc_d, args, prefix=prefix)

    # # 固有値のプロット
    # plt.figure()
    # plt.plot(sub_val)
    # plt.savefig('SUB_VAL_{}.png'.format(prefix))
    # plt.close()

    # 学習済みの画像も追加
    # output_learned = model(torch.from_numpy(np.asarray(learned_image)))
    # output_learned = output_learned.cpu().numpy()
    # d_learned, _ = calc_errorval(output_learned, sub_vec)
    # target_learned = np.full(len(d_learned), 5)
    # d_plot = np.concatenate((d, d_learned))
    # target_plot = np.concatenate((target, target_learned))
    # print(d.shape)
    # print(np.asarray(d_learned).shape)
    # print(d_plot.shape)
    # print(target.shape)
    # print(np.asarray(target_learned).shape)
    # print(target_plot.shape)
    learned_indices = []
    for i, img in enumerate(images):
        for l_img in learned_image:
            if np.allclose(img.numpy(), l_img):
                learned_indices.append(i)
                break
    target[learned_indices] = 5
    target_plot = target

    # 異常度のプロットと保存
    plt.figure()
    # plt.ylim(0, 1)
    # plt.ylim(0, 0.2)
    plt.scatter(range(len(d)), d, c=target_plot, cmap=cm.nipy_spectral)
    plt.colorbar()
    plt.title(str(val_loader.dataset.classes))
    plt.savefig(os.path.join(args.pngdir, 'D_{}_{}.png'.format(prefix, str(threshold))))
    plt.close()
    with open(os.path.join(args.pngdir, "d_list.csv"), "w") as f:
        temp = (d, target_plot, path_list[0])
        csvdata = list(zip(*temp))
        writer = csv.writer(f)
        for s in csvdata:
            writer.writerow(s)

    print("Test Time : {}".format(time.time() - start_time))


def calc_SVD(features):
    # svd
    U, s, V = np.linalg.svd(features)
    e_val = s**2 / features.shape[0]
    e_vec = V.T
    return e_vec, e_val


def calc_sub_vec(e_vec, e_val, threshold, args):
    print("e_vec.shape", e_vec.shape)
    print("e_val.shape", e_val.shape)
    print("e_val", e_val)

    # 固有値の絶対値を取る
    # e_val = np.abs(e_val)

    sum_all = np.sum(e_val)
    sum_val = np.array([np.sum(e_val[:i])/sum_all for i in range(1, len(e_val)+1)])
    # # 固有値の値でソート
    # sort_index = np.argsort(e_val)[::-1]
    # e_val = e_val[sort_index]
    # e_vec = e_vec.T[sort_index].T
    # 部分空間を作成
    r = int(min(np.where(sum_val >= threshold)[0])+1)
    sub_vec = e_vec.T[:r].T
    sub_val = e_val[:r]

    plt.figure()
    plt.plot(sum_val)
    plt.savefig(os.path.join(args.pngdir, 'SUM_E_VAL.png'))
    plt.close()
    print("sub_vec.shape", sub_vec.shape)
    # print("sub_vec", sub_vec)
    print("sub_val.shape", sub_val.shape)
    print("sub_val", sub_val)
    return sub_vec, sub_val


def incremental_PCA(features, sub_vec, sub_val, n, args, state="addgoodonly"):
    h = np.reshape(features[0] - sub_vec @ sub_vec.T @ features[0], (-1, 1))
    h_norm = np.linalg.norm(h)
    h_hat = h / h_norm if h_norm > 0.1 else np.zeros(h.shape)
    g = np.reshape(sub_vec.T @ features[0], (-1, 1))
    gamma = h_hat.T @ features[0]
    # calc L
    lamd = np.diag(sub_val)  # del new vec
    l1 = np.block([[lamd, np.zeros((lamd.shape[0], 1))],
                  [np.zeros((1, lamd.shape[1])), 0]])
    # l1 = lamd  # del new vec
    l2 = np.block([[g @ g.T, gamma * g],
                   [gamma * g.T, gamma**2]])
    # l2 = g.reshape(-1,1) @ g.reshape(1,-1)

    # 正常のみに正常を追加するインクリメンタルPCA
    if state == "addgoodonly":
        ll = (n/(n+1) * l1) + (1/(n+1) * l2)
    # 正常を追加するインクリメンタルPCA
    elif state == "addgood":
        ll = l1 + (1/(n+1) * l2)
    # 異常を追加するインクリメンタルPCA
    elif state == "adddefective":
        defect_c = args.prmc
        ll = l1 - (defect_c/(n+1) * l2)

    # calc rotation matrix
    e_val, rot = np.linalg.eigh(ll)
    e_val = e_val[::-1]
    # update e_vec
    print(rot)
    e_vec = np.block([sub_vec, h_hat]) @ rot
    # e_vec = sub_vec @ rot  # del new vec
    e_vec = e_vec.T[::-1].T
    return e_vec, e_val


def calc_errorval(features, sub_vec):
    # Y_rec
    y = features @ sub_vec @ sub_vec.T

    # ユークリッド距離
    # dist = np.linalg.norm(output-y, axis=1)
    # cos
    dist = [np.inner(oi, yi) / (np.linalg.norm(oi) * np.linalg.norm(yi)) for oi, yi in zip(features, y)]
    # dist = []
    # for oi, yi in zip(features, y):
    #     norm_oi = np.linalg.norm(oi)
    #     norm_yi = np.linalg.norm(yi)
    #     if (norm_yi < 1e-10):
    #         dist.append(0)
    #     else:
    #         dist.append(np.inner(oi, yi) / (norm_oi * norm_yi))

    # sin
    dist = np.clip(np.asarray(dist)**2, 0, 1)
    dist = 1-dist
    # dist = np.sqrt(1-np.power(dist, 2))
    print(dist)

    # 分散
    # good_variarance = dist.var()
    # 標準偏差
    stddev = dist.std()
    return dist, stddev


def calc_roc(target, d, args, prefix=""):

    fpr, tpr, thresholds = metrics.roc_curve(target, (d*(-1)))
    auc = metrics.roc_auc_score(target, (d*(-1)))

    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (AUC = %.2f)' % auc)
    plt.legend()
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)
    plt.savefig(os.path.join(args.pngdir, 'ROC_{}.png'.format(prefix)))
    plt.close()
    print("auc", auc, d.shape)


def calc_umap(features, target, args, prefix=""):
    # umap
    embedding = umap.UMAP(n_neighbors=5, random_state=0).fit_transform(features)
    plt.figure()
    plt.scatter(embedding[:, 0], embedding[:, 1], c=target, cmap=cm.nipy_spectral)
    plt.colorbar()
    plt.savefig(os.path.join(args.pngdir, 'UMAP_{}.png'.format(prefix)))
    plt.close()


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
