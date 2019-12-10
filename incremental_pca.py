import argparse
import os
import sys
# import csv
import shutil
import random
import time
import warnings
import umap
import pickle
import math
# from scipy.sparse.csgraph import connected_components
import scipy.stats as stats
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

import kernellib
import hyperprmselect as hyp

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
parser.add_argument('--useparam', default=0, type=float)
parser.add_argument('--mul_sig', default=3.0, type=float)
parser.add_argument('--usekernel', action='store_true')
parser.add_argument('--usepseudo', action='store_true')
parser.add_argument('--usereject', action='store_true')
parser.add_argument('--flatten', action='store_true')
parser.add_argument('--judge', action='store_true')
parser.add_argument('--gamma', action='store_true')
parser.add_argument('--n_images', default=None, type=int)
parser.add_argument('--clear_gn', action='store_true')

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
    # unnorm = UnNormalize(mean=[0.485, 0.456, 0.406],
    #                      std=[0.229, 0.224, 0.225])
    for i, img in enumerate(images):
        # img = deepcopy(unnorm(img[0]))
        # img = img.numpy()
        plt.figure()
        plt.imshow(np.transpose(img, (1, 2, 0)))
        plt.savefig(os.path.join(args.pngdir, prefix+str(i)+'.png'))
        plt.close()


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


class KFoldSampler(torch.utils.data.Sampler):

    def __init__(self, data_source, args, k=5, seed=None):
        self.data_source = data_source
        self.good_index = [i for i, (x, y) in enumerate(data_source.imgs) if data_source.classes[int(y)] == "good"]
        self.defective_index = [i for i, (x, y) in enumerate(data_source.imgs) if data_source.classes[int(y)] == "defective"]
        if args.n_images is not None:
            self.good_index = np.tile(self.good_index, np.max((1, np.ceil(args.n_images/len(self.good_index)).astype(np.int))))
            self.good_index = self.good_index[:args.n_images]
            print("n_images : ", len(self.good_index))
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


class SaveAUCGraph(object):

    def __init__(self, pngpath):
        self.pngpath = pngpath
        self.auclist = []
        self.acclist = []
        self.dlist = []
        self.updlist = []

    def add(self, auc):
        self.auclist.append(auc)
        self.save()

    def addacc(self, acc):
        self.acclist.append(acc)
        self.saveacc()

    def addd(self, d):
        self.dlist.append(d)
        self.saved()

    def addupd(self, upd):
        self.updlist.append(upd)
        self.saveupd()

    def save(self):
        plt.figure()
        plt.ylim(0, 1)
        plt.plot(self.auclist, label="Max = %.2f" % max(self.auclist))
        plt.legend()
        plt.savefig(os.path.join(self.pngpath, 'AUClog{}.png'.format(os.path.basename(self.pngpath))))
        plt.close()
        with open(os.path.join(self.pngpath, 'AUClog{}.pcl'.format(os.path.basename(self.pngpath))), 'wb') as f:
            pickle.dump(self.auclist, f)

    def saveacc(self):
        plt.figure()
        plt.ylim(0, 1)
        plt.plot(self.acclist, label="Max = %.2f" % max(self.acclist))
        plt.legend()
        plt.savefig(os.path.join(self.pngpath, 'ACClog{}.png'.format(os.path.basename(self.pngpath))))
        plt.close()
        with open(os.path.join(self.pngpath, 'ACClog{}.pcl'.format(os.path.basename(self.pngpath))), 'wb') as f:
            pickle.dump(self.acclist, f)

    def saved(self):
        plt.figure()
        plt.plot(self.dlist)
        plt.savefig(os.path.join(self.pngpath, 'Dlog{}.png'.format(os.path.basename(self.pngpath))))
        plt.close()
        with open(os.path.join(self.pngpath, 'Dlog{}.pcl'.format(os.path.basename(self.pngpath))), 'wb') as f:
            pickle.dump(self.dlist, f)

    def saveupd(self):
        plt.figure()
        plt.plot(self.updlist)
        plt.savefig(os.path.join(self.pngpath, 'UPDlog{}.png'.format(os.path.basename(self.pngpath))))
        plt.close()
        with open(os.path.join(self.pngpath, 'UPDlog{}.pcl'.format(os.path.basename(self.pngpath))), 'wb') as f:
            pickle.dump(self.updlist, f)


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


class FlattenMobilenetV2(nn.Module):
    def __init__(self, n_layer, flatten):
        super(FlattenMobilenetV2, self).__init__()
        # mobilenetv2
        model = torch.hub.load('pytorch/vision', 'mobilenet_v2', pretrained=True)
        model = nn.Sequential(*list(model.features.children())[:n_layer])
        print(model)
        print(len(model))
        self.model = model
        self.flatten = flatten

    def forward(self, x):
        x = self.model(x)
        if self.flatten:
            x = torch.flatten(x, 1)
        else:
            x = x.mean([2, 3])
        return x


def get_model_layer(n_layer, args, ngpus_per_node):
    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'test')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if n_layer:
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

    train_sampler = KFoldSampler(train_dataset, seed=args.seed, args=args, k=5)

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

    if n_layer:
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

    if not n_layer:
        return None, train_loader1000, train_loader1, val_loader

    model = FlattenMobilenetV2(n_layer, args.flatten)
    torch.onnx.export(model, torch.randn(10, 3, 224, 224), 'mobilenet_v2.onnx', verbose=False)

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

    # switch to evaluate mode
    model.eval()

    return model, train_loader1000, train_loader1, val_loader


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

    cudnn.benchmark = True

    aucg = SaveAUCGraph(args.pngdir)

    #  モデルの層の決定
    if args.uselayer:
        layer_list = [args.uselayer]
    else:
        layer_list = [0, 6, 12, 18]
    results = {}
    d_good_list = {}
    th_gooddef_results = {}
    for test_layer in layer_list:
        model, train_loader1000, train_loader1, val_loader = get_model_layer(test_layer, args, ngpus_per_node)
        # しきい値の決定
        if args.useparam:
            thresholds_list = [args.useparam]
        else:
            thresholds_list = [0.85, 0.9, 0.95, 0.99, 0.999]
        for test_threshold in thresholds_list:
            d_average = 0
            th_gooddef_average = 0.0
            for k_count in range(args.kfold):
                # Train
                print("\n#" + "="*30 + ' train_good SVD ' + str(test_layer) + '/' + str(test_threshold) + '/' + str(k_count) + ' ' + "="*30 + "#")
                train_loader1000.sampler.set_k(k_count)
                print("train index", train_loader1000.sampler.train)
                print("val_index", train_loader1000.sampler.val)
                d_result, d_good, th_gooddef = train_good(train_loader1000, val_loader,
                                                          model, args,
                                                          k_count, test_threshold, "good_train")
                d_average += d_result
                th_gooddef_average += th_gooddef if th_gooddef is not None else 0.0
                d_good_list[(test_layer, test_threshold, k_count)] = d_good
                print("d_good, th_gooddef", np.mean(d_good), th_gooddef)
            results[(test_layer, test_threshold)] = d_average/args.kfold
            th_gooddef_results[(test_layer, test_threshold)] = th_gooddef_average/args.kfold
            print(str(test_threshold), d_average/args.kfold)
            print(str(test_threshold), th_gooddef_average/args.kfold)
        print(results)
    with open(os.path.join(args.pngdir, 'paramsearch.pcl'), 'wb') as f:
        pickle.dump(results, f)
    with open(os.path.join(args.pngdir, 'd_good_list.pcl'), 'wb') as f:
        pickle.dump(d_good_list, f)
    layer, threshold = min(results, key=results.get)
    gooddef_threshold = th_gooddef_results[(layer, threshold)]
    print(layer, threshold, results[(layer, threshold)], gooddef_threshold)

    # 決めたしきい値を用いて、すべてのデータで正常部分空間を作る
    model, train_loader1000, train_loader1, val_loader = get_model_layer(layer, args, ngpus_per_node)
    print(train_loader1000, val_loader,
          model, args,
          k_count, threshold, "good",
          aucg, True)

    sub_vec, sub_val, good_mean, good_stddev, kernel_inst = train_good(train_loader1000, val_loader,
                                                                       model, args,
                                                                       k_count, threshold, "good",
                                                                       gooddef_threshold=gooddef_threshold,
                                                                       aucg=aucg,
                                                                       final=True)

    # それに対して異常データを追加していく
    train_defective(train_loader1, val_loader, model, args, threshold, sub_vec, sub_val, good_mean, good_stddev,
                    gooddef_threshold=gooddef_threshold, aucg=aucg, kernel_inst=kernel_inst)


def train_good(train_loader, val_loader, model,
               args, k_count, test_threshold, sampler_state, gooddef_threshold=None, aucg=None, final=False):
    start_time = time.time()

    # Train
    # 正常データのみを用いたトレーニング
    with torch.no_grad():
        sub_vec = None
        sub_val = None
        train_loader.sampler.set_state(sampler_state)
        outputs_list = []
        targets_list = []
        for i, (images, target) in enumerate(train_loader):  # TODO batch dependancies
            learned_good_image = images.numpy()
            # DEBUG_SHOW(images)
            end = time.time()
            # Reverse order
            images = torch.from_numpy(images.numpy()[::-1].copy())
            target = torch.from_numpy(target.numpy()[::-1].copy())

            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            if model is not None:
                output = model(images)
            else:
                # 画像を表示したい
                output = images.reshape(images.shape[0], -1)
            output = output.cpu()
            target = target.cpu()

            outputs_list.append(output)
            targets_list.append(target)

        output = torch.cat(outputs_list).numpy()
        target = torch.cat(targets_list).numpy()

        # カーネルを使用
        kernel_inst = None
        if args.usekernel:
            kernel_inst = kernellib.UseKernel(kernel="rbf")
            kernel_inst.fit(output)
            output = kernel_inst.transform(output)

        # normlization
        # scaler = MinMaxScaler()
        # print(scaler.fit(output))
        # output = scaler.transform(output)
        # output = output.numpy()
        # output = output/np.linalg.norm(output, axis=1).reshape(-1, 1)

        print("### Calc eigenvalue, eigenvector")
        print("input shape (model output)", output.shape)
        # SVDを用いた正常部分空間作成
        e_vec, e_val = calc_SVD(output)

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

        # if sampler_state == 'good':
        if final:
            testall(val_loader, model, args, test_threshold,
                    sub_vec, sub_val, learned_good_image, prefix="good", gooddef_threshold=gooddef_threshold, aucg=aucg, kernel_inst=kernel_inst)
        elif k_count == 0:
            testall(val_loader, model, args, test_threshold,
                    sub_vec, sub_val, learned_good_image, prefix="good_crossval_{}".format(test_threshold), aucg=None, kernel_inst=kernel_inst)

        if model is None:
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
        # DEBUG 寄与率決定の判断をgoodではなくgood_valでやる
        # train_loader.sampler.set_state("good")
        train_loader.sampler.set_state("good_val")
        outputs_list = []
        targets_list = []
        for i, (images, target) in enumerate(train_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            if model is not None:
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

        if args.usekernel:
            output = kernel_inst.transform(output)

        print("### Calc Error")
        good_d, good_stddev = calc_errorval(output, sub_vec)

        gooddef_threshold = None
        if args.judge and not final:
            # pisson分布を仮定して正常と異常の閾値を決定
            print("np.mean(good_d), min, max, np.std(good_d)", np.mean(good_d), np.min(good_d), np.max(good_d), np.std(good_d))
            print("len(good_d)", len(good_d))
            if args.gamma:
                a_hat, loc_hat, scale_hat = stats.gamma.fit(good_d)
                print("a_hat, loc_hat, scale_hat", a_hat, loc_hat, scale_hat)
                lambda_hat = stats.gamma.mean(a_hat, loc=loc_hat, scale=scale_hat)
                p_mean, p_var, p_skew, p_kurt = stats.gamma.stats(a_hat, moments='mvsk', scale=scale_hat, loc=loc_hat)
                gooddef_threshold = p_mean + args.mul_sig * math.sqrt(p_var)
                print("good or defect threshold", "%.5f" % gooddef_threshold, lambda_hat)
                # print("poisson mean", "%.5f" % lambda_hat)
                # print("good_d mean", "%.5f" % np.mean(good_d))
                xs = np.linspace(stats.gamma.ppf(0.01, a_hat, scale=scale_hat, loc=loc_hat),
                                 stats.gamma.ppf(0.99, a_hat, scale=scale_hat, loc=loc_hat), 100)
                ps_hat = stats.gamma.pdf(xs, a_hat, loc=loc_hat, scale=scale_hat)
                fig = plt.figure(1, figsize=(12, 8))
                ax = fig.add_subplot(111)
                ax.plot(xs, ps_hat, 'g-', lw=2, label='fitted pdf')
                ax.hist(good_d, density=True, histtype='stepfilled', alpha=0.2, bins=20)
                ax.grid(True)
                fig.savefig(os.path.join(args.pngdir, "poisson_{}.png".format(k_count)))

            else:  # norm
                print("norm")
                loc_hat, scale_hat = stats.norm.fit(good_d)
                print("loc_hat, scale_hat", loc_hat, scale_hat)
                lambda_hat = stats.norm.mean(loc=loc_hat, scale=scale_hat)
                p_mean, p_var, p_skew, p_kurt = stats.norm.stats(moments='mvsk', scale=scale_hat, loc=loc_hat)
                gooddef_threshold = p_mean + args.mul_sig * math.sqrt(p_var)
                print("good or defect threshold", "%.5f" % gooddef_threshold, lambda_hat)
                # print("poisson mean", "%.5f" % lambda_hat)
                # print("good_d mean", "%.5f" % np.mean(good_d))
                xs = np.linspace(stats.norm.ppf(0.01, scale=scale_hat, loc=loc_hat),
                                 stats.norm.ppf(0.99, scale=scale_hat, loc=loc_hat), 100)
                ps_hat = stats.norm.pdf(xs, loc=loc_hat, scale=scale_hat)
                fig = plt.figure(1, figsize=(12, 8))
                ax = fig.add_subplot(111)
                ax.plot(xs, ps_hat, 'g-', lw=2, label='fitted pdf')
                ax.hist(good_d, density=True, histtype='stepfilled', alpha=0.2, bins=20)
                ax.grid(True)
                fig.savefig(os.path.join(args.pngdir, "norm_{}.png".format(k_count)))

        if args.usepseudo:
            # ----------------validation------------------------
            # 生成した部分空間を擬似データのAUCで評価（寄与率決定用）
            # 正常データから疑似データを生成
            pseudo_output, pseudo_target = hyp.makedata(output)
            print("pseudo_output", pseudo_output.shape)
            print("pseudo_target", pseudo_target.shape)
            pseudo_d, pseudo_stddev = calc_errorval(pseudo_output, sub_vec)
            calc_roc(pseudo_target, pseudo_d, args, prefix="pseudo_{}".format(test_threshold), aucg=None)

    print("Train good Time {}".format(time.time() - start_time))

    # 実際に使う部分空間を返す
    if final:
        return sub_vec, sub_val, np.mean(good_d), good_stddev, kernel_inst
    else:
        return np.mean(good_d) + good_stddev*10, good_d, gooddef_threshold


def train_defective(train_loader, val_loader, model, args, threshold, sub_vec, sub_val, good_mean, good_stddev,
                    aucg, gooddef_threshold, kernel_inst=None):

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
            if model is not None:
                output = model(images)
            else:
                # 画像を表示したい
                output = images.reshape(images.shape[0], -1)
            output = output.cpu().numpy()
            target = target.cpu().numpy()

            if args.usekernel:
                output = kernel_inst.transform(output)

            print("### Calc eigenvalue, eigenvector")
            print("input shape (model output)", output.shape)

            if args.usereject:
                # # dを用いて学習するかどうかを判定
                def_d, _ = calc_errorval(output, sub_vec)
                # if ((good_mean + 3*good_mean) >= def_d[0]):
                # gooddef_thresholdを用いて学習するかどうか判定
                # 閾値を下回ったらReject
                if (gooddef_threshold >= def_d):
                    print("[[[Reject]]]")
                    continue

            learned_defect_image.append(images.numpy()[0])
            # learned_defect_image.append(images)
            saveimg(learned_defect_image, args, prefix="learned")

            # インクリメンタルPCAを用いた更新
            # incremental
            print("Incremental")
            if args.clear_gn: # clear good number of image
                nn = i
            else:
                nn = i+n_good
            e_vec, e_val = incremental_PCA(output, sub_vec, sub_val, nn, args, state="adddefective")

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
            # p1 = sub_vec @ sub_vec.T
            p1 = sub_vec
            if p0 is not None:
                upd = np.linalg.norm(p0.T @ p1)
                print("p0.T @ p1", upd)
                aucg.addupd(upd)
                # p_e_val, _ = np.linalg.eigh(p0 @ p1 @ p0)
                # p_e_val = p_e_val[::-1]
            p0 = p1

            d, _ = calc_errorval(output, sub_vec)
            aucg.addd(d)
            print("d", d)

            stime = time.time() - end
            print("Time : ", f"{stime:.3f}")

            testall(val_loader, model, args, threshold, sub_vec,
                    sub_val, learned_defect_image, prefix="test_{}".format(i),
                    gooddef_threshold=gooddef_threshold,
                    aucg=aucg, kernel_inst=kernel_inst)


def testall(val_loader, model, args, threshold, sub_vec, sub_val, learned_image, prefix="", gooddef_threshold=None, aucg=None, kernel_inst=None):
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
        if model is not None:
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

    if args.usekernel:
        output = kernel_inst.transform(output)

    print("### Calc Error")
    d, _ = calc_errorval(output, sub_vec)

    # UMAPの計算
    # calc_umap(output, target, prefix=prefix)

    # テスト用データの取り出し
    labelg = val_loader.dataset.class_to_idx["good"]
    labeld = val_loader.dataset.class_to_idx["defective"]
    roc_label = np.where((target == labelg) | (target == labeld))
    roc_d = d[roc_label]
    roc_targetd = target[roc_label]

    # ROCとAUCの計算
    calc_roc(roc_targetd, roc_d, args, prefix=prefix, aucg=aucg)

    # gooddef_thresholdを使って異常検知
    if gooddef_threshold is not None:
        judge = np.zeros(roc_d.shape[0])
        judge[np.where(roc_d <= gooddef_threshold)] = labelg
        judge[np.where(roc_d > gooddef_threshold)] = labeld
        judge_diff = np.sum(roc_targetd == judge)
        print("judge_diff", judge_diff)
        acc = judge_diff/roc_d.shape[0]
        print("acc", acc)
        if aucg is not None:
            aucg.addacc(acc)

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

    latest_learned_image = None
    learned_indices = []
    for i, img in enumerate(images):
        for l_img in learned_image:
            if np.allclose(img.numpy(), l_img):
                learned_indices.append(i)
                break
        if np.allclose(img.numpy(), learned_image[-1]):
            latest_learned_image = i
    print("learned_indices", learned_indices)
    if latest_learned_image is None:
        print("latest_learned_image is nOne")
    else:
        print(latest_learned_image)
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
    temp = (d, target_plot, path_list[0], latest_learned_image)
    # with open(os.path.join(args.pngdir, "d_list.csv"), "w") as f:
    #     csvdata = list(zip(*temp))
    #     writer = csv.writer(f)
    #     for s in csvdata:
    #         writer.writerow(s)
    with open(os.path.join(args.pngdir, 'D_{}_{}.pcl'.format(prefix, str(threshold))), 'wb') as f:
        pickle.dump(temp, f)

    print("Test Time : {}".format(time.time() - start_time))


def calc_SVD(features):
    # svd
    print("SVD...", end=' ')
    sys.stdout.flush()
    U, s, V = np.linalg.svd(features)
    e_val = s**2 / features.shape[0]
    e_vec = V.T
    print("finish")
    return e_vec, e_val


def calc_sub_vec(e_vec, e_val, threshold, args):
    print("e_vec.shape", e_vec.shape)
    print("e_val.shape", e_val.shape)
    # print("e_val", e_val)

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
    # print("sub_val", sub_val)
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
    # print(rot)
    e_vec = np.block([sub_vec, h_hat]) @ rot
    # e_vec = sub_vec @ rot  # del new vec
    e_vec = e_vec.T[::-1].T
    return e_vec, e_val


def calc_errorval(features, sub_vec):
    COS = True
    if COS:
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
        # print(dist)

        # 分散
        # good_variarance = dist.var()
        # 標準偏差
    else:
        features_norm = np.linalg.norm(features, axis=1).reshape(-1, 1)
        dist = np.sum(1-((features @ sub_vec)/features_norm)**2, axis=1)
    stddev = dist.std()
    return dist, stddev


def calc_roc(target, d, args, prefix="", aucg=None):

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
    if aucg is not None:
        aucg.add(auc)
    with open(os.path.join(args.pngdir, 'ROC_{}.pcl'.format(prefix)), 'wb') as f:
        pickle.dump((fpr, tpr), f)


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
