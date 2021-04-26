"""
This code is the main training code.
"""
import argparse
import itertools
import logging
import os
import sys

import paddle
from paddle.optimizer.lr import CosineAnnealingDecay, MultiStepDecay
from paddle.io import DataLoader, ComposeDataset

from vision.datasets.voc_dataset import VOCDataset
from vision.nn.multibox_loss import MultiboxLoss
from vision.ssd.config.fd_config import define_img_size
from vision.utils.misc import str2bool, Timer, freeze_net_layers, store_labels

parser = argparse.ArgumentParser(
    description='train With Pytorch')

parser.add_argument("--dataset_type", default="voc", type=str,
                    help='Specify dataset type. Currently support voc.')

parser.add_argument('--datasets', nargs='+', help='Dataset directory path')
parser.add_argument('--validation_dataset', help='Dataset directory path')
parser.add_argument('--balance_data', action='store_true',
                    help="Balance training data by down-sampling more frequent labels.")

parser.add_argument('--net', default="RFB",
                    help="The network architecture ,optional(RFB , slim)")
parser.add_argument('--freeze_base_net', action='store_true',
                    help="Freeze base net layers.")
parser.add_argument('--freeze_net', action='store_true',
                    help="Freeze all the layers except the prediction head.")

# Params for SGD
parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--base_net_lr', default=None, type=float,
                    help='initial learning rate for base net.')
parser.add_argument('--extra_layers_lr', default=None, type=float,
                    help='initial learning rate for the layers not in base net and prediction heads.')

# Params for loading pretrained basenet or checkpoints.
parser.add_argument('--base_net',
                    help='Pretrained base model')
parser.add_argument('--pretrained_ssd', help='Pre-trained base model')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')

# Scheduler
parser.add_argument('--scheduler', default="multi-step", type=str,
                    help="Scheduler for SGD. It can one of multi-step and cosine")

# Params for Multi-step Scheduler
parser.add_argument('--milestones', default="80,100", type=str,
                    help="milestones for MultiStepLR")

# Params for Cosine Annealing
parser.add_argument('--t_max', default=120, type=float,
                    help='T_max value for Cosine Annealing Scheduler.')

# Train params
parser.add_argument('--batch_size', default=24, type=int,
                    help='Batch size for training')
parser.add_argument('--num_epochs', default=200, type=int,
                    help='the number epochs')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--validation_epochs', default=5, type=int,
                    help='the number epochs')
parser.add_argument('--debug_steps', default=100, type=int,
                    help='Set the debug log output frequency.')
parser.add_argument('--use_cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')

parser.add_argument('--checkpoint_folder', default='models/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--log_dir', default='./models/Ultra-Light(1MB)_&_Fast_Face_Detector/logs',
                    help='lod dir')
parser.add_argument('--cuda_index', default="0", type=str,
                    help='Choose cuda index.If you have 4 GPUs, you can set it like 0,1,2,3')
parser.add_argument('--power', default=2, type=int,
                    help='poly lr pow')
parser.add_argument('--overlap_threshold', default=0.35, type=float,
                    help='overlap_threshold')
parser.add_argument('--optimizer_type', default="SGD", type=str,
                    help='optimizer_type')
parser.add_argument('--input_size', default=320, type=int,
                    help='define network input size,default optional value 128/160/320/480/640/1280')

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
args = parser.parse_args()

input_img_size = args.input_size  # define input size ,default optional(128/160/320/480/640/1280)
logging.info("inpu size :{}".format(input_img_size))
define_img_size(input_img_size)  # must put define_img_size() before 'import fd_config'

from vision.ssd.config import fd_config
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd
from vision.ssd.mb_tiny_fd import create_mb_tiny_fd
from vision.ssd.ssd import MatchPrior

# GPU settings
if paddle.is_compiled_with_cuda() and args.use_cuda:
    paddle.set_device("gpu")
    logging.info("Use gpu.")
else:
    paddle.set_device("cpu")
    logging.info("Use cpu.")


def lr_poly(base_lr, iter):
    """
    learning rate poly decay
    Args:
        base_lr: the base learning rate
        iter: the current iter

    Returns:
        the current learning rate
    """
    return base_lr * ((1 - float(iter) / args.num_epochs) ** (args.power))


def adjust_learning_rate(optimizer, i_iter):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(args.lr, i_iter)
    optimizer.param_groups[0]['lr'] = lr


def train(loader, net, criterion, optimizer, debug_steps=100, epoch=-1):
    net.train()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    for i, data in enumerate(loader):
        print(".", end="", flush=True)
        images, boxes, labels = data

        optimizer.clear_grad()
        confidence, locations = net(images)
        regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)  # TODO CHANGE BOXES
        loss = regression_loss + classification_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.numpy().tolist()[0]
        running_regression_loss += regression_loss.numpy().tolist()[0]
        running_classification_loss += classification_loss.numpy().tolist()[0]
        if i and i % debug_steps == 0:
            print(".", flush=True)
            avg_loss = running_loss / debug_steps
            avg_reg_loss = running_regression_loss / debug_steps
            avg_clf_loss = running_classification_loss / debug_steps
            logging.info(
                f"Epoch: {epoch}, Step: {i}, " +
                f"Average Loss: {avg_loss:.4f}, " +
                f"Average Regression Loss {avg_reg_loss:.4f}, " +
                f"Average Classification Loss: {avg_clf_loss:.4f}"
            )

            running_loss = 0.0
            running_regression_loss = 0.0
            running_classification_loss = 0.0


def test(loader, net, criterion):
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    for _, data in enumerate(loader):
        images, boxes, labels = data
        num += 1

        with paddle.no_grad():
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

        running_loss += loss.numpy().tolist()[0]
        running_regression_loss += regression_loss.numpy().tolist()[0]
        running_classification_loss += classification_loss.numpy().tolist()[0]
    return running_loss / num, running_regression_loss / num, running_classification_loss / num


if __name__ == '__main__':
    timer = Timer()

    logging.info(args)
    if args.net == 'slim':
        create_net = create_mb_tiny_fd
        config = fd_config
    elif args.net == 'RFB':
        create_net = create_Mb_Tiny_RFB_fd
        config = fd_config
    else:
        logging.fatal("The net type is wrong.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, args.overlap_threshold)

    test_transform = TestTransform(config.image_size, config.image_mean_test, config.image_std)

    if not os.path.exists(args.checkpoint_folder):
        os.makedirs(args.checkpoint_folder)
    logging.info("Prepare training datasets.")
    datasets = []
    for dataset_path in args.datasets:
        if args.dataset_type == 'voc':
            dataset = VOCDataset(dataset_path, transform=train_transform,
                                 target_transform=target_transform)
            label_file = os.path.join(args.checkpoint_folder, "voc-model-labels.txt")
            store_labels(label_file, dataset.class_names)
            num_classes = len(dataset.class_names)

        else:
            raise ValueError(f"Dataset tpye {args.dataset_type} is not supported.")
        datasets.append(dataset)
    logging.info(f"Stored labels into file {label_file}.")
    train_dataset = ComposeDataset(datasets)
    logging.info("Train dataset size: {}".format(len(train_dataset)))
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=True)
    logging.info("Prepare Validation datasets.")
    if args.dataset_type == "voc":
        val_dataset = VOCDataset(args.validation_dataset,
                                 transform=test_transform,
                                 target_transform=target_transform,
                                 is_test=True)
    logging.info("validation dataset size: {}".format(len(val_dataset)))

    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=False)
    logging.info("Build network.")
    net = create_net(num_classes)

    # add multigpu_train
    if paddle.distributed.get_world_size() != 1:
        # cuda_index_list = [int(v.strip()) for v in args.cuda_index.split(",")]
        paddle.distributed.init_parallel_env()
        net = paddle.DataParallel(net)
        logging.info("use multi gpu.")

    min_loss = -10000.0
    last_epoch = -1

    base_net_lr = args.base_net_lr if args.base_net_lr is not None else args.lr
    extra_layers_lr = args.extra_layers_lr if args.extra_layers_lr is not None else args.lr
    if args.freeze_base_net:
        logging.info("Freeze base net.")
        freeze_net_layers(net.base_net)
        params = itertools.chain(net.source_layer_add_ons.parameters(), net.extras.parameters(),
                                 net.regression_headers.parameters(), net.classification_headers.parameters())
        params = [
            {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()
            ), 'lr': extra_layers_lr},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}
        ]
    elif args.freeze_net:
        freeze_net_layers(net.base_net)
        freeze_net_layers(net.source_layer_add_ons)
        freeze_net_layers(net.extras)
        params = itertools.chain(net.regression_headers.parameters(), net.classification_headers.parameters())
        logging.info("Freeze all the layers except prediction heads.")
    else:
        params = [
            {'params': net.base_net.parameters(), 'lr': base_net_lr},
            {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()
            ), 'lr': extra_layers_lr},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}
        ]

    timer.start("Load Model")
    if args.resume:
        logging.info(f"Resume from the model {args.resume}")
        net.load(args.resume)
    elif args.base_net:
        logging.info(f"Init from base net {args.base_net}")
        net.init_from_base_net(args.base_net)
    elif args.pretrained_ssd:
        logging.info(f"Init from pretrained ssd {args.pretrained_ssd}")
        net.init_from_pretrained_ssd(args.pretrained_ssd)
    logging.info(f'Took {timer.end("Load Model"):.2f} seconds to load the model.')

    criterion = MultiboxLoss(config.priors, neg_pos_ratio=3, center_variance=0.1, size_variance=0.2)

    if args.optimizer_type != "Adam":
        if args.scheduler == 'multi-step':
            logging.info("Uses MultiStepLR scheduler.")
            milestones = [int(v.strip()) for v in args.milestones.split(",")]
            scheduler = MultiStepDecay(args.lr, milestones=milestones,
                                    gamma=0.1, last_epoch=last_epoch)
        elif args.scheduler == 'cosine':
            logging.info("Uses CosineAnnealingLR scheduler.")
            scheduler = CosineAnnealingDecay(args.lr, args.t_max, last_epoch=last_epoch)
        elif args.scheduler == 'poly':
            logging.info("Uses PolyLR scheduler.")
        else:
            logging.fatal(f"Unsupported Scheduler: {args.scheduler}.")
            parser.print_help(sys.stderr)
            sys.exit(1)

    if args.optimizer_type == "SGD":
        optimizer = paddle.optimizer.Momentum(learning_rate=scheduler,
                                              momentum=args.momentum,
                                              parameters=net.parameters(),               #######################,
                                              weight_decay=args.weight_decay)
    elif args.optimizer_type == "Adam":
        optimizer = paddle.optimizer.Adam(learning_rate=scheduler, parameters=params)
        logging.info("use Adam optimizer")
    else:
        logging.fatal(f"Unsupported optimizer: {args.scheduler}.")
        parser.print_help(sys.stderr)
        sys.exit(1)
    logging.info(f"Learning rate: {args.lr}, Base net learning rate: {base_net_lr}, "
                 + f"Extra Layers learning rate: {extra_layers_lr}.")

    logging.info(f"Start training from epoch {last_epoch + 1}.")
    for epoch in range(last_epoch + 1, args.num_epochs):
        if args.optimizer_type != "Adam":
            if args.scheduler != "poly":
                if epoch != 0:
                    scheduler.step()
        train(train_loader, net, criterion, optimizer, debug_steps=args.debug_steps, epoch=epoch)
        if args.scheduler == "poly":
            adjust_learning_rate(optimizer, epoch)
        logging.info("lr rate :{}".format(scheduler.get_lr()))

        if epoch % args.validation_epochs == 0 or epoch == args.num_epochs - 1:
            logging.info("lr rate :{}".format(scheduler.get_lr()))
            val_loss, val_regression_loss, val_classification_loss = test(val_loader, net, criterion)
            logging.info(
                f"Epoch: {epoch}, " +
                f"Validation Loss: {val_loss:.4f}, " +
                f"Validation Regression Loss {val_regression_loss:.4f}, " +
                f"Validation Classification Loss: {val_classification_loss:.4f}"
            )
            model_path = os.path.join(args.checkpoint_folder, f"{args.net}-Epoch-{epoch}-Loss-{val_loss}.ppmodel")
            net.save(model_path)
            logging.info(f"Saved model {model_path}")
