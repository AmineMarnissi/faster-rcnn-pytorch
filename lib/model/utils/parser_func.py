import argparse
from model.utils.config import cfg, cfg_from_file, cfg_from_list


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--log_ckpt_name', dest='log_ckpt_name',
                        help='saved log and ckpt dir name',
                        default='kaist', type=str)
    parser.add_argument('--dataset', dest='dataset',
                        help='source training dataset',
                        default='cs_combine_fg', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res101 res50',
                        default='vgg16', type=str)

    ##########################################################################
    ##########################################################################
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=[5], type=int)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=1, type=int)
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=10, type=int)
    parser.add_argument('--disp_interval', dest='disp_interval',
                        help='number of iterations to display',
                        default=100, type=int)
    parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                        help='number of iterations to display',
                        default=10000, type=int)

    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models', default="models",
                        type=str)
    parser.add_argument('--load_name', dest='load_name',
                        help='path to load models', default="",
                        type=str)
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=0, type=int)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        default= True)

    parser.add_argument('--detach', dest='detach',
                        help='whether use detach',
                        action='store_false')
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')

    # config optimization
    parser.add_argument('--o', dest='optimizer',
                        help='training optimizer',
                        default="sgd", type=str)
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=0.001, type=float)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)
    parser.add_argument('--s', dest='session',
                        help='training session',
                        default=1, type=int)
    parser.add_argument('--r', dest='resume',
                        help='resume checkpoint or not',
                        default=False, type=bool)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load model',
                        default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load model',
                        default=5, type=int)
    # log and diaplay
    parser.add_argument('--use_tfb', dest='use_tfboard',
                        help='whether use tensorboard',
                        default= False)
    parser.add_argument('--image_dir', dest='image_dir',
                        help='directory to load images for demo',
                        default="images")
    ##########################################################################
    ##########################################################################
    parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      action='store_true')
    parser.add_argument('--webcam_num', dest='webcam_num',
                      help='webcam ID number',
                      default=0, type=int)
    args = parser.parse_args()
    return args

def set_dataset_args(args, test=False):
    if not test:
        if args.dataset == "voc_0712":
            args.imdb_name = "voc_2012_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "cs":
            args.imdb_name = "cs_2007_train"
            args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
        elif args.dataset == "sim":
            args.imdb_name = "sim10k_2012_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "cs_fg":
            args.imdb_name = "cs_fg_2007_train"
            args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
        elif args.dataset == "cs_car":
            args.imdb_name = "cs_car_2007_train"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "kaist_tr":
            args.imdb_name = "kaist_tr_2007_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "kaist_vis":
            args.imdb_name = "kaist_vis_2007_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    else:
        if args.dataset == "clipart":
            args.imdb_name = "clipart_2007_val"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "sim":
            args.imdb_name = "sim10k_2012_test"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "cs_fg":
            args.imdb_name = "cs_fg_2007_val"
            args.set_cfgs = ['ANCHOR_SCALES', '[4,8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "cs_car":
            args.imdb_name = "cs_car_2007_val"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "cs":
            args.imdb_name = "cs_2007_val"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "voc_0712":
            args.imdb_name = "voc_2012_val"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES','20']
        elif args.dataset == "kaist_tr":
            args.imdb_name = "kaist_tr_2007_test"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES','20']
        elif args.dataset == "kaist_vis":
            args.imdb_name = "kaist_vis_2007_test"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES','20']
    args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

    return args
