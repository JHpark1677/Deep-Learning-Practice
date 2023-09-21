import configargparse


def get_args_parser():
    # * config
    parser = configargparse.ArgumentParser(add_help=False)
    parser.add_argument("--path", default="../data", type=str,help='data path')
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--dataset", default="cifar10", type=str)
    parser.add_argument("--resume",'-r',action='store_true', help='resume from checkpoint')
    parser.add_argument('--load_ckp',default='ckpt_vit.pth',type=str, help='checkpoint_name')
    parser.add_argument('--save_ckp',default='ckpt_vit.pth',type=str, help='checkpoint_name')

    return parser