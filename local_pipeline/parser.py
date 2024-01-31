import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='RHWF', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training")
    parser.add_argument('--validation', type=str, nargs='+')
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    
    parser.add_argument('--gpuid', type=int, nargs='+', default = [0])

    parser.add_argument('--lev0', default=True, action='store_true', help='warp no')
    parser.add_argument('--lev1', default=False, action='store_true', help='warp once')
    parser.add_argument('--iters_lev0', type=int, default=6)
    parser.add_argument('--iters_lev1', type=int, default=6)
    
    parser.add_argument('--val_freq', type=int, default=10000, help='validation frequency')
    parser.add_argument('--print_freq', type=int, default=100, help='printing frequency')

    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--num_steps', type=int, default=200000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--image_size', type=int, nargs='+', default=[512, 512])
    parser.add_argument('--wdecay', type=float, default=0.00001)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.85, help='exponential weighting')
    parser.add_argument('--mixed_precision', default=False, action='store_true', help='use mixed precision')
    
    parser.add_argument('--model_name', default='', help='specify model name')
    parser.add_argument('--resume', default=False, action='store_true', help='resume_training')
    
    parser.add_argument('--weight', action='store_true')
    parser.add_argument(
        "--datasets_folder", type=str, default="datasets", help="Path with all datasets"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="satellite_0_satellite_0_dense",
        help="Relative path of the dataset",
    )
    parser.add_argument(
        "--prior_location_threshold",
        type=int,
        default=-1,
        help="The threshold of search region from prior knowledge for train and test. If -1, then no prior knowledge"
    )
    parser.add_argument("--val_positive_dist_threshold", type=int, default=50, help="_")
    parser.add_argument(
        "--G_contrast",
        type=str,
        default="none",
        choices=["none", "manual", "autocontrast", "equalize"],
        help="G_contrast"
    )
    parser.add_argument(
        "--use_ue",
        action="store_true",
        help="train uncertainty estimator with GAN"
    )
    parser.add_argument(
        "--G_loss_lambda",
        type=float,
        default=1.0,
        help="G_loss_lambda only for homo"
    )
    parser.add_argument(
        "--D_net",
        type=str,
        default="patchGAN_deep",
        choices=["none", "patchGAN", "patchGAN_deep"],
        help="D_net"
    )
    parser.add_argument(
        "--GAN_mode",
        type=str,
        default="macegan",
        choices=["vanilla", "lsgan", "macegan", "macegancross"],
        help="Choices of GAN loss"
    )
    parser.add_argument("--device", type=str,
        default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--train_ue_method",
        type=str,
        choices=['train_end_to_end', 'train_only_ue', 'train_only_ue_raw_input'],
        default='train_end_to_end',
        help="train uncertainty estimator"
    )
    parser.add_argument(
        "--ue_alpha",
        type=float,
        default=-0.1,
        help="Alpha for ue"
    )
    parser.add_argument(
        "--permute",
        action="store_true",
        help="Permute input images"
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        default=10
    )
    parser.add_argument(
        "--sample_method",
        type=str,
        choices=['target', 'raw', 'target_raw'],
        default='target_raw',
        help="sample noise"
    )
    args = parser.parse_args()
    args.resize_small = True
    
    return args