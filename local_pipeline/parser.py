import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='RHWF', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training")
    parser.add_argument('--validation', type=str, nargs='+')
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--gpuid', type=int, nargs='+', default=[0])
    parser.add_argument('--lev0', default=True, action='store_true', help='warp no')
    # parser.add_argument('--lev1', default=False, action='store_true', help='warp once')
    parser.add_argument('--iters_lev0', type=int, default=6)
    parser.add_argument('--iters_lev1', type=int, default=6)
    parser.add_argument('--val_freq', type=int, default=10000, help='validation frequency')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_steps', type=int, default=200000)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--image_size', type=int, nargs='+', default=[512, 512])
    parser.add_argument('--wdecay', type=float, default=0.00001)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.85, help='exponential weighting')
    parser.add_argument('--mixed_precision', default=False, action='store_true', help='use mixed precision')
    parser.add_argument('--model_name', default='', help='specify model name')
    parser.add_argument('--eval_model', type=str, default=None)
    parser.add_argument('--eval_model_ue', type=str, default=None)
    parser.add_argument('--resume', default=False, action='store_true', help='resume_training')
    parser.add_argument('--weight', action='store_true')
    parser.add_argument("--datasets_folder", type=str, default="datasets", help="Path with all datasets")
    parser.add_argument("--dataset_name", type=str, help="Relative path of the dataset")
    parser.add_argument("--prior_location_threshold", type=int, default=-1, help="The threshold of search region from prior knowledge for train and test. If -1, then no prior knowledge")
    parser.add_argument("--val_positive_dist_threshold", type=int, default=50, help="_")
    parser.add_argument("--G_contrast", type=str, default="none", choices=["none", "manual", "autocontrast", "equalize"], help="G_contrast")
    parser.add_argument("--use_ue", action="store_true", help="train uncertainty estimator with GAN")
    parser.add_argument("--G_loss_lambda", type=float, default=1.0, help="G_loss_lambda only for homo")
    parser.add_argument("--D_net", type=str, default="patchGAN", choices=["none", "patchGAN", "patchGAN_deep", "ue_branch"], help="D_net")
    parser.add_argument("--GAN_mode", type=str, default="macegan", choices=["vanilla", "lsgan", "macegan", "vanilla_rej"], help="Choices of GAN loss")
    parser.add_argument("--device", type=str, default="cpu", choices=["cuda", "cpu"])
    parser.add_argument("--train_ue_method", type=str, choices=['train_end_to_end', 'train_only_ue', 'train_only_ue_raw_input', 'finetune'], default='train_end_to_end', help="train uncertainty estimator")
    parser.add_argument("--ue_alpha", type=float, default=-0.1, help="Alpha for ue")
    parser.add_argument("--permute", type=str, default="none", choices=["none", "img", "ue"])
    parser.add_argument("--noise_std", type=float, default=10.0)
    parser.add_argument("--sample_method", type=str, choices=['target', 'raw', 'target_raw'], default='raw', help="sample noise")
    parser.add_argument("--database_size", type=int, default=512, choices=[512, 1024, 1536], help="database_size")
    parser.add_argument("--exclude_val_region", action="store_true", help="exclude_val_region")
    parser.add_argument("--test", action="store_true", help="test mode")
    parser.add_argument("--two_stages", action="store_true", help="crop at level 2 but same scale")
    parser.add_argument("--fine_padding", type=int, default=0, help="expanding region of refinement")
    parser.add_argument("--corr_level", type=int, default=2, choices=[2, 4, 6], help="expanding region of refinement")
    parser.add_argument("--corr_radius", type=int, default=4, choices=[4, 6], help="expanding region of refinement")
    parser.add_argument("--resize_width", type=int, default=256, choices=[256, 512], help="expanding region of refinement")
    parser.add_argument("--fnet_cat", action="store_true", help="fnet_cat")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--vis_all", action="store_true")
    parser.add_argument("--identity", action="store_true")
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--lam_ue", type=float, default=1.0)
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--rej_threshold", type=float, default=128.0)
    parser.add_argument('--eval_model_fine', type=str, default=None, help="restore checkpoint")
    parser.add_argument('--augment_two_stages', type=float, default=0)
    parser.add_argument("--points_detector", type=str, default='orb', choices=['sift', 'orb', 'brisk'],
                        help="key points detector")
    parser.add_argument("--solver", type=str, default='magsac++', choices=['ransac', 'magsac++'],
                        help="solver to be used in cv2.find_homography")
    parser.add_argument("--match_threshold", type=float, default=0.8,
                        help="threshold used to remove mismatch points")
    
    args = parser.parse_args()
    args.save_dir = "local_he"

    if args.use_ue:
        ratio = args.rej_threshold / 512.0
        args.bce_weight = (1- ratio)/ratio
    return args