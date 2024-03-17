"""
With this script you can evaluate checkpoints or test models from two popular
landmark retrieval github repos.
The first is https://github.com/naver/deep-image-retrieval from Naver labs, 
provides ResNet-50 and ResNet-101 trained with AP on Google Landmarks 18 clean.
$ python eval.py --off_the_shelf=naver --l2=none --backbone=resnet101conv5 --aggregation=gem --fc_output_dim=2048

The second is https://github.com/filipradenovic/cnnimageretrieval-pytorch from
Radenovic, provides ResNet-50 and ResNet-101 trained with a triplet loss
on Google Landmarks 18 and sfm120k.
$ python eval.py --off_the_shelf=radenovic_gldv1 --l2=after_pool --backbone=resnet101conv5 --aggregation=gem --fc_output_dim=2048
$ python eval.py --off_the_shelf=radenovic_sfm --l2=after_pool --backbone=resnet101conv5 --aggregation=gem --fc_output_dim=2048

Note that although the architectures are almost the same, Naver's
implementation does not use a l2 normalization before/after the GeM aggregation,
while Radenovic's uses it after (and we use it before, which shows better
results in VG)
"""

import os
import sys
import torch
import parser
import logging
import sklearn
from os.path import join
from datetime import datetime
from torch.utils.model_zoo import load_url
from google_drive_downloader import GoogleDriveDownloader as gdd
import copy
import numpy as np
import test_anyloc
import util
import commons
import datasets_ws
from model import network
from anyloc import utilities

######################################### SETUP #########################################
args = parser.parse_arguments()
start_time = datetime.now()
args.save_dir = join(
    "test",
    args.save_dir,
    f"{args.dataset_name}-{start_time.strftime('%Y-%m-%d_%H-%M-%S')}",
)
commons.setup_logging(args.save_dir)
commons.make_deterministic(args.seed)
if args.use_sparse_database!= -1:
    logging.info(f"Using sparse sampling database. Reset the train and val positive threshold to {args.use_sparse_database}")
    args.val_positive_dist_threshold = args.use_sparse_database
    args.train_positives_dist_threshold = args.use_sparse_database
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.save_dir}")

######################################### DATASETS #########################################
test_ds = datasets_ws.BaseDataset(
    args, args.datasets_folder, args.dataset_name, "test")
train_ds = datasets_ws.BaseDataset(
    args, args.datasets_folder, args.dataset_name, "train")
logging.info(f"Test set: {test_ds}")
logging.info(f"Train set: {train_ds}")
pca = None

######################################### FINETUNE #########################################

desc_layer = 31
desc_facet = "value"
num_c = 32
model = utilities.DinoV2ExtractFeatures("dinov2_vitg14", desc_layer,
        desc_facet, device="cuda")
# model = model.to(args.device)
cache_dir = "./cache"
ext_specifier = f"dinov2_vitg14/"\
    f"l{desc_layer}_{desc_facet}_c{num_c}"
c_centers_file = os.path.join(cache_dir, "vocabulary", 
    ext_specifier, "thermal", "c_centers.pt")
if not os.path.isfile(c_centers_file):
    # Main VLAD object
    vlad = utilities.VLAD(num_c, desc_dim=None,
    cache_dir=os.path.dirname(c_centers_file))
    args.infer_batch_size = 4
    test_anyloc.fit_anyloc(args, train_ds, model, test_method = args.test_method, pca = pca, visualize=args.visual_all, vlad=vlad)

######################################### MODEL #########################################
desc_layer = 31
desc_facet = "value"
num_c = 32
model = utilities.DinoV2ExtractFeatures("dinov2_vitg14", desc_layer,
        desc_facet, device="cuda")
# model = model.to(args.device)
cache_dir = "./cache"
ext_specifier = f"dinov2_vitg14/"\
    f"l{desc_layer}_{desc_facet}_c{num_c}"
c_centers_file = os.path.join(cache_dir, "vocabulary", 
    ext_specifier, "thermal", "c_centers.pt")
assert os.path.isfile(c_centers_file), "Vocabulary not cached!"
c_centers = torch.load(c_centers_file)
assert c_centers.shape[0] == num_c, "Wrong number of clusters!"
# Main VLAD object
vlad = utilities.VLAD(num_c, desc_dim=None, 
cache_dir=os.path.dirname(c_centers_file))
vlad.fit(None)  # Load the vocabulary

######################################### TEST on TEST SET #########################################
args.infer_batch_size = 1
recalls, recalls_str = test_anyloc.test_anyloc(args, test_ds, model, test_method = args.test_method, pca = pca, visualize=args.visual_all, vlad=vlad)
logging.info(f"Recalls on {test_ds}: {recalls_str}")

logging.info(f"Finished in {str(datetime.now() - start_time)[:-7]}")
