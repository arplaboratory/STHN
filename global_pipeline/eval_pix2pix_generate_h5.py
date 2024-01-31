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

import global_pipeline.test as test
import util
import commons
import datasets_ws
from model import network

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
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.save_dir}")

######################################### MODEL #########################################
model = network.pix2pix(args, 3, 1)

if args.resume is not None:
    logging.info(f"Resuming model from {args.resume}")
    model = util.resume_model_pix2pix(args, model)
# Enable DataParallel after loading checkpoint, otherwise doing it before
# would append "module." in front of the keys of the state dict triggering errors

model.setup()

######################################### DATASETS #########################################
train_ds = datasets_ws.TranslationDataset(
    args, args.datasets_folder, args.dataset_name, "train", loading_queries=False)
logging.info(f"Train set: {train_ds}")

######################################### TEST on TEST SET #########################################
test.test_translation_pix2pix_generate_h5(args, train_ds, model)

logging.info(f"Finished in {str(datetime.now() - start_time)[:-7]}")
