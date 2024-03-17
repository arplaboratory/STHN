import faiss
import torch
import logging
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from plotting import process_results_simulation
from h5_transformer import calc_overlap
from model.functional import calculate_psnr
import yaml
import os
from PIL import Image
import shutil
import datasets_ws
import h5py
import einops as ein

def test_anyloc(args, eval_ds, model, test_method="hard_resize", pca=None, visualize=False, vlad=None):
    """Compute features of the given dataset and compute the recalls."""

    assert test_method in [
        "hard_resize",
        "single_query",
        "central_crop",
        "five_crops",
        "nearest_crop",
        "maj_voting",
    ], f"test_method can't be {test_method}"

    if args.efficient_ram_testing:
        return test_efficient_ram_usage(args, eval_ds, model, test_method)

    model = model
    with torch.no_grad():
        logging.debug("Extracting database features for evaluation/testing")
        # For database use "hard_resize", although it usually has no effect because database images have same resolution
        eval_ds.test_method = "hard_resize"
        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
        database_dataloader = DataLoader(
            dataset=database_subset_ds,
            num_workers=args.num_workers,
            batch_size=args.infer_batch_size,
            pin_memory=(args.device == "cuda"),
        )

        if test_method == "nearest_crop" or test_method == "maj_voting":
            all_features = np.empty(
                (5 * eval_ds.queries_num + eval_ds.database_num, args.features_dim),
                dtype="float32",
            )
        else:
            all_features = np.empty(
                (len(eval_ds), 49152), dtype="float32")

        for inputs, indices in tqdm(database_dataloader, ncols=100):
            ret = model(inputs.to(args.device))
            features = vlad.generate(ret.cpu().squeeze())[np.newaxis, ...]
            features = features.cpu().numpy()
            if pca != None:
                features = pca.transform(features)
            all_features[indices.numpy(), :] = features

        logging.debug("Extracting queries features for evaluation/testing")
        queries_infer_batch_size = (
            1 if test_method == "single_query" else args.infer_batch_size
        )
        eval_ds.test_method = test_method
        queries_subset_ds = Subset(
            eval_ds,
            list(
                range(eval_ds.database_num,
                      eval_ds.database_num + eval_ds.queries_num)
            ),
        )
        queries_dataloader = DataLoader(
            dataset=queries_subset_ds,
            num_workers=args.num_workers,
            batch_size=queries_infer_batch_size,
            pin_memory=(args.device == "cuda"),
        )
        for inputs, indices in tqdm(queries_dataloader, ncols=100):
            if (
                test_method == "five_crops"
                or test_method == "nearest_crop"
                or test_method == "maj_voting"
            ):
                # shape = 5*bs x 3 x 480 x 480
                inputs = torch.cat(tuple(inputs))
            ret = model(inputs.to(args.device))
            features = vlad.generate(ret.cpu().squeeze())[np.newaxis, ...]
            if test_method == "five_crops":  # Compute mean along the 5 crops
                features = torch.stack(torch.split(features, 5)).mean(1)
            features = features.cpu().numpy()
            if pca != None:
                features = pca.transform(features)

            if (
                test_method == "nearest_crop" or test_method == "maj_voting"
            ):  # store the features of all 5 crops
                start_idx = (
                    eval_ds.database_num +
                    (indices[0] - eval_ds.database_num) * 5
                )
                end_idx = start_idx + indices.shape[0] * 5
                indices = np.arange(start_idx, end_idx)
                all_features[indices, :] = features
            else:
                all_features[indices.numpy(), :] = features

    queries_features = all_features[eval_ds.database_num:]
    database_features = all_features[: eval_ds.database_num]
    logging.info(f"Final feature dim: {queries_features.shape[1]}")
        
    del all_features

    logging.debug("Calculating recalls")
    if args.prior_location_threshold == -1:
        faiss_index = faiss.IndexFlatL2(49152)
        faiss_index.add(database_features)
        distances, predictions = faiss_index.search(
            queries_features, max(args.recall_values)
        )
        del database_features
    else:
        distances, predictions = [[] for i in range(len(queries_features))], [[] for i in range(len(queries_features))]
        hard_negatives_per_query = eval_ds.get_hard_negatives()
        for query_index in tqdm(range(len(predictions))):
            faiss_index = faiss.IndexFlatL2(49152)
            faiss_index.add(database_features[hard_negatives_per_query[query_index]])
            distances_single, local_predictions_single = faiss_index.search(
                np.expand_dims(queries_features[query_index], axis=0), max(args.recall_values)
                )
            # logging.debug(f"distances_single:{distances_single}")
            # logging.debug(f"predictions_single:{predictions_single}")
            distances[query_index] = distances_single
            predictions_single = hard_negatives_per_query[query_index][local_predictions_single]
            predictions[query_index] = predictions_single
        distances = np.concatenate(distances, axis=0)
        predictions = np.concatenate(predictions, axis=0)
        del database_features
    if test_method == "nearest_crop":
        distances = np.reshape(distances, (eval_ds.queries_num, 20 * 5))
        predictions = np.reshape(predictions, (eval_ds.queries_num, 20 * 5))
        for q in range(eval_ds.queries_num):
            # sort predictions by distance
            sort_idx = np.argsort(distances[q])
            predictions[q] = predictions[q, sort_idx]
            # remove duplicated predictions, i.e. keep only the closest ones
            _, unique_idx = np.unique(predictions[q], return_index=True)
            # unique_idx is sorted based on the unique values, sort it again
            predictions[q, :20] = predictions[q, np.sort(unique_idx)][:20]
        predictions = predictions[
            :, :20
        ]  # keep only the closer 20 predictions for each query
    elif test_method == "maj_voting":
        distances = np.reshape(distances, (eval_ds.queries_num, 5, 20))
        predictions = np.reshape(predictions, (eval_ds.queries_num, 5, 20))
        for q in range(eval_ds.queries_num):
            # votings, modify distances in-place
            top_n_voting("top1", predictions[q],
                         distances[q], args.majority_weight)
            top_n_voting("top5", predictions[q],
                         distances[q], args.majority_weight)
            top_n_voting("top10", predictions[q],
                         distances[q], args.majority_weight)

            # flatten dist and preds from 5, 20 -> 20*5
            # and then proceed as usual to keep only first 20
            dists = distances[q].flatten()
            preds = predictions[q].flatten()

            # sort predictions by distance
            sort_idx = np.argsort(dists)
            preds = preds[sort_idx]
            # remove duplicated predictions, i.e. keep only the closest ones
            _, unique_idx = np.unique(preds, return_index=True)
            # unique_idx is sorted based on the unique values, sort it again
            # here the row corresponding to the first crop is used as a
            # 'buffer' for each query, and in the end the dimension
            # relative to crops is eliminated
            predictions[q, 0, :20] = preds[np.sort(unique_idx)][:20]
        predictions = predictions[
            :, 0, :20
        ]  # keep only the closer 20 predictions for each query

    # For each query, check if the predictions are correct
    positives_per_query = eval_ds.get_positives()
    # args.recall_values by default is [1, 5, 10, 20]
    recalls = np.zeros(len(args.recall_values))
    for query_index, pred in enumerate(predictions):
        for i, n in enumerate(args.recall_values):
            if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                break
    # Divide by the number of queries*100, so the recalls are in percentages
    recalls = recalls / eval_ds.queries_num * 100
    recalls_str = ", ".join(
        [f"R@{val}: {rec:.1f}" for val,
            rec in zip(args.recall_values, recalls)]
    )

    if args.use_best_n > 0:
        if visualize:
            if os.path.isdir("visual_loc"):
                shutil.rmtree("visual_loc")
            os.mkdir("visual_loc")
            save_dir = "visual_loc"
            # init dataset
            eval_ds.__getitem__(0)
        samples_to_be_used = args.use_best_n
        error_m = []
        position_m = []
        for query_index in tqdm(range(len(predictions))):
            distance = distances[query_index]
            prediction = predictions[query_index]
            sort_idx = np.argsort(distance)
            if args.use_best_n == 1:
                best_position = eval_ds.database_utms[prediction[sort_idx[0]]]
            else:
                if distance[sort_idx[0]] == 0:
                    best_position = eval_ds.database_utms[prediction[sort_idx[0]]]
                else:
                    mean = distance[sort_idx[0]]
                    sigma = distance[sort_idx[0]] / distance[sort_idx[-1]]
                    X = np.array(distance[sort_idx[:samples_to_be_used]]).reshape((-1,))
                    weights = np.exp(-np.square(X - mean) / (2 * sigma ** 2))  # gauss
                    weights = weights / np.sum(weights)

                    x = y = 0
                    for p, w in zip(eval_ds.database_utms[prediction[sort_idx[:samples_to_be_used]]], weights.tolist()):
                        y += p[0] * w
                        x += p[1] * w
                    best_position = (y, x)
            actual_position = eval_ds.queries_utms[query_index]
            error = np.linalg.norm((actual_position[0]-best_position[0], actual_position[1]-best_position[1]))
            if error >= args.val_positive_dist_threshold and visualize: # Wrong results
                database_index = prediction[sort_idx[0]]
                database_img = eval_ds._find_img_in_h5(database_index, "database")
                if args.G_contrast!="none":
                    if args.G_contrast == "manual":
                        query_img = transforms.functional.adjust_contrast(eval_ds._find_img_in_h5(query_index, "queries"), contrast_factor=3)
                    elif args.G_contrast == "autocontrast":
                        query_img = transforms.functional.autocontrast(eval_ds._find_img_in_h5(query_index, "queries"))
                    elif args.G_contrast == "equalize":
                        query_img = transforms.functional.equalize(eval_ds._find_img_in_h5(query_index, "queries"))
                    else:
                        raise NotImplementedError()
                else:
                    query_img = eval_ds._find_img_in_h5(query_index, "queries")
                result = Image.new(database_img.mode, (524, 524), (255, 0, 0))
                result.paste(database_img, (6, 6))
                database_img = result
                database_img.save(f"{save_dir}/{query_index}_{best_position}_wrong_d.png")
                query_img.save(f"{save_dir}/{query_index}_{actual_position}_wrong_q.png")
            elif error <= args.train_positives_dist_threshold and visualize: # Wrong results
                database_index = prediction[sort_idx[0]]
                database_img = eval_ds._find_img_in_h5(database_index, "database")
                if args.G_contrast!="none":
                    if args.G_contrast == "manual":
                        query_img = transforms.functional.adjust_contrast(eval_ds._find_img_in_h5(query_index, "queries"), contrast_factor=3)
                    elif args.G_contrast == "autocontrast":
                        query_img = transforms.functional.autocontrast(eval_ds._find_img_in_h5(query_index, "queries"))
                    elif args.G_contrast == "equalize":
                        query_img = transforms.functional.equalize(eval_ds._find_img_in_h5(query_index, "queries"))
                    else:
                        raise NotImplementedError()
                else:
                    query_img = eval_ds._find_img_in_h5(query_index, "queries")
                result = Image.new(database_img.mode, (524, 524), (0, 255, 0))
                result.paste(database_img, (6, 6))
                database_img = result
                database_img.save(f"{save_dir}/{query_index}_correct_d.png")
                query_img.save(f"{save_dir}/{query_index}_correct_q.png")
            elif visualize: # Ambiguous results
                database_index = prediction[sort_idx[0]]
                database_img = eval_ds._find_img_in_h5(database_index, "database")
                if args.G_contrast:
                    query_img = transforms.functional.adjust_contrast(eval_ds._find_img_in_h5(query_index, "queries"), contrast_factor=3)
                else:
                    query_img = eval_ds._find_img_in_h5(query_index, "queries")
                result = Image.new(database_img.mode, (524, 524), (128, 128, 128))
                result.paste(database_img, (6, 6))
                database_img = result
                database_img.save(f"{save_dir}/{query_index}_d.png")
                query_img.save(f"{save_dir}/{query_index}_q.png")
            
            error_m.append(error)
            position_m.append(actual_position)
        process_results_simulation(error_m, args.save_dir)
            
    return recalls, recalls_str

def fit_anyloc(args, eval_ds, model, test_method="hard_resize", pca=None, visualize=False, vlad=None):
    """Compute features of the given dataset and compute the recalls."""

    assert test_method in [
        "hard_resize",
        "single_query",
        "central_crop",
        "five_crops",
        "nearest_crop",
        "maj_voting",
    ], f"test_method can't be {test_method}"

    if args.efficient_ram_testing:
        return test_efficient_ram_usage(args, eval_ds, model, test_method)

    model = model
    with torch.no_grad():
        logging.debug("Extracting database features for evaluation/testing")
        # For database use "hard_resize", although it usually has no effect because database images have same resolution
        eval_ds.test_method = "hard_resize"
        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
        database_dataloader = DataLoader(
            dataset=database_subset_ds,
            num_workers=args.num_workers,
            batch_size=args.infer_batch_size,
            pin_memory=(args.device == "cuda"),
        )

        if test_method == "nearest_crop" or test_method == "maj_voting":
            all_features = np.empty(
                (5 * eval_ds.queries_num + eval_ds.database_num, args.features_dim),
                dtype="float32",
            )
        else:
            all_features = np.empty(
                (len(eval_ds), 1296*1536), dtype="float32")

        for inputs, indices in tqdm(database_dataloader, ncols=100):
            ret = model(inputs.to(args.device))
            features = ret.view(-1, 1296*1536)
            features = features.cpu().numpy()
            if pca != None:
                features = pca.transform(features)
            all_features[indices.numpy(), :] = features

        logging.debug("Extracting queries features for evaluation/testing")
        queries_infer_batch_size = (
            1 if test_method == "single_query" else args.infer_batch_size
        )
        eval_ds.test_method = test_method
        queries_subset_ds = Subset(
            eval_ds,
            list(
                range(eval_ds.database_num,
                      eval_ds.database_num + eval_ds.queries_num)
            ),
        )
        queries_dataloader = DataLoader(
            dataset=queries_subset_ds,
            num_workers=args.num_workers,
            batch_size=queries_infer_batch_size,
            pin_memory=(args.device == "cuda"),
        )
        for inputs, indices in tqdm(queries_dataloader, ncols=100):
            if (
                test_method == "five_crops"
                or test_method == "nearest_crop"
                or test_method == "maj_voting"
            ):
                # shape = 5*bs x 3 x 480 x 480
                inputs = torch.cat(tuple(inputs))
            ret = model(inputs.to(args.device))
            features = ret.view(-1, 1296*1536)
            if test_method == "five_crops":  # Compute mean along the 5 crops
                features = torch.stack(torch.split(features, 5)).mean(1)
            features = features.cpu().numpy()
            if pca != None:
                features = pca.transform(features)

            if (
                test_method == "nearest_crop" or test_method == "maj_voting"
            ):  # store the features of all 5 crops
                start_idx = (
                    eval_ds.database_num +
                    (indices[0] - eval_ds.database_num) * 5
                )
                end_idx = start_idx + indices.shape[0] * 5
                indices = np.arange(start_idx, end_idx)
                all_features[indices, :] = features
            else:
                all_features[indices.numpy(), :] = features
    all_features=np.reshape(all_features, (len(eval_ds), 1296, 1536))
    vlad.fit(ein.rearrange(all_features, "n k d -> (n k) d"))
