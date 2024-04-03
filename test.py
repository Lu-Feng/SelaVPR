
import faiss
import torch
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

import math
import datetime
import os
from os.path import join
from local_matching import local_sim


def test(args, eval_ds, model, test_method="hard_resize", pca=None):
    """Compute features of the given dataset and compute the recalls."""

    assert test_method in ["hard_resize", "single_query", "central_crop", "five_crops",
                            "nearest_crop", "maj_voting"], f"test_method can't be {test_method}"

    if args.efficient_ram_testing:
        return test_efficient_ram_usage(args, eval_ds, model, test_method)    

    model = model.eval()
    with torch.no_grad():
        logging.debug("Extracting database features for evaluation/testing")
        # For database use "hard_resize", although it usually has no effect because database images have same resolution
        eval_ds.test_method = "hard_resize"
        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
        database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                        batch_size=args.infer_batch_size, pin_memory=(args.device=="cuda"))
        
        all_features = np.empty((len(eval_ds), args.features_dim), dtype="float32")
        W, H, C = args.dense_feature_map_size
        all_local_features = np.empty((len(eval_ds), W, H, C), dtype="float32")

        for inputs, indices in tqdm(database_dataloader, ncols=100):
            local_features, features = model(inputs.to(args.device))
            features = features.cpu().numpy()
            local_features = local_features.cpu().numpy()
            if pca != None:
                features = pca.transform(features)
            all_features[indices.numpy(), :] = features
            all_local_features[indices.numpy(), :] = local_features
        
        logging.debug("Extracting queries features for evaluation/testing")
        queries_infer_batch_size = 1 if test_method == "single_query" else args.infer_batch_size
        eval_ds.test_method = test_method
        queries_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num, eval_ds.database_num+eval_ds.queries_num)))
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                        batch_size=queries_infer_batch_size, pin_memory=(args.device=="cuda"))
        for inputs, indices in tqdm(queries_dataloader, ncols=100):
            if test_method == "five_crops" or test_method == "nearest_crop" or test_method == 'maj_voting':
                inputs = torch.cat(tuple(inputs))  # shape = 5*bs x 3 x 480 x 480
            local_features, features = model(inputs.to(args.device))
            if test_method == "five_crops":  # Compute mean along the 5 crops
                features = torch.stack(torch.split(features, 5)).mean(1)
            features = features.cpu().numpy()
            local_features = local_features.cpu().numpy()

            if pca != None:
                features = pca.transform(features)
            
            if test_method == "nearest_crop" or test_method == 'maj_voting':  # store the features of all 5 crops
                start_idx = eval_ds.database_num + (indices[0] - eval_ds.database_num) * 5
                end_idx   = start_idx + indices.shape[0] * 5
                indices = np.arange(start_idx, end_idx)
                all_features[indices, :] = features
            else:
                all_features[indices.numpy(), :] = features
                all_local_features[indices.numpy(), :] = local_features
    
    queries_features = all_features[eval_ds.database_num:]
    database_features = all_features[:eval_ds.database_num]
    queries_local_features = all_local_features[eval_ds.database_num:]
    database_local_features = all_local_features[:eval_ds.database_num]

    faiss_index = faiss.IndexFlatL2(args.features_dim)
    faiss_index.add(database_features)
    del database_features, all_features
    
    logging.debug("Calculating recalls")
    distances, predictions = faiss_index.search(queries_features, args.rerank_num)

    #### For each query, check if the predictions are correct
    positives_per_query = eval_ds.get_positives()
    # args.recall_values by default is [1, 5, 10, 20]
    recalls = np.zeros(len(args.recall_values))
    for query_index, pred in enumerate(predictions):
        for i, n in enumerate(args.recall_values):
            if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                # print(query_index,n)
                break
    # Divide by the number of queries*100, so the recalls are in percentages
    recalls = recalls / eval_ds.queries_num * 100
    recalls_str =", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)])
    logging.info(f"First ranking recalls: {recalls_str}")

    predictions = rerank(predictions,queries_local_features,database_local_features)

    #### For each query, check if the predictions are correct
    positives_per_query = eval_ds.get_positives()
    # args.recall_values by default is [1, 5, 10, 20]
    recalls = np.zeros(len(args.recall_values))
    for query_index, pred in enumerate(predictions):
        for i, n in enumerate(args.recall_values):
            if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                # print(query_index,n)
                break
    # Divide by the number of queries*100, so the recalls are in percentages
    recalls = recalls / eval_ds.queries_num * 100
    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)])
    return recalls, recalls_str

def rerank(predictions,queries_local_features,database_local_features):
    pred2 = []
    print("reranking...")
    for query_index, pred in enumerate(tqdm(predictions)):
        query_local_features = queries_local_features[query_index]
        candidates_local_features = database_local_features[pred]
        query_local_features = torch.Tensor(query_local_features).cuda()
        candidates_local_features = torch.Tensor(candidates_local_features).cuda()
        rerank_index = local_sim(query_local_features, candidates_local_features).cpu().numpy().argsort()[::-1]
        pred2.append(predictions[query_index][rerank_index])     
    return np.array(pred2)

def test_efficient_ram_usage(args, eval_ds, model, test_method="hard_resize"):
    """This function gives the same output as test(), but uses much less RAM.
    It first saves the extracted local features in "./output_local_features/" instead of reading into RAM, 
    and then calls the "rerank_efficient_ram_usage()" function for local matching,
     which loads only the local features currently needed into RAM each time.
    Obviously it is slower than test().
    """

    assert test_method in ["hard_resize", "single_query", "central_crop", "five_crops",
                            "nearest_crop", "maj_voting"], f"test_method can't be {test_method}"
    
    model = model.eval()
    with torch.no_grad():
        logging.debug("Extracting database features for evaluation/testing")
        # For database use "hard_resize", although it usually has no effect because database images have same resolution
        eval_ds.test_method = "hard_resize"
        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
        database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                        batch_size=args.infer_batch_size, pin_memory=(args.device=="cuda"))

        database_features_save_dir = join("output_local_features", args.dataset_name, "database")
        queries_features_save_dir = join("output_local_features", args.dataset_name, "queries")
        os.makedirs(database_features_save_dir, exist_ok=True)  
        os.makedirs(queries_features_save_dir, exist_ok=True)    
        all_features = np.empty((len(eval_ds), args.features_dim), dtype="float32")

        for inputs, indices in tqdm(database_dataloader, ncols=100):
            local_features, features = model(inputs.to(args.device))
            features = features.cpu().numpy()
            local_features = local_features.cpu().numpy()
            for i in range(len(indices)):
                # print(eval_ds.database_paths[i].split("/")[-1].strip(".jpg"))  
                # features_name = eval_ds.database_paths[i].split("/")[-1].replace("jpg","npy")
                features_path = join(database_features_save_dir, str(indices.numpy()[i])+".npy")
                np.save(features_path, local_features[i]) 
            all_features[indices.numpy(), :] = features
        
        logging.debug("Extracting queries features for evaluation/testing")
        queries_infer_batch_size = 1 if test_method == "single_query" else args.infer_batch_size
        eval_ds.test_method = test_method
        queries_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num, eval_ds.database_num+eval_ds.queries_num)))
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                        batch_size=queries_infer_batch_size, pin_memory=(args.device=="cuda"))
        for inputs, indices in tqdm(queries_dataloader, ncols=100):
            local_features, features = model(inputs.to(args.device))
            features = features.cpu().numpy()
            local_features = local_features.cpu().numpy()
            for i in range(len(indices)):
                features_path = join(queries_features_save_dir, str(indices.numpy()[i]-eval_ds.database_num)+".npy")
                np.save(features_path, local_features[i]) 
            all_features[indices.numpy(), :] = features
    
    queries_features = all_features[eval_ds.database_num:]
    database_features = all_features[:eval_ds.database_num]

    faiss_index = faiss.IndexFlatL2(args.features_dim)
    faiss_index.add(database_features)
    del database_features, all_features
    
    logging.debug("Calculating recalls")
    distances, predictions = faiss_index.search(queries_features, args.rerank_num)

    #### For each query, check if the predictions are correct
    positives_per_query = eval_ds.get_positives()
    # args.recall_values by default is [1, 5, 10, 20]
    recalls = np.zeros(len(args.recall_values))
    for query_index, pred in enumerate(predictions):
        for i, n in enumerate(args.recall_values):
            if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                # print(query_index,n)
                break
    # Divide by the number of queries*100, so the recalls are in percentages
    recalls = recalls / eval_ds.queries_num * 100
    recalls_str =", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)])
    logging.info(f"First ranking recalls: {recalls_str}")

    predictions = rerank_efficient_ram_usage(predictions,queries_features_save_dir,database_features_save_dir)

    #### For each query, check if the predictions are correct
    positives_per_query = eval_ds.get_positives()
    # args.recall_values by default is [1, 5, 10, 20]
    recalls = np.zeros(len(args.recall_values))
    for query_index, pred in enumerate(predictions):
        for i, n in enumerate(args.recall_values):
            if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                # print(query_index,n)
                break
    # Divide by the number of queries*100, so the recalls are in percentages
    recalls = recalls / eval_ds.queries_num * 100
    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)])
    return recalls, recalls_str

def rerank_efficient_ram_usage(predictions,queries_features_save_dir,database_features_save_dir):
    pred2 = []
    print("reranking...")
    for query_index, pred in enumerate(tqdm(predictions)):
        query_features_path = join(queries_features_save_dir, str(query_index)+".npy")
        query_local_features = np.load(query_features_path)
        query_local_features = torch.Tensor(query_local_features).cuda()
        
        W, H, C = query_local_features.shape
        candidates_local_features = np.empty((len(pred), W, H, C), dtype="float32")
        for i in range(len(pred)):
            predi_features_path = join(database_features_save_dir, str(pred[i])+".npy")
            candidates_local_features[i] = np.load(predi_features_path)
        candidates_local_features = torch.Tensor(candidates_local_features).cuda()
        rerank_index = local_sim(query_local_features,candidates_local_features).cpu().numpy().argsort()[::-1]
        pred2.append(predictions[query_index][rerank_index])     
    return np.array(pred2)
