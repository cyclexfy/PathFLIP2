from src.model.pathflip_pl import pathflip_pl 
from torch.utils.data import DataLoader
import pandas as pd
import os
import h5py
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer
import numpy as np
import json
import argparse


def calculate_recall(similarity_matrix, query_labels, gallery_labels, top_k=[1, 5, 10]):
    """Calculate retrieval recall for each requested top-k."""
    num_queries = len(query_labels)
    recall_results = {k: 0.0 for k in top_k}
    
    for i in range(num_queries):
        target_label = query_labels[i]
        sorted_indices = np.argsort(-similarity_matrix[i])
        max_k = max(top_k)
        top_k_indices = sorted_indices[:max_k]
        top_k_labels = gallery_labels[top_k_indices]
        
        for k in top_k:
            if target_label in top_k_labels[:k]:
                recall_results[k] += 1
    
    for k in top_k:
        recall_results[k] /= num_queries
    
    return recall_results


def cross_modal_retrieval(data_loader, model, device=None, dtype=torch.bfloat16):
    all_image_feats = []
    all_text_feats = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="")):
            for key in batch:
                if batch[key] is not None and isinstance(batch[key], torch.Tensor):
                    if 'input_ids' in key or 'attn_mask' in key:
                        batch[key] = batch[key].to(device)
                    else:
                        batch[key] = batch[key].to(device).to(dtype)
            
            image_feat_global, text_feat_global = model.forward_global(batch)
            
            image_feat_global = F.normalize(image_feat_global, dim=-1)
            text_feat_global = F.normalize(text_feat_global, dim=-1)
            
            all_image_feats.append(image_feat_global.float().cpu())
            all_text_feats.append(text_feat_global.float().cpu())
    
    all_image_feats = torch.cat(all_image_feats, dim=0).numpy()
    all_text_feats = torch.cat(all_text_feats, dim=0).numpy()    # [N, D]
    
    print(f"\n: {all_image_feats.shape[0]}")
    print(f": {all_image_feats.shape[1]}")
    
    num_samples = all_image_feats.shape[0]
    labels = np.arange(num_samples)
    print("\n===== Image-to-Text =====")
    i2t_similarity = all_image_feats @ all_text_feats.T  # [N, N]
    i2t_recall = calculate_recall(i2t_similarity, labels, labels, top_k=[1, 5, 10])
    for k, v in i2t_recall.items():
        print(f"Recall@{k}: {v:.4f}")
    
    print("\n===== Text-to-Image =====")
    t2i_similarity = all_text_feats @ all_image_feats.T  # [N, N]
    t2i_recall = calculate_recall(t2i_similarity, labels, labels, top_k=[1, 5, 10])
    for k, v in t2i_recall.items():
        print(f"Recall@{k}: {v:.4f}")
    
    return i2t_recall, t2i_recall


def load_config(args_path):
    import yaml
    from argparse import Namespace
    with open(args_path, 'r') as f:
        args_dict = yaml.load(f, Loader=yaml.FullLoader)
    return Namespace(**args_dict)


def main():
    from src.dataset.dataset_pathflip import dataset_pathflip
    parser = argparse.ArgumentParser()
    parser.add_argument('--args_path', type=str, default='outputs/pathflip/lightning_logs/version_0/hparams.yaml')
    parser.add_argument('--ckpt_path', type=str, default='outputs/pathflip/checkpoint/pytorch_model.bin')
    parser.add_argument('--test_data_path', type=str, default='datasets/SlideBench-Caption-TCGA-plus.json')
    parser.add_argument('--ckpt_id', type=str, default='pathflip_retrieval')
    parser.add_argument('--batch_size', type=int, default=None)
    cli_args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16

    args_path = cli_args.args_path
    ckpt_path = cli_args.ckpt_path
    
    test_data_path = cli_args.test_data_path

    args = load_config(args_path)
    if cli_args.batch_size is not None:
        args.batch_size = cli_args.batch_size
    model_pl = pathflip_pl(args)
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    missing_keys, unexpected_keys = model_pl.load_state_dict(state_dict, strict=False)
    
    print("=" * 80)
    print(f": {len(state_dict) - len(unexpected_keys)} / {len(state_dict)}")
    print(f"Missing keys: {len(missing_keys)}, unexpected keys: {len(unexpected_keys)}")
    print("=" * 80)
    
    model_pl.eval()
    model = model_pl.model.to(device).to(dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    test_dataset = dataset_pathflip(
        data_path=test_data_path,
        patch_sample=True,
        num_patch_samples=args.num_patch_samples,
        num_text_samples=args.num_text_samples,
        tokenizer=tokenizer,
        detail_description=False,
        max_text_length=args.max_text_length,
    )
    
    def collate_fn(instances):
        from src.dataset.dataset_pathflip import default_collate_fn
        return default_collate_fn(instances, tokenizer)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    print(f"\n: {len(test_dataset)}")
    print(f": {args.batch_size}")
    print(f"Number of test batches: {len(test_loader)}")

    i2t_recall, t2i_recall = cross_modal_retrieval(test_loader, model, device, dtype)

    res_path = './eval_results/pathflip_retrieval.csv'
    os.makedirs(os.path.dirname(res_path), exist_ok=True)
    
    if os.path.exists(res_path):
        results = pd.read_csv(res_path)
    else:
        results = pd.DataFrame(columns=["ckpt_id", "i2t_R@1", "i2t_R@5", "i2t_R@10", "t2i_R@1", "t2i_R@5", "t2i_R@10"])

    new_result = {
        "ckpt_id": [cli_args.ckpt_id],
        "i2t_R@1": [i2t_recall[1]],
        "i2t_R@5": [i2t_recall[5]],
        "i2t_R@10": [i2t_recall[10]],
        "t2i_R@1": [t2i_recall[1]],
        "t2i_R@5": [t2i_recall[5]],
        "t2i_R@10": [t2i_recall[10]]
    }
    new_result_df = pd.DataFrame(new_result)
    results = pd.concat([results, new_result_df], ignore_index=True)
    results.to_csv(res_path, index=False)
    print(f"\n: {res_path}")


if __name__ == "__main__":
    main()
