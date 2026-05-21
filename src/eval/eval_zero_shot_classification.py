from src.model.pathflip_pl import pathflip_pl 
from src.dataset.dataset_pathflip import dataset_pathflip
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import h5py
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, f1_score
import argparse


class GenericDataset(Dataset):
    def __init__(self, csv_file, label_mapping, feature_col='conch_feature', label_col='label'):
        self.data = pd.read_csv(csv_file)
        self.feature_col = feature_col
        self.label_col = label_col
        self.label_mapping = label_mapping
        
        valid_mask = self.data[feature_col].apply(os.path.exists)
        self.data = self.data[valid_mask].reset_index(drop=True)
        
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        data = self.data.iloc[idx]
        image_feat_path = data[self.feature_col]
        if image_feat_path.endswith('.h5'):
            image_feat = h5py.File(image_feat_path, 'r')['features'][:]
            image_coords = h5py.File(image_feat_path, 'r')['coords'][:]
            image_feat = torch.from_numpy(image_feat).to(torch.float32)
            image_coords = torch.from_numpy(image_coords).to(torch.long)
        elif image_feat_path.endswith('.pt'):
            image_feat = torch.load(image_feat_path)
            image_coords = None
        else:
            raise ValueError(f"Invalid image feature path {image_feat_path}")
        label = self.label_mapping[data[self.label_col]]
        return image_feat, image_coords, label

def slide_level_zero_shot_classification(data_loader, model, tokenizer, prompts, device=None, dtype=torch.bfloat16):

    with torch.no_grad():
        input_ids = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
        batch = {
            'input_ids_global': input_ids['input_ids'],
            'attn_mask_global': input_ids['attention_mask'],
        }
        text_feats = model.forward_text_global(batch) # [num_classes, D]
        text_feats = F.normalize(text_feats, dim=-1) # [num_classes, D]

    all_predictions = []
    all_labels = []
    all_probabilities = []
    total_samples = 0
    total_correct = 0
    num_classes = len(prompts)

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(data_loader)):
            image_feat, image_coords, label = data

            image_feats = image_feat.to(device).to(dtype) # [B, L, D]
            image_coords = image_coords.to(device).to(dtype) # [B, L, 2]
            labels = label # [B,]

            batch = {
                'image': image_feats,
                'image_coords': image_coords,
            }
            image_feats = model.forward_image_global(batch) # [B, D]
            image_feats = F.normalize(image_feats, dim=-1) # [B, D]
            
            logits = image_feats @ text_feats.T  # [B, num_classes]
            probs = logits.float().softmax(dim=-1)

            predictions = torch.argmax(probs, dim=-1)

            correct = (predictions == labels).sum().item()
            total_correct += correct
            total_samples += labels.size(0)
            
            all_predictions.append(predictions.numpy())
            all_labels.append(labels.numpy())
            all_probabilities.append(probs.detach().cpu().float().numpy())
        
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_probabilities = np.concatenate(all_probabilities, axis=0)

    assert np.all(all_labels < num_classes), (
        f"Label value out of range: max label {all_labels.max()} >= num_classes {num_classes}"
    )

    if num_classes == 2:
        auc_roc = roc_auc_score(all_labels, all_probabilities[:, 1])
        f1 = f1_score(all_labels, all_predictions)
    else:
        auc_roc = roc_auc_score(all_labels, all_probabilities, multi_class='ovr')
        f1 = f1_score(all_labels, all_predictions, average='weighted')
    print(f'Num classes: {num_classes}')
    print(f'Total samples: {total_samples}')
    print(f'Overall Accuracy: {total_correct}/{total_samples} = {overall_accuracy:.4f}')
    print(f'AUC-ROC Score: {auc_roc:.4f}')
    print(f'F1 Score: {f1:.4f}')
    
    return overall_accuracy, auc_roc, f1, all_predictions, all_labels, all_probabilities
    
def load_config(args_path):
    import yaml
    from argparse import Namespace
    with open(args_path, 'r') as f:
        args_dict = yaml.load(f, Loader=yaml.FullLoader)
    return Namespace(**args_dict)

def main():
    parser = argparse.ArgumentParser(description='PathFLIP zero-shot classification evaluation')
    parser.add_argument('--args_path', type=str, 
                        default="outputs/pathflip/lightning_logs/version_0/hparams.yaml",
                        help='Path to model hparams.yaml config file')
    parser.add_argument('--ckpt_path', type=str,
                        default="outputs/pathflip/checkpoint/pytorch_model.bin",
                        help='Path to model pytorch_model.bin checkpoint')
    parser.add_argument('--ckpt_id', type=str, default='pathflip',
                        help='Checkpoint ID for result identification')
    parser.add_argument('--datasets', nargs='+', choices=['Camelyon16', 'TCGA_NSCLC', 'TCGA_RCC', 'CPTAC_NSCLC'],
                        default=['Camelyon16', 'TCGA_NSCLC', 'TCGA_RCC', 'CPTAC_NSCLC'],
                        help='List of datasets to evaluate (default: all datasets)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Evaluation batch size (default: 1)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16

    model_args = load_config(args.args_path)
    model_pl = pathflip_pl(model_args)

    state_dict = torch.load(args.ckpt_path, map_location="cpu", weights_only=True)
    missing_keys, unexpected_keys = model_pl.load_state_dict(state_dict, strict=False)
    
    print("=" * 80)
    print(f": {len(state_dict) - len(unexpected_keys)} / {len(state_dict)}")
    print(f"Missing keys: {len(missing_keys)}")
    if missing_keys:
        print(":")
        for key in missing_keys:
            print(f"  - {key}")
    print(f": {len(unexpected_keys)}")
    if unexpected_keys:
        print("Unexpected key list:")
        for key in unexpected_keys:
            print(f"  - {key}")
    print("=" * 80)
    model_pl.eval()

    model = model_pl.model.to(device).to(dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer)

    dataset_configs = {
        "Camelyon16": {
            "csv_path": "datasets/Camelyon16_test.csv",
            "label_mapping": {
                'Normal': 0,
                'Tumor': 1,
            },
            "feature_col": "conch_feature",
            "label_col": "label",
            "prompts" : [
                "Normal",
                "metastatic breast cancer"
            ],
        },
        "TCGA_NSCLC": {
            "csv_path": "datasets/TCGA_NSCLC_test.csv",
            "label_mapping": {
                'TCGA-LUAD': 0,
                'TCGA-LUSC': 1,
            },
            "feature_col": "conch_feature",
            "prompts": [
                "H&E stained histopathology image of lung adenocarcinoma",
                "H&E stained histopathology image of lung squamous cell carcinoma"
            ]
        },
        "TCGA_RCC": {
            "csv_path": "datasets/TCGA_RCC_test.csv",
            "label_mapping": {
                'TCGA-KICH': 0,
                'TCGA-KIRC': 1,
                'TCGA-KIRP': 2,
            },
            "feature_col": "conch_feature",
            "prompts": [
                "H&E stained histopathology image of kidney chromophobe carcinoma",
                "H&E stained histopathology image of kidney renal clear cell carcinoma",
                "H&E stained histopathology image of kidney renal papillary cell carcinoma"
            ]
        },
        "CPTAC_NSCLC": {
            "csv_path": "datasets/CPTAC_NSCLC_test.csv",
            "label_mapping": {
                'CPTAC-LUAD': 0,
                'CPTAC-LSCC': 1,
            },
            "feature_col": "conch_feature",
            "label_col": "label",
            "prompts": [
                "H&E stained histopathology image of lung adenocarcinoma",
                "H&E stained histopathology image of lung squamous cell carcinoma"
            ]
        }
    }

    datasets_to_eval = args.datasets
    ckpt_id = args.ckpt_id

    res_path = f'./eval_results/pathflip_zero_shot_classification.csv'
    if os.path.exists(res_path):
        results = pd.read_csv(res_path)
    else:
        results = pd.DataFrame(columns=["dataset", "accuracy", "auc_roc", "f1", "ckpt_id"])

    for dataset_name in datasets_to_eval:
        print(f"\n{'='*60}")
        print(f"Evaluating dataset: {dataset_name}")
        print(f"{'='*60}")

        config = dataset_configs[dataset_name]
        dataset = GenericDataset(
            config["csv_path"], 
            config["label_mapping"],
            feature_col=config.get("feature_col", "conch_feature"),
            label_col=config.get("label_col", "label")
        )
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        
        overall_accuracy, auc_roc, f1, all_predictions, all_labels, all_probabilities = slide_level_zero_shot_classification(
            data_loader, model, tokenizer, config["prompts"], device, dtype
        )

        new_result = {
            "dataset": [dataset_name],
            "accuracy": [overall_accuracy],
            "auc_roc": [auc_roc],
            "f1": [f1],
            "ckpt_id": [ckpt_id]
        }
        new_result_df = pd.DataFrame(new_result)
        results = pd.concat([results, new_result_df], ignore_index=True)
        results.to_csv(res_path, index=False)

if __name__ == "__main__":
    main()

