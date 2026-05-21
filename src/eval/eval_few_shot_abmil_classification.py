from src.model.pathflip_pl import pathflip_pl 
from src.model.modules import Attn_Net_Gated
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
import torch.nn as nn


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


class ABMILClassifier(nn.Module):
    def __init__(self, feature_dim, hidden_dim=256, n_classes=1, dropout=0.25):
        super(ABMILClassifier, self).__init__()
        self.attention_net = Attn_Net_Gated(L=feature_dim, D=hidden_dim, dropout=dropout, n_classes=1)
        self.classifier = nn.Linear(feature_dim, n_classes)
        self.n_classes = n_classes

    def forward(self, features):
        bag_features, attn_weights = self.attention_net(features)
        logits = self.classifier(bag_features)
        return logits, attn_weights

    def predict_proba(self, features):
        logits, _ = self(features)
        if self.n_classes == 1:
            return torch.sigmoid(logits)
        return F.softmax(logits, dim=-1)


def extract_local_features(dataset, model, device, dtype, num_samples=None):
    """Extract patch-level features and labels from a dataset."""
    all_features = []
    all_labels = []
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    model.eval()
    with torch.no_grad():
        for batch_idx, (image_feat, image_coords, label) in enumerate(tqdm(data_loader, desc="Extracting local features")):
            if num_samples is not None and batch_idx >= num_samples:
                break
            image_feats = image_feat.to(device).to(dtype)
            image_coords_batch = image_coords.to(device) if image_coords is not None else None
            batch = {'image': image_feats, 'image_coords': image_coords_batch}
            local_features = model.forward_image_local(batch)
            all_features.append(local_features.detach().cpu().float().numpy().squeeze(0))  # [L, D]
            all_labels.append(label.item())
    return all_features, np.array(all_labels)


def slide_level_few_shot_abmil_probing(train_features_list, train_labels, test_features_list, test_labels_np, num_classes, hidden_dim=256, max_iter=300, lr=5e-4, accum_steps=4):
    """Run few-shot ABMIL classification and keep the best epoch result."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_dim = train_features_list[0].shape[-1]
    
    mil_classifier = ABMILClassifier(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        n_classes=num_classes if num_classes > 1 else 1,
        dropout=0.25
    ).to(device)
    
    optimizer = torch.optim.Adam(mil_classifier.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss() if num_classes > 1 else nn.BCEWithLogitsLoss()
    
    train_tensors = [(torch.from_numpy(f).float(), torch.tensor(l).long()) for f, l in zip(train_features_list, train_labels)]
    
    best_metric = 0.0
    best_results = None
    
    mil_classifier.train()
    for epoch in range(max_iter):
        indices = np.random.permutation(len(train_tensors))
        optimizer.zero_grad()
        
        for step_idx, idx in enumerate(indices):
            features, label = train_tensors[idx]
            features = features.unsqueeze(0).to(device)
            label = label.unsqueeze(0).to(device)
            
            logits, _ = mil_classifier(features)
            
            if num_classes == 1:
                loss = criterion(logits.squeeze(-1), label.float())
            else:
                loss = criterion(logits, label)
            
            loss = loss / accum_steps
            loss.backward()
            
            if (step_idx + 1) % accum_steps == 0 or step_idx == len(indices) - 1:
                optimizer.step()
                optimizer.zero_grad()
        
        all_predictions = []
        all_probs = []
        
        with torch.no_grad():
            for features in test_features_list:
                features_t = torch.from_numpy(features).float().unsqueeze(0).to(device)
                logits, _ = mil_classifier(features_t)
                
                if num_classes == 1:
                    prob = torch.sigmoid(logits).cpu().numpy().squeeze(-1).squeeze(0)
                    pred = int(prob > 0.5)
                else:
                    prob = F.softmax(logits, dim=-1).cpu().numpy().squeeze(0)
                    pred = int(np.argmax(prob))
                
                all_predictions.append(pred)
                all_probs.append(prob)
        
        predictions = np.array(all_predictions)
        probs = np.array(all_probs)
        
        epoch_acc = np.mean(predictions == test_labels_np)
        if num_classes == 2:
            epoch_auc = roc_auc_score(test_labels_np, probs[:, 1])
        else:
            epoch_auc = roc_auc_score(test_labels_np, probs, multi_class='ovr')
        
        
        if current_metric > best_metric:
            best_metric = current_metric
            best_results = {
                'accuracy': epoch_acc,
                'auc_roc': epoch_auc,
                'predictions': predictions.copy(),
                'probs': probs.copy(),
                'best_epoch': epoch + 1
            }
        
        mil_classifier.train()
    
    auc_roc = best_results['auc_roc']
    predictions = best_results['predictions']
    probs = best_results['probs']
    
    if num_classes == 2:
        f1 = f1_score(test_labels_np, predictions)
    else:
        f1 = f1_score(test_labels_np, predictions, average='weighted')
    
    print(f'Num classes: {num_classes}')
    print(f'Train samples per class: {[len(np.where(train_labels == c)[0]) for c in range(num_classes)]}')
    print(f'Total test samples: {len(test_labels_np)}')
    print(f'Best epoch: {best_results["best_epoch"]}/{max_iter}')
    print(f'Best Accuracy: {overall_accuracy:.4f} (AUC: {auc_roc:.4f})')
    print(f'Final F1 Score: {f1:.4f}')
    
    return overall_accuracy, auc_roc, f1, predictions, test_labels_np, probs


def load_config(args_path):
    import yaml
    from argparse import Namespace
    with open(args_path, 'r') as f:
        args_dict = yaml.load(f, Loader=yaml.FullLoader)
    return Namespace(**args_dict)

def main():
    parser = argparse.ArgumentParser(description='PathFLIP few-shot linear probing classification evaluation')
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
    parser.add_argument('--k_shot', type=int, default=5,
                        help='Number of samples per class for few-shot training (default: 5)')
    parser.add_argument('--num_runs', type=int, default=3,
                        help='Number of independent runs for averaging results (default: 3)')
    parser.add_argument('--max_iter', type=int, default=500,
                        help='Maximum iterations for ABMIL training (default: 500)')
    parser.add_argument('--abmil_hidden_dim', type=int, default=256,
                        help='Hidden dimension for ABMIL attention network (default: 256)')
    parser.add_argument('--abmil_lr', type=float, default=5e-4,
                        help='Learning rate for ABMIL training (default: 5e-4)')
    parser.add_argument('--accum_steps', type=int, default=4,
                        help='Gradient accumulation steps for ABMIL training (default: 4)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (default: None)')
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
        print(f"[INFO] Random seed set to: {args.seed}")

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

    dataset_configs = {
        "Camelyon16": {
            "csv_path": "datasets/Camelyon16_test.csv",
            "label_mapping": {
                'Normal': 0,
                'Tumor': 1,
            },
            "feature_col": "conch_feature",
            "label_col": "label",
        },
        "TCGA_NSCLC": {
            "csv_path": "datasets/TCGA_NSCLC_test.csv",
            "label_mapping": {
                'TCGA-LUAD': 0,
                'TCGA-LUSC': 1,
            },
            "feature_col": "conch_feature",
            "label_col": "label",
        },
        "TCGA_RCC": {
            "csv_path": "datasets/TCGA_RCC_test.csv",
            "label_mapping": {
                'TCGA-KICH': 0,
                'TCGA-KIRC': 1,
                'TCGA-KIRP': 2,
            },
            "feature_col": "conch_feature",
            "label_col": "label",
        },
        "CPTAC_NSCLC": {
            "csv_path": "datasets/CPTAC_NSCLC_test.csv",
            "train_csv_path": "datasets/CPTAC_NSCLC_train.csv",
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
    k_shot = args.k_shot
    num_runs = args.num_runs
    max_iter = args.max_iter
    abmil_hidden_dim = args.abmil_hidden_dim
    abmil_lr = args.abmil_lr
    accum_steps = args.accum_steps

    res_path = f'./eval_results/pathflip_few_shot_abmil_{k_shot}shot.csv'
    if os.path.exists(res_path):
        results = pd.read_csv(res_path)
    else:
        results = pd.DataFrame(columns=["dataset", "k_shot", "run_id", "accuracy", "auc_roc", "f1", "ckpt_id"])

    for dataset_name in datasets_to_eval:
        print(f"\n{'='*60}")
        print(f"Evaluating dataset: {dataset_name} ( {k_shot}-shot )")
        print(f"{'='*60}")

        config = dataset_configs[dataset_name]
        
        num_classes = len(config["label_mapping"])
        
        if has_separate_train:
            train_dataset = GenericDataset(
                config["train_csv_path"],
                config["label_mapping"],
                feature_col=config.get("feature_col", "conch_feature"),
                label_col=config.get("label_col", "label")
            )
            test_dataset = GenericDataset(
                config["csv_path"],
                config["label_mapping"],
                feature_col=config.get("feature_col", "conch_feature"),
                label_col=config.get("label_col", "label")
            )
            
            print(f"\nExtracting train local features from {len(train_dataset)} samples...")
            train_features, train_labels = extract_local_features(train_dataset, model, device, dtype)
            print(f"Extracting test local features from {len(test_dataset)} samples...")
            test_features, test_labels = extract_local_features(test_dataset, model, device, dtype)
        else:
            dataset = GenericDataset(
                config["csv_path"],
                config["label_mapping"],
                feature_col=config.get("feature_col", "conch_feature"),
                label_col=config.get("label_col", "label")
            )
            print(f"\nExtracting local features from {len(dataset)} samples...")
            all_features, all_labels = extract_local_features(dataset, model, device, dtype)
        
            print(f"\nRun {run_id + 1}/{num_runs}:")
            
            run_seed = args.seed + run_id
            np.random.seed(run_seed)
            print(f"[INFO] Run {run_id + 1} using seed: {run_seed}")
            
            if has_separate_train:
                train_indices = []
                for c in range(num_classes):
                    class_indices = np.where(train_labels == c)[0]
                    np.random.shuffle(class_indices)
                    train_indices.extend(class_indices[:k_shot])
                
                run_train_features = [train_features[i] for i in train_indices]
                run_train_labels = train_labels[train_indices]
            else:
                train_indices = []
                test_indices = []
                
                for c in range(num_classes):
                    class_indices = np.where(all_labels == c)[0]
                    np.random.shuffle(class_indices)
                    train_indices.extend(class_indices[:k_shot])
                    test_indices.extend(class_indices[k_shot:])
                
                run_train_features = [all_features[i] for i in train_indices]
                run_train_labels = all_labels[train_indices]
                test_features = [all_features[i] for i in test_indices]
                test_labels = all_labels[test_indices]
            
            overall_accuracy, auc_roc, f1, predictions, labels, probs = slide_level_few_shot_abmil_probing(
                run_train_features, run_train_labels, test_features, test_labels, num_classes, 
                hidden_dim=abmil_hidden_dim, max_iter=max_iter, lr=abmil_lr, accum_steps=accum_steps
            )

            new_result = {
                "dataset": [dataset_name],
                "k_shot": [k_shot],
                "run_id": [run_id],
                "accuracy": [overall_accuracy],
                "auc_roc": [auc_roc],
                "f1": [f1],
                "ckpt_id": [ckpt_id]
            }
            new_result_df = pd.DataFrame(new_result)
            results = pd.concat([results, new_result_df], ignore_index=True)
            results.to_csv(res_path, index=False)
    
    print(f"Average results over {num_runs} runs:")

    summary_data = []
    for dataset_name in datasets_to_eval:
        dataset_results = results[results["dataset"] == dataset_name]
        mean_acc = dataset_results["accuracy"].mean()
        std_acc = dataset_results["accuracy"].std()
        mean_auc = dataset_results["auc_roc"].mean()
        std_auc = dataset_results["auc_roc"].std()
        mean_f1 = dataset_results["f1"].mean()
        std_f1 = dataset_results["f1"].std()
        print(f"\n{dataset_name}:")
        print(f"  Accuracy: {mean_acc:.4f}  {std_acc:.4f}")
        print(f"  AUC-ROC: {mean_auc:.4f}  {std_auc:.4f}")
        print(f"  F1: {mean_f1:.4f}  {std_f1:.4f}")

        summary_data.append({
            "dataset": dataset_name,
            "k_shot": k_shot,
            "accuracy_mean": mean_acc,
            "accuracy_std": std_acc,
            "auc_roc_mean": mean_auc,
            "auc_roc_std": std_auc,
            "f1_mean": mean_f1,
            "f1_std": std_f1,
            "num_runs": num_runs,
            "ckpt_id": ckpt_id
        })

    summary_df = pd.DataFrame(summary_data)
    summary_path = f'./eval_results/pathflip_few_shot_abmil_{k_shot}shot_summary.csv'
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    if os.path.exists(summary_path):
        existing_summary = pd.read_csv(summary_path)
        summary_df = pd.concat([existing_summary, summary_df], ignore_index=True)
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[INFO] Summary saved to: {summary_path}")

if __name__ == "__main__":
    main()

