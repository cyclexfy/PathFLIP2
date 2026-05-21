import torch
import numpy as np
import json
import os
import h5py

from typing import Dict, Sequence
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from torch.utils.data.distributed import DistributedSampler


class dataset_pathflip(Dataset):

    def __init__(self,
        data_path=None,
        patch_sample=True,
        num_patch_samples=4096,
        num_text_samples=8,
        tokenizer=None,
        detail_description=True,
        max_text_length=1024,
    ):
        super().__init__()
    
        self.patch_sample = patch_sample
        self.num_patch_samples = num_patch_samples
        self.num_text_samples = num_text_samples
        self.tokenizer = tokenizer
        self.detail_description = detail_description
        self.max_text_length = max_text_length

        if data_path.endswith('.json'):
            with open(data_path, 'r') as f:
                json_data = json.load(f)
        else:
            raise NotImplementedError
        
        for item in json_data:
            if 'image' in item and item['image']:
                feature_path = item['image']
                if not os.path.exists(feature_path):
                    continue
                valid_data.append(item)
        self.json_data = valid_data

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, index):
        data_dict = self.json_data[index]
        data_return = {}

        # global
        text_global = data_dict['caption']
        text_global_tokenized = self.tokenizer(text_global, truncation=True, max_length=self.max_text_length, return_tensors=None)
        input_ids_global = text_global_tokenized['input_ids']

        # local (+sample)
        if self.detail_description:
            local_text_description = data_dict.get('fine_grained_concepts', [])
            current_len = len(local_text_description)
            target_k = getattr(self, 'num_text_samples', 8) 
            if current_len == 0:
                selected_items = []
            elif current_len == target_k:
                selected_items = local_text_description
            elif current_len > target_k:
                indices = np.random.choice(current_len, size=target_k, replace=False)
            else:
                base_indices = np.arange(current_len)
                num_padding = target_k - current_len
                
                indices = np.concatenate([base_indices, pad_indices])
                selected_items = [local_text_description[i] for i in indices]

            input_ids_local = []
            for item in selected_items:
                text_local = item['visual_description']
                text_local_tokenized = self.tokenizer(
                    text_local, 
                    truncation=True, 
                    max_length=self.max_text_length, 
                    return_tensors=None
                )
                input_ids_local.append(text_local_tokenized['input_ids'])
            
        else:
            input_ids_local = None

        data_return['input_ids_global'] = input_ids_global
        data_return['input_ids_local'] = input_ids_local

        # processing image features
        image_file = data_dict['image']
        if image_file.endswith('.pt'):
            patch_features = torch.load(image_file, weights_only=True)
            patch_coords = torch.zeros((patch_features.shape[0], 2), dtype=torch.long)
        elif image_file.endswith('.h5'):
            with h5py.File(image_file, 'r') as f:
                patch_features = torch.from_numpy(f['features'][:])
                patch_coords = torch.from_numpy(f['coords'][:])
        else:
            raise NotImplementedError

        if self.patch_sample:
            # random sampling patches
            max_patches = self.num_patch_samples
            n_samples = min(patch_features.shape[0], max_patches)
            idx = np.sort(np.random.choice(patch_features.shape[0], n_samples, replace=False))
            patch_features = patch_features[idx, :]
            if patch_coords is not None:
                patch_coords = patch_coords[idx, :]

            if n_samples < max_patches:
                ### random padding
                num_to_pad = max_patches - n_samples
                pad_indices = np.random.choice(patch_features.shape[0], num_to_pad)
                pad_features = patch_features[pad_indices]
                patch_features = torch.cat([patch_features, pad_features], dim=0)

                if patch_coords is not None:
                    pad_coords = patch_coords[pad_indices]
                    patch_coords = torch.cat([patch_coords, pad_coords], dim=0)

                patch_mask = torch.cat([torch.ones(n_samples), torch.zeros(max_patches - n_samples)])
            else:
                patch_mask = torch.ones(n_samples)
        else:
            patch_mask = torch.ones(patch_features.shape[0])

        data_return['image'] = patch_features
        data_return['image_coords'] = patch_coords
        data_return['image_mask'] = patch_mask

        return data_return


def default_collate_fn(instances: Sequence[Dict], tokenizer=None):
    # check instances
    if not instances:
        raise ValueError("Input instances list is empty")

    input_ids_global_list = []
    attn_mask_global_list = []
    input_ids_local_list = []
    attn_mask_local_list = []
    image_features = []
    image_coords = []
    image_mask = []

    max_len_global = max(len(x["input_ids_global"]) for x in instances)
    max_len_local = 0

    local_input = instances[0]['input_ids_local'] is not None
    if local_input:
        for instance in instances:
            max_len_local = max(max_len_local, max(len(x) for x in instance['input_ids_local']))

    for instance in instances:
        # global
        seq_len_global = len(instance["input_ids_global"])
        pad_len_global = max_len_global - seq_len_global
        input_ids_global_list.append(instance['input_ids_global'] + [tokenizer.pad_token_id] * pad_len_global)
        attn_mask_global = [1] * seq_len_global + [0] * pad_len_global
        attn_mask_global_list.append(attn_mask_global)

        # local
        if local_input:
            input_ids_local_list_i = []
            attn_mask_local_list_i = []
            for local_ids_local_i in instance['input_ids_local']:
                seq_len_local = len(local_ids_local_i)
                pad_len_local = max_len_local - seq_len_local
                input_ids_local_list_i.append(local_ids_local_i + [tokenizer.pad_token_id] * pad_len_local)
                attn_mask_local_list_i.append([1] * seq_len_local + [0] * pad_len_local)
            input_ids_local_list.append(input_ids_local_list_i)
            attn_mask_local_list.append(attn_mask_local_list_i)

        image_features.append(instance['image'])
        image_mask.append(instance['image_mask'])
        image_coords.append(instance['image_coords'])

    
    image_features = torch.stack(image_features)
    image_mask = torch.stack(image_mask)
    image_coords = torch.stack(image_coords)

    if local_input:
        return {
            'input_ids_global': torch.tensor(input_ids_global_list, dtype=torch.long), # [B, max_len_global]
            'attn_mask_global': torch.tensor(attn_mask_global_list, dtype=torch.long), # [B, max_len_global]
            'input_ids_local': torch.tensor(input_ids_local_list, dtype=torch.long), # [B, num_local, max_len_local]
            'attn_mask_local': torch.tensor(attn_mask_local_list, dtype=torch.long), # [B, num_local, max_len_local]
            'image': image_features,
            'image_coords': image_coords,
            'image_mask': image_mask
        }
    else:
        return {
            'input_ids_global': torch.tensor(input_ids_global_list, dtype=torch.long), # [B, max_len_global]
            'attn_mask_global': torch.tensor(attn_mask_global_list, dtype=torch.long), # [B, max_len_global]
            'input_ids_local': None,
            'attn_mask_local': None,
            'image': image_features,
            'image_coords': image_coords,
            'image_mask': image_mask
        }


class datamodule_pathflip(LightningDataModule):
    def __init__(
        self,
        train_data_path=None,
        val_data_path=None,
        patch_sample=True,
        num_patch_samples=4096,
        num_text_samples=8,
        tokenizer=None,
        max_text_length=1024,
        max_dataset_length=None,
        batch_size: int = 4,
        num_workers: int = 4,
        args=None,
    ):
        super().__init__()
        self.patch_sample = patch_sample 
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.args = args
        self.tokenizer = tokenizer

        self.train_dataset = dataset_pathflip(
            data_path=train_data_path,
            patch_sample=patch_sample,
            num_patch_samples=num_patch_samples,  
            num_text_samples=num_text_samples,
            tokenizer=tokenizer,  
            max_text_length=max_text_length,
        )

        if val_data_path is not None:
            self.val_dataset = dataset_pathflip(
                data_path=val_data_path,
                patch_sample=patch_sample,
                num_patch_samples=num_patch_samples,
                # num_text_samples=num_text_samples,
                tokenizer=tokenizer,
                detail_description=False,
                max_text_length=max_text_length
            )

        self.collate_fn = lambda instances: default_collate_fn(instances, self.tokenizer)
    
    def train_dataloader(self):
        if self.trainer and self.trainer.world_size > 1:
            sampler = DistributedSampler(self.train_dataset)
            shuffle = False
        else:
            sampler = None
            shuffle = True

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            shuffle=shuffle,
            sampler=sampler,
            collate_fn=self.collate_fn,
        )
        
        return train_loader

    def val_dataloader(self):
        if self.trainer and self.trainer.world_size > 1:
            sampler = DistributedSampler(self.val_dataset)
            shuffle = False
        else:
            sampler = None
            shuffle = False

        if hasattr(self, 'val_dataset'):
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False,
                persistent_workers=True,
                sampler=sampler,
                shuffle=shuffle,
                collate_fn=self.collate_fn,
            )

            return val_loader

if __name__ == "__main__":
    from src.tools.process_args_align import get_args
    from transformers import AutoTokenizer

    args = get_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    datamodule = datamodule_pathflip(
        train_data_path=args.train_data_path,
        val_data_path=args.val_data_path,
        patch_sample=args.patch_sample,
        num_patch_samples=args.num_patch_samples,
        num_text_samples=args.num_text_samples, 
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        args=args
    )

    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    # print(val_dataloader)

    print("\n1. Inspect one batch:")
    for batch_idx, batch in enumerate(train_dataloader):
        print(f"Batch {batch_idx} keys: {batch.keys()}")
        for key, value in batch.items():
            if torch.is_tensor(value):
                print(f"{key}: {value.shape}")
            elif isinstance(value, dict):
                print(f"{key}: contains {len(value)} nested keys")
                for sub_key, sub_value in value.items():
                    if torch.is_tensor(sub_value):
                        print(f"{sub_key}: {sub_value.shape}")
            else:
                print(f"{key}:  {type(value)}")
        
        break
