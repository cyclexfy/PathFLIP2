import torch
import numpy as np
import json
import os
import h5py

from typing import Dict, Sequence
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from torch.utils.data.distributed import DistributedSampler


class dataset_pathflip_vlm(Dataset):
    def __init__(self,
        data_path=None,
        patch_sample=True,
        num_patch_samples=4096,
        tokenizer=None,
        text_encoder_tokenizer=None,
        conversation_type='qwen',
        max_text_length=1024,
        caption_repeat=5,
    ):
        super().__init__()
    
        self.patch_sample = patch_sample
        self.num_patch_samples = num_patch_samples
        self.tokenizer = tokenizer
        self.text_encoder_tokenizer = text_encoder_tokenizer
        self.conversation_type = conversation_type
        self.max_text_length = max_text_length

        if len(data_path) == 1:
            with open(data_path[0], 'r') as f:
                json_data = json.load(f)
        elif len(data_path) == 2:
            with open(data_path[0], 'r') as f:
                caption_data = json.load(f)
            with open(data_path[1], 'r') as f:
                vqa_data = json.load(f)
                json_data = (caption_data * caption_repeat) + vqa_data
        else:
            raise NotImplementedError

        valid_data = []

        for item in json_data:
            if 'image' in item and item['image']:
                if not os.path.exists(item['image']):
                    continue
                valid_data.append(item)

        self.json_data = valid_data

    def _load_slide_features(self, image_file):

        if image_file.endswith('.pt'):
            patch_features = torch.load(image_file, weights_only=True)
            patch_coords = None
        elif image_file.endswith('.h5'):
            try:
                with h5py.File(image_file, 'r') as f:
                    patch_features = torch.from_numpy(f['features'][:])
                    patch_coords = torch.from_numpy(f['coords'][:])
            except OSError as e:
                print(f"[WARNING] Corrupted H5 file detected: {image_file}, error: {str(e)}")
                raise
        else:
            raise NotImplementedError

        if self.patch_sample:
            max_patches = self.num_patch_samples
            n_samples = min(patch_features.shape[0], max_patches)
            idx = np.sort(np.random.choice(patch_features.shape[0], n_samples, replace=False))
            patch_features = patch_features[idx, :]
            if patch_coords is not None:
                patch_coords = patch_coords[idx, :]

            if n_samples < max_patches:
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
        
        return patch_features, patch_coords, patch_mask

    def _build_conversation(self, conversations):
        if self.conversation_type == 'qwen':
            messages = []
            for turn in conversations:
                role = "user" if turn["from"] == "human" else "assistant"
                content = turn["value"]
                messages.append({"role": role, "content": content})
        else:
            raise ValueError("Invalid conversation type")

        return messages

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, index):
        data_dict = self.json_data[index]
        data_return = {}

        query_text_tokenized = self.text_encoder_tokenizer(data_dict['conversations'][0]['value'], truncation=True, max_length=self.max_text_length, return_tensors=None)
        input_ids_query = query_text_tokenized['input_ids']

        messages = self._build_conversation(data_dict['conversations'])
        prompt_str = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        tokenized = self.tokenizer(
            prompt_str,
            truncation=True,
            max_length=self.max_text_length,
            add_special_tokens=False,
            return_tensors=None 
        )

        input_ids = tokenized["input_ids"]
        labels = list(input_ids)

        user_str = self.tokenizer.apply_chat_template(
            [messages[0]], tokenize=False, add_generation_prompt=True
        )
        user_ids = self.tokenizer(
            user_str, add_special_tokens=False, return_tensors=None
        )["input_ids"]

        data_return['input_ids'] = input_ids
        data_return['labels'] = labels

        for i in range(len(user_ids)):
            labels[i] = -100

        image_file = data_dict['image']
        try:
            patch_features, patch_coords, patch_mask = self._load_slide_features(image_file)
        except OSError:
            return self.__getitem__((index + 1) % len(self.json_data))

        data_return['image'] = patch_features
        data_return['image_coords'] = patch_coords
        data_return['image_mask'] = patch_mask
        data_return['input_ids_query'] = input_ids_query

        return data_return

def default_collate_fn(instances: Sequence[Dict], tokenizer=None, text_encoder_tokenizer=None):
    # check instances
    if not instances:
        raise ValueError("Input instances list is empty")

    max_len = max(len(x["input_ids"]) for x in instances)
    
    input_ids_batch = []
    labels_batch = []
    attn_mask_batch = []
    image_features = []
    image_coords = []
    image_mask = []

    input_ids_query_batch = []
    attn_mask_query_list = []
    max_len_query = max(len(x["input_ids_query"]) for x in instances)


    for x in instances:
        seq_len = len(x["input_ids"])
        pad_len = max_len - seq_len

        # padding input_ids
        input_ids = x["input_ids"] + [tokenizer.pad_token_id] * pad_len
        # padding labels
        labels = x["labels"] + [-100] * pad_len
        # attention mask
        attn_mask = [1] * seq_len + [0] * pad_len

        # padding input_ids_query
        seq_len_query = len(x["input_ids_query"])
        pad_len_query = max_len_query - seq_len_query
        input_ids_query = x["input_ids_query"] + [text_encoder_tokenizer.pad_token_id] * pad_len_query
        attn_mask_query = [1] * seq_len_query + [0] * pad_len_query
        input_ids_query_batch.append(input_ids_query)
        attn_mask_query_list.append(attn_mask_query)

        input_ids_batch.append(input_ids)
        labels_batch.append(labels)
        attn_mask_batch.append(attn_mask)
        image_features.append(x["image"])
        image_coords.append(x["image_coords"])
        image_mask.append(x["image_mask"])

    return {
        "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),  # [B, max_len]
        "labels": torch.tensor(labels_batch, dtype=torch.long),  # [B, max_len]
        "attn_mask": torch.tensor(attn_mask_batch, dtype=torch.long),  # [B, max_len]
        "image": torch.stack(image_features, dim=0),  # [B, N, D]
        "image_coords": torch.stack(image_coords, dim=0),  # [B, N, 2]
        "image_mask": torch.stack(image_mask, dim=0),  # [B, N]
        "input_ids_query": torch.tensor(input_ids_query_batch, dtype=torch.long),  # [B, max_len_query]
        "attn_mask_query": torch.tensor(attn_mask_query_list, dtype=torch.long),  # [B, max_len_query]
    }


class datamodule_pathflip_vlm(LightningDataModule):
    def __init__(
        self,
        args=None,
        tokenizer=None,
        text_encoder_tokenizer=None,
    ):
        super().__init__()
        self.patch_sample = args.patch_sample
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer
        self.text_encoder_tokenizer = text_encoder_tokenizer

        self.train_dataset = dataset_pathflip_vlm(
            data_path=args.train_data_path,
            patch_sample=args.patch_sample,
            num_patch_samples=args.num_patch_samples,
            tokenizer=tokenizer,
            text_encoder_tokenizer=text_encoder_tokenizer,
            conversation_type=args.conversation_type,
            max_text_length=args.max_text_length,
            caption_repeat=args.caption_repeat,
        )

        self.val_dataset = None
        if args.val_data_path is not None:
            self.val_dataset = dataset_pathflip_vlm(
                data_path=args.val_data_path,
                patch_sample=args.patch_sample,
                num_patch_samples=args.num_patch_samples,
                tokenizer=tokenizer,
                text_encoder_tokenizer=text_encoder_tokenizer,
                conversation_type=args.conversation_type,
                max_text_length=args.max_text_length,
            )

        self.collate_fn = lambda instances: default_collate_fn(instances, tokenizer=tokenizer, text_encoder_tokenizer=text_encoder_tokenizer)
    
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
    from src.tools.process_args_vlm import get_args
    from transformers import AutoTokenizer

    args = get_args()
    tokenizer = AutoTokenizer.from_pretrained(args.llm_name_or_path)
    if "<image>" not in tokenizer.get_vocab():
        tokenizer.add_tokens(["<image>"], special_tokens=True)

    # dataset = dataset_pathflip_vlm(
    #     data_path=args.train_data_path,
    #     patch_sample=args.patch_sample,
    #     patch_sample_num=args.patch_sample_num,
    #     tokenizer=tokenizer,  
    # )
    # data_i = dataset[0]
    # for k, v in data_i.items():
    #     print(k, v.shape)

    datamodule = datamodule_pathflip_vlm(
        train_data_path=args.train_data_path,
        val_data_path=None,
        patch_sample=args.patch_sample,
        patch_sample_num=args.patch_sample_num,
        tokenizer=tokenizer,
        text_max_length=args.text_max_length,
        conversation_type=args.conversation_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        args=args
    )

    train_dataloader = datamodule.train_dataloader()
    print(train_dataloader)

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
