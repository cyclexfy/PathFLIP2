import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import yaml
import torch
import h5py
import pandas as pd
from transformers import AutoTokenizer
from argparse import Namespace
from tqdm import tqdm

from src.model.pathflip_vlm_pl import pathflip_vlm_pl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    with open(args_path, 'r') as f:
        args_dict = yaml.safe_load(f)
        args = Namespace(**args_dict)
    
    tokenizer = AutoTokenizer.from_pretrained(args.llm_name_or_path)
    if "<image>" not in tokenizer.get_vocab():
        tokenizer.add_tokens(["<image>"], special_tokens=True)
    
    model = pathflip_vlm_pl(args, tokenizer)

    ckpt_path_1 = "outputs/pathflip_vlm_stage2/checkpoint/pytorch_model-00001-of-00002.bin"
    ckpt_path_2 = "outputs/pathflip_vlm_stage2/checkpoint/pytorch_model-00002-of-00002.bin"

    state_dict_1 = torch.load(ckpt_path_1, map_location="cpu")
    print("Loaded first checkpoint shard")
    state_dict_2 = torch.load(ckpt_path_2, map_location="cpu")
    print("Loaded second checkpoint shard")
    merged_state_dict = {}
    merged_state_dict.update(state_dict_1)
    merged_state_dict.update(state_dict_2)
    missing_keys, unexpected_keys = model.load_state_dict(merged_state_dict, strict=False)
    print(f"Missing keys: {len(missing_keys)}")
    if missing_keys:
        print("Missing key list:")
        for key in missing_keys:
            print(f"  - {key}")
    print(f"Unexpected keys: {len(unexpected_keys)}")
    if unexpected_keys:
        print("Unexpected key list:")
        for key in unexpected_keys:
            print(f"  - {key}")
    print("=" * 80)
    model.eval()

    text_encoder_tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_name_or_path)

    return model, tokenizer, text_encoder_tokenizer

def generate_response(model, tokenizer, text_encoder_tokenizer, image_file, question):
    with h5py.File(image_file, 'r') as f:
        patch_features = torch.from_numpy(f['features'][:])
        patch_coords = torch.from_numpy(f['coords'][:])

    image_feats = patch_features.unsqueeze(0).to(device)
    image_coords = patch_coords.unsqueeze(0).to(device)
    messages = [{
        "role": "user",
        "content": f"<image>\n{question}"
    }]

    text_inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    ).to(device)
    
    input_ids = text_inputs.input_ids
    attn_mask = text_inputs.attention_mask

    query_text = text_encoder_tokenizer(question, return_tensors="pt").to(device)
    attn_mask_query = query_text.attention_mask
    input_ids_query = query_text.input_ids
    
    image_feats = image_feats.bfloat16()

    with torch.no_grad():
        outputs = model.generate(
            image_feats=image_feats,
            image_coords=image_coords,
            input_ids=input_ids,
            attn_mask=attn_mask,
            input_ids_query=input_ids_query,
            attn_mask_query=attn_mask_query,
            tokenizer=tokenizer,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )
    
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full_text.strip()

def build_question_with_options(row, is_choice=False):
    question = row['Question']
    
    if is_choice:
        option_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        options_text = []
        
        for letter in option_letters:
            if letter in row and pd.notna(row[letter]):
                options_text.append(f"{letter}. {row[letter]}")
            else:
                break
        
        if options_text:
            question = f"{question}\n\nOptions:\n" + "\n".join(options_text)
    
    return question

def test_model(model, tokenizer):
    image_file = "datasets/features/TCGA/BLCA/CONCH/h5_files/TCGA-GV-A40G-01Z-00-DX1.AD1A709F-A10C-4E69-B4ED-6361777361FD.h5"
    question = "Construct a brief narrative that clearly summarizes the essential findings of the pathology analysis from the whole slide image."
    
    response = generate_response(model, tokenizer, image_file, question)
    print(f"Response: {response}")


def main(csv_path, output_path, is_choice=False):
    print("Loading model...", flush=True)
    model, tokenizer, text_encoder_tokenizer = load_model()
    model.eval()
    model = model.to(device)
    model = model.bfloat16()
    print("Model loaded", flush=True)
    
    print(f"Reading CSV: {csv_path}", flush=True)
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples", flush=True)
    print(f"Choice mode: {is_choice}", flush=True)
    
    responses = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=""):
        # if idx >= 10:
        #     break
        image_file = row['Slide']
        question = build_question_with_options(row, is_choice)
        
        try:
            response = generate_response(model, tokenizer, text_encoder_tokenizer, image_file, question)
            responses.append(response)
            print(f"\nProcessed {idx + 1}/{len(df)}")
            print(f"Response: {response[:200]}..." if len(response) > 200 else f"Response: {response}")
        except Exception as e:
            print(f"\nSample {idx + 1} failed: {str(e)}")
            responses.append(f"ERROR: {str(e)}")
    
    df_result = df.iloc[:len(responses)].copy()
    df_result['Model_Response'] = responses
    df_result.to_csv(output_path, index=False)
    print(f"\nSaved results to {output_path} ({len(responses)} responses)", flush=True)

if __name__ == "__main__":
    print("Starting evaluation...", flush=True)
    
    csv_path = "datasets/SlideBench-VQA-TCGA.csv"
    output_path = "eval_results/SlideBench-VQA-TCGA_results.csv"
    is_choice = True
    
    main(csv_path, output_path, is_choice)

    
