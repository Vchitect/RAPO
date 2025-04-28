import random
import torch
import json
import networkx as nx
from sentence_transformers import SentenceTransformer
from torch.nn.functional import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import csv
import os
import argparse
import pandas as pd
import string
import re
import numpy as np


def open_dataset(filename):
    """Load a JSON file"""
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

# Remove punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

# Compute similarity between two texts
def compute_similarity(text1, text2, model, embeddings_cache):
    if text1 not in embeddings_cache:
        embeddings_cache[text1] = model.encode(text1)
    if text2 not in embeddings_cache:
        embeddings_cache[text2] = model.encode(text2)
    embedding1 = torch.tensor(embeddings_cache[text1]).unsqueeze(0)
    embedding2 = torch.tensor(embeddings_cache[text2]).unsqueeze(0)
    similarity = cosine_similarity(embedding1, embedding2).item()
    return similarity


# Extract similarity score from string
def extract_sim_score(text):
    match = re.search(r'sim_score=([0-9.]+)', text)
    if match:
        return float(match.group(1))
    return 0.0

# Extract the final output from model output
def extract_output(model_output):
    # Assume output contains 'Final Output: ' followed by the desired result
    match = re.search(r'Final Output:\s*(.*)', model_output, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return model_output.strip()


# Get max number of columns in a CSV file
def get_max_columns(input_csv_path):
    max_columns = 0
    with open(input_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            max_columns = max(max_columns, len(row))
    return max_columns

### Similarity Ranking
def Similarity_Ranking(input_txt, simrank_path, SentenceTransformer_model):
    """
    input_txt: each line formatted as 'prefix.suffix1,suffix2,...'
    simrank_path: path to output CSV
    SentenceTransformer_model: model for computing similarity
    """
    embeddings_cache = {}

    with open(input_txt, 'r', encoding='utf-8') as f, \
         open(simrank_path, 'w', encoding='utf-8', newline='') as csv_file:

        writer = csv.writer(csv_file, quoting=csv.QUOTE_MINIMAL)

        for line in f:
            line = line.strip()
            if not line:
                continue

            # 1. Split at the first '.'
            idx = line.find('.')
            if idx == -1:
                # Skip if no '.' is found
                continue

            first_part = line[:idx+1].strip()
            rest = line[idx+1:].strip()

            # 2. Split suffixes by comma and remove whitespace
            other_parts = [part.strip() for part in rest.split(',') if part.strip()]

            # 3. Combine into parts list
            parts = [first_part] + other_parts

            # 4. Remove punctuation from parts
            clean_parts_no_punct = [parts[0]] + [
                remove_punctuation(part) for part in parts[1:]
            ]

            # 5. Compute similarity scores
            processed_with_similarity = []
            before_period = clean_parts_no_punct[0]
            for part in clean_parts_no_punct[1:]:
                sim_score = compute_similarity(
                    before_period, part, SentenceTransformer_model, embeddings_cache
                )
                sim_score = round(sim_score, 4)
                processed_with_similarity.append((part, sim_score))
                print(f"processed_part: {part}, sim_score={sim_score}")

            # 6. Sort by similarity descending
            processed_with_similarity_sorted = sorted(
                processed_with_similarity,
                key=lambda x: x[1],
                reverse=True
            )

            # 7. Format output fields
            formatted_processed = [first_part]
            for part, sim in processed_with_similarity_sorted:
                formatted_processed.append(f"{part}, sim_score={sim}")

            # 8. Write a row to the output CSV
            csv_line = [first_part, rest] + formatted_processed
            writer.writerow(csv_line)

    print(f"Similarity Ranking completed, results saved to {simrank_path}")
    return simrank_path
### Similarity Ranking

### Iteractively Merging 
def Iteractively_Merging(simrank_path, merging_path, selected_modifiers, SIMILARITY_THRESHOLD):
    max_columns = get_max_columns(simrank_path)
    print(f"Max columns: {max_columns}")
    simrank_file = pd.read_csv(simrank_path, header=None, names=[f"col{i}" for i in range(max_columns)], encoding='utf-8')
    output_data = []
    for index, row in simrank_file.iterrows():
        try:
            original_text = row.iloc[1]  
            modifiers = row.iloc[2:]  
            modifiers_with_scores = []
            for modifier in modifiers:
                if pd.isna(modifier):
                    continue
                parts = modifier.split(", sim_score=")
                if len(parts) == 2:
                    mod_text = parts[0].strip()
                    sim_score = extract_sim_score(modifier)
                    if sim_score >= SIMILARITY_THRESHOLD:
                        modifiers_with_scores.append((mod_text, sim_score))
            modifiers_with_scores_sorted = sorted(modifiers_with_scores, key=lambda x: x[1], reverse=True)

            current_description = original_text
            processed_outputs = []
            for modifier, sim_score in modifiers_with_scores_sorted:
                if sim_score < SIMILARITY_THRESHOLD:
                    print(f"Row {index+1}: sim_score={sim_score} below threshold {SIMILARITY_THRESHOLD}, stopping inference.")
                    break  
                try:
                    prompt = f"""Suppose you are a Text Rewriter, and your role is to transform user-entered text into a concise, detailed description. You receive two inputs from the user: description body and relevant modifiers. Your task is to enrich the description body with relevant modifiers while retaining the description body. You should ensure that the output text is coherent, contextually relevant, and follows the same structure as the examples provided.
Examples provided:
(1) Description body: a group of dancers performing a ballet routine in a studio. The dancers are wearing ballet shoes.  
Relevant modifiers: dressed in black leotards. 
Output: a group of dancers performing a ballet routine in a studio. The dancers are wearing ballet shoes and are dressed in black leotards. 
(2) Description body: a woman sitting at a desk and working on her laptop. 
Relevant modifiers: appears to be focused on her work. 
Output: a woman sitting at a desk and working on her laptop, appears to be focused on her work. 
(3) Description body: they seem to be having a good time and enjoying each other's company. 
Relevant modifiers: a casual and relaxed setting.  
Output: They seem to be having a good time and enjoying each other's company. It appears to be a casual and relaxed setting. 
(4) Description body: a woman preparing a delicious meal in her kitchen. 
Relevant modifiers: cutting various fruits and vegetables on a cutting board. 
Output: a woman preparing a delicious meal in her kitchen. She is seen cutting various fruits and vegetables on a cutting board and placing them on a tray. 
The Description body: {current_description}, Relevant modifiers: {modifier}, Final Output:"""
                    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
                    attention_mask = torch.ones(inputs.shape, dtype=torch.long, device=device)
                    outputs = merging_model.generate(
                        inputs,
                        max_new_tokens=500,
                        pad_token_id=tokenizer.eos_token_id,
                        attention_mask=attention_mask
                    )
                    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    extracted_output = extract_output(output_text)
                    processed_output = f"{modifier}, sim_score={sim_score}"
                    processed_outputs.append(processed_output)
                    current_description = extracted_output
                    print(f"Row {index+1}, Step with modifier '{modifier}': {processed_output}")
                except Exception as e:
                    print(f"Error during inference on row {index+1}: {e}")
                    continue  

            # Save final description
            output_row = {
                "original_text": original_text,
                "before_period": row.iloc[0] if len(row) > 0 else "",
                "final_description": current_description,  
                "processed_outputs": "; ".join(processed_outputs)
            }
            output_data.append(output_row)
            with open(merging_path, 'a', encoding='utf-8') as desc_file:
                desc_file.write(current_description + '\n')

        except Exception as e:
            print(f"Error processing row {index+1}: {e}")
            continue  
    output_df = pd.DataFrame(output_data)
    output_df.to_csv(selected_modifiers, index=False, encoding='utf-8')
    return merging_path, selected_modifiers, output_data
### Iteractively Merging 



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Retrieve graph construction.')
    parser.add_argument('--retrieved_words', type=str, default="./output/all_dimension.csv", help='path of retrieved modifiers')
    parser.add_argument('--pretrained_SentenceTransformer', type=str, default='./ckpt/all-MiniLM-L6-v2', help='SentenceTransformer model path')
    parser.add_argument('--pretrained_merging', type=str, default='./ckpt/Mistral-7B-Instruct-v0.3/', help='merging model path')
    parser.add_argument('--input_path', type=str, required=True, help='input text file path')
    parser.add_argument('--output_simrank', type=str, required=True, help='output ranking CSV')
    parser.add_argument('--output_selected_modifiers', type=str, required=True, help='output selected modifiers CSV')
    parser.add_argument('--output_interactive_merging', type=str, required=True, help='results after interactive merging')
    parser.add_argument('--SIMILARITY_THRESHOLD', type=float, required=True, help='similarity threshold')

    args = parser.parse_args()
    SentenceTransformer_model = SentenceTransformer(args.pretrained_SentenceTransformer)
    merging_model_path = args.pretrained_merging
    input_path = args.input_path
    retrieved_words = args.retrieved_words
    simrank_path = args.output_simrank
    selected_modifiers_path = args.output_selected_modifiers
    merging_path = args.output_interactive_merging
    SIMILARITY_THRESHOLD = args.SIMILARITY_THRESHOLD

    output_folder = os.path.dirname(simrank_path)
    os.makedirs(output_folder, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(merging_model_path, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    merging_model = AutoModelForCausalLM.from_pretrained(merging_model_path, torch_dtype=torch.bfloat16, device_map="auto", offload_state_dict=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"


    ### Similarity Ranking
    simrank_path = Similarity_Ranking(retrieved_words, simrank_path, SentenceTransformer_model)
    ### Similarity Ranking
    
    ### Iteractively Merging 
    merging_path, selected_modifiers_path, output_data = Iteractively_Merging(simrank_path, merging_path, selected_modifiers_path, SIMILARITY_THRESHOLD)
    ### Iteractively Merging 
    