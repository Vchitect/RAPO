import os
import json
import ast
import torch
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from collections import defaultdict
from sentence_transformers import SentenceTransformer

def open_dataset(filename):
    """Load a JSON file and return its content."""
    with open(filename, 'r') as file:
        return json.load(file)

def update_graph_from_csv(
    csv_file: str,
    data_prefix_before: str,
    data_prefix_after: str,
    model_path: str = './ckpt/all-MiniLM-L6-v2',
    valid_sentence_log: str = 'valid_sentence.txt'
):
    """Update word embeddings, indices, and co-occurrence graphs from new CSV data."""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_path)

    # Load dictionaries
    verb_to_idx = open_dataset(f'{data_prefix_before}/verb_to_idx.json')
    scenario_to_idx = open_dataset(f'{data_prefix_before}/scenario_to_idx.json')
    place_to_idx = open_dataset(f'{data_prefix_before}/place_to_idx.json')

    # Load sentence index mappings
    verb_in_sentence = open_dataset(f'{data_prefix_before}/verb_in_sentence.json')
    scenario_in_sentence = open_dataset(f'{data_prefix_before}/scenario_in_sentence.json')
    place_in_sentence = open_dataset(f'{data_prefix_before}/place_in_sentence.json')

    # Load embeddings
    verb_words_embed = open_dataset(f'{data_prefix_before}/verb_words_embed.json')
    scenario_words_embed = open_dataset(f'{data_prefix_before}/scenario_words_embed.json')
    place_embed = open_dataset(f'{data_prefix_before}/place_embed.json')

    # Load graphs
    G_place_verb = nx.read_graphml(f'{data_prefix_before}/graph_place_verb.graphml')
    G_place_scene = nx.read_graphml(f'{data_prefix_before}/graph_place_scene.graphml')

    # Load meta information
    data_info = open_dataset(f'{data_prefix_before}/data_info.json')
    valid_sentence = valid_cnt = data_info['valid_sentence']
    v_idx, s_idx, p_idx = data_info['v_idx'], data_info['s_idx'], data_info['p_idx']

    # Cache to avoid redundant encoding
    verb_cache, scenario_cache, place_cache = {}, {}, {}

    # Read new CSV data
    df = pd.read_csv(csv_file)
    texts = []
    
    for i, row in df.iterrows():
        sentence = row['Input']
        try:
            verb_obj_word = ast.literal_eval(row['verb_obj_word'])
            scenario_word = ast.literal_eval(row['scenario_word'])
            place = ast.literal_eval(row['place'])
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing row {i}: {e}")
            continue

        # Sanitize empty lists
        verb_obj_word = [] if not verb_obj_word or verb_obj_word[0] == '' else verb_obj_word
        scenario_word = [] if not scenario_word or scenario_word[0] == '' else scenario_word
        place = [] if not place or place[0] == '' else place

        texts.append([verb_obj_word, scenario_word, place])

        if len(verb_obj_word) > 0 and len(scenario_word) > 0 and len(place) > 0:
            with open(valid_sentence_log, 'a') as f_valid:
                f_valid.write(f'{sentence}\n')
            valid_sentence += 1

    print(f"{len(texts)} sentences have been read from the CSV file.")

    # Process and update graph/embedding/index info
    for i in tqdm(range(len(texts))):
        verbs, scenes, places = texts[i]
        if len(verbs) and len(scenes) and len(places):
            for p in places:
                p = p.strip()
                for s in scenes:
                    s = s.strip()
                    if s not in scenario_cache:
                        s_emb = model.encode(s)
                        scenario_cache[s] = s_emb.tolist()
                        if s not in scenario_to_idx:
                            scenario_to_idx[s] = s_idx
                            s_idx += 1
                            scenario_words_embed.append(scenario_cache[s])
                    scenario_in_sentence.setdefault(s, []).append(valid_cnt)
                    G_place_scene.add_edge(p, s)

                for v in verbs:
                    v = v.strip()
                    if v not in verb_cache:
                        v_emb = model.encode(v)
                        verb_cache[v] = v_emb.tolist()
                        if v not in verb_to_idx:
                            verb_to_idx[v] = v_idx
                            v_idx += 1
                            verb_words_embed.append(verb_cache[v])
                    verb_in_sentence.setdefault(v, []).append(valid_cnt)
                    G_place_verb.add_edge(p, v)

                if p not in place_cache:
                    p_emb = model.encode(p)
                    place_cache[p] = p_emb.tolist()
                    if p not in place_to_idx:
                        place_to_idx[p] = p_idx
                        p_idx += 1
                        place_embed.append(place_cache[p])
                place_in_sentence.setdefault(p, []).append(valid_cnt)

            valid_cnt += 1

    print(f"Valid sentences processed: {valid_cnt}")
    print(f"Original valid sentence count: {valid_sentence}")

    # Update and save metadata
    data_info.update({
        'valid_sentence': valid_sentence,
        'p_idx': p_idx,
        's_idx': s_idx,
        'v_idx': v_idx
    })

    os.makedirs(data_prefix_after, exist_ok=True)

    def save_json(data, name):
        with open(os.path.join(data_prefix_after, f'{name}.json'), 'w') as f:
            json.dump(data, f, indent=4)
        print(f"{name} saved!")

    # Save all updated data
    save_json(data_info, 'data_info')
    save_json(verb_to_idx, 'verb_to_idx')
    save_json(scenario_to_idx, 'scenario_to_idx')
    save_json(place_to_idx, 'place_to_idx')
    save_json(verb_in_sentence, 'verb_in_sentence')
    save_json(scenario_in_sentence, 'scenario_in_sentence')
    save_json(place_in_sentence, 'place_in_sentence')
    save_json(verb_words_embed, 'verb_words_embed')
    save_json(scenario_words_embed, 'scenario_words_embed')
    save_json(place_embed, 'place_embed')

    # Save updated graphs
    nx.write_graphml(G_place_verb, os.path.join(data_prefix_after, 'graph_place_verb.graphml'))
    nx.write_graphml(G_place_scene, os.path.join(data_prefix_after, 'graph_place_scene.graphml'))

    print("Graphs are saved!")

# Example usage
if __name__ == "__main__":
    update_graph_from_csv(
        csv_file="./data/graph_test2.csv",
        data_prefix_before="./graph/graph_test1",
        data_prefix_after="./graph/graph_test2"
    )
