import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import networkx as nx
from tqdm import tqdm
import json
import ast
from collections import defaultdict
import os

def process_and_save_graph_data(
    csv_file_path: str,
    data_prefix: str,
    model_path: str = './ckpt/all-MiniLM-L6-v2',
    valid_sentence_log: str = 'valid_sentence.txt'
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_path)

    # Initialize word-to-index dictionaries
    verb_to_idx, scenario_to_idx, place_to_idx = {}, {}, {}

    # Track sentence indices containing each word
    verb_in_sentence = defaultdict(list)
    scenario_in_sentence = defaultdict(list)
    place_in_sentence = defaultdict(list)

    # Store embeddings
    verb_words_embed, scenario_words_embed, place_embed = [], [], []

    # Cache for already encoded words
    verb_cache, scenario_cache, place_cache = {}, {}, {}

    # Graphs for co-occurrence relationships
    G_place_scene = nx.Graph()
    G_place_verb = nx.Graph()

    data_info = {}
    texts = []
    valid_sentence = 0

    df = pd.read_csv(csv_file_path)

    # Read and preprocess CSV data
    for i, row in df.iterrows():
        sentence = row['Input']
        try:
            verb_obj_word = ast.literal_eval(row['verb_obj_word'])
            scenario_word = ast.literal_eval(row['scenario_word'])
            place = ast.literal_eval(row['place'])
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing row {i}: {e}")
            continue

        # Handle empty lists
        verb_obj_word = [] if not verb_obj_word or verb_obj_word[0] == '' else verb_obj_word
        scenario_word = [] if not scenario_word or scenario_word[0] == '' else scenario_word
        place = [] if not place or place[0] == '' else place

        texts.append([verb_obj_word, scenario_word, place])

        if len(verb_obj_word) > 0 and len(scenario_word) > 0 and len(place) > 0:
            with open(valid_sentence_log, 'a') as f_valid:
                f_valid.write(f'{sentence}\n')
            valid_sentence += 1

    print(f"{len(texts)} sentences have been read from the CSV file.")

    v_idx = s_idx = p_idx = 0
    valid_cnt = 0

    # Batch process all tokens and encode them if needed
    for i in tqdm(range(len(texts))):
        verbs, scenes, places = texts[i]
        if len(verbs) and len(scenes) and len(places):
            for p in places:
                # Process scene tokens
                for s in scenes:
                    if s not in scenario_cache:
                        s_emb = model.encode(s)
                        scenario_cache[s] = s_emb.tolist()
                        if s not in scenario_to_idx:
                            scenario_to_idx[s] = s_idx
                            s_idx += 1
                            scenario_words_embed.append(scenario_cache[s])
                    scenario_in_sentence[s].append(valid_cnt)
                    G_place_scene.add_edge(p, s)

                # Process verb tokens
                for v in verbs:
                    if v not in verb_cache:
                        v_emb = model.encode(v)
                        verb_cache[v] = v_emb.tolist()
                        if v not in verb_to_idx:
                            verb_to_idx[v] = v_idx
                            v_idx += 1
                            verb_words_embed.append(verb_cache[v])
                    verb_in_sentence[v].append(valid_cnt)
                    G_place_verb.add_edge(p, v)

                # Process place tokens
                if p not in place_cache:
                    p_emb = model.encode(p)
                    place_cache[p] = p_emb.tolist()
                    if p not in place_to_idx:
                        place_to_idx[p] = p_idx
                        p_idx += 1
                        place_embed.append(place_cache[p])
                place_in_sentence[p].append(valid_cnt)

            valid_cnt += 1

    assert valid_cnt == valid_sentence
    data_info.update({
        'valid_sentence': valid_sentence,
        'p_idx': p_idx,
        's_idx': s_idx,
        'v_idx': v_idx
    })

    os.makedirs(data_prefix, exist_ok=True)

    # Save dictionaries
    def save_json(data, name):
        with open(os.path.join(data_prefix, f'{name}.json'), 'w') as f:
            json.dump(data, f, indent=4)
        print(f"{name} saved!")

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

    # Save graph files
    nx.write_graphml(G_place_verb, os.path.join(data_prefix, 'graph_place_verb.graphml'))
    nx.write_graphml(G_place_scene, os.path.join(data_prefix, 'graph_place_scene.graphml'))
    print("Graphs are saved!")

# Example usage
if __name__ == "__main__":
    process_and_save_graph_data(
        csv_file_path="./data/graph_test1.csv",
        data_prefix="./graph/graph_test1"
    )
