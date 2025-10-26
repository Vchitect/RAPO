import random
import torch
import json
import networkx as nx
from sentence_transformers import SentenceTransformer
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
import csv
import os
import argparse

model = SentenceTransformer('./ckpt/all-MiniLM-L6-v2')

def open_dataset(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Retrieve graph construction.')
    parser.add_argument('--graph_data_dir', type=str, required=True)
    parser.add_argument('--output_filename', type=str, required=True)

    # Get command line arguments
    args = parser.parse_args()
    place_num = 3
    verb_num = 5
    topk_num = verb_num
    Retrieve_num = 30
    
    # Setting variables via command line arguments
    graph_data_dir = args.graph_data_dir
    output_filename = args.output_filename
    
    test_file = f'./data/test_prompts.txt'
    output_txt = f'./output/retrieve_words/{output_filename}.txt'
    output_csv = f'./output/retrieve_words/{output_filename}.csv'
    
    # load verb_to_idx, scenario_to_idx, place_to_idx，verb_in_sentence, scenario_in_sentence, place_in_sentence，verb_words_embed, scenario_words_embed, place_embed
    verb_to_idx = open_dataset(f'{graph_data_dir}/verb_to_idx.json')
    scenario_to_idx = open_dataset(f'{graph_data_dir}/scenario_to_idx.json')
    place_to_idx = open_dataset(f'{graph_data_dir}/place_to_idx.json')
    idx_to_place = {v: k for k, v in place_to_idx.items()}
    verb_in_sentence = open_dataset(f'{graph_data_dir}/verb_in_sentence.json')
    scenario_in_sentence = open_dataset(f'{graph_data_dir}/scenario_in_sentence.json')
    place_in_sentence = open_dataset(f'{graph_data_dir}/place_in_sentence.json')
    verb_words_embed = open_dataset(f'{graph_data_dir}/verb_words_embed.json')
    scenario_words_embed = open_dataset(f'{graph_data_dir}/scenario_words_embed.json')
    place_embed = open_dataset(f'{graph_data_dir}/place_embed.json')

    # Loading graph structure
    G_place_verb = nx.read_graphml(f'{graph_data_dir}/graph_place_verb.graphml')
    G_place_scene = nx.read_graphml(f'{graph_data_dir}/graph_place_scene.graphml')
    
    output_folder = os.path.dirname(output_txt)
    os.makedirs(output_folder, exist_ok=True)
    output_csv_folder = os.path.dirname(output_csv)
    os.makedirs(output_csv_folder, exist_ok=True)

    verb_obj_word, scenario_word, place = "", "", ""
    with open(test_file, 'r') as f:
        total_line = sum(1 for _ in f)
        f.seek(0)
        for i, line in enumerate(tqdm(f.readlines(), total=total_line)):
            sentence = line.replace('\n', "")
            potential_action, potential_sub_atmos, potential_scene = [], [], []
            sentence_emb = model.encode(sentence)
            sim = cosine_similarity(torch.tensor(sentence_emb).unsqueeze(0), torch.tensor(place_embed))
            top1_idx = torch.topk(sim, place_num).indices
            for idx in top1_idx.numpy().tolist():
                verb_neighbors = list(G_place_verb.neighbors(idx_to_place[idx]))
                scene_neighbors = list(G_place_scene.neighbors(idx_to_place[idx]))
                verb_random = random.sample(verb_neighbors, verb_num) if len(verb_neighbors) >= verb_num else verb_neighbors
                scene_random = random.sample(scene_neighbors, verb_num) if len(scene_neighbors) >= verb_num else scene_neighbors

                v_random_embed = []
                for v_random in verb_random:
                    if v_random in verb_to_idx:
                        v_random_embed.append(verb_words_embed[verb_to_idx[v_random]])
                if len(v_random_embed) > 0:
                    v_sim = cosine_similarity(torch.tensor(v_random_embed), torch.tensor(sentence_emb).unsqueeze(0))
                    v_random_candidate = torch.topk(v_sim, topk_num if len(v_sim) >= topk_num else len(v_sim)).indices.numpy().tolist()
                    potential_action += [verb_random[i] for i in v_random_candidate]

                s_random_embed, p_random_embed = [], []
                place_cross_word = []
                for s_random in scene_random:
                    if s_random in scenario_to_idx:
                        s_random_embed.append(scenario_words_embed[scenario_to_idx[s_random]])
                    else:
                        p_random_embed.append(place_embed[place_to_idx[s_random]])
                        place_cross_word.append(s_random)
                if len(s_random_embed) > 0:
                    s_sim = cosine_similarity(torch.tensor(s_random_embed), torch.tensor(sentence_emb).unsqueeze(0))
                    s_random_candidate = torch.topk(s_sim, topk_num if len(s_sim) >= topk_num else len(s_sim)).indices.numpy().tolist()

                if len(place_cross_word) > 0:
                    p_sim = cosine_similarity(torch.tensor(p_random_embed), torch.tensor(sentence_emb).unsqueeze(0))
                    p_random_candidate = torch.topk(p_sim, 1 if len(p_sim) >= 1 else len(p_sim)).indices.numpy().tolist()
                    potential_scene += [place_cross_word[k] for k in p_random_candidate]

                potential_sub_atmos += [scene_random[j] for j in s_random_candidate]
                potential_scene.append(idx_to_place[idx])

            word_set = set(potential_action + potential_sub_atmos + potential_scene)
            
            with open(output_txt, 'a') as f_txt:
                f_txt.write(f'{sentence}. {", ".join(word_set)}\n')

            with open(output_csv, 'a', encoding='utf-8', newline="") as fc:
                writer = csv.writer(fc)
                if i < Retrieve_num:
                    writer.writerow(['sentence', 'potential_action', 'potential_sub_atmos', 'potential_scene'])
                writer.writerow([sentence, set(potential_action), set(potential_sub_atmos), set(potential_scene)])

    print("Retrieve process is finished!")
