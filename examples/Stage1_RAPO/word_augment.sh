#!/bin/bash

input_path="./data/test_prompts.txt"
output_dir="./output/refactor/"
SIMILARITY_THRESHOLD=0.6

output_simrank="${output_dir}/simrank.csv"
output_selected_modifiers="${output_dir}/selected_modifiers.txt"
output_interactive_merging="${output_dir}/merging_reuslts.txt"

python word_augment.py\
--retrieved_words "./output/retrieve_words/retrieved_words.txt" \
--input_path "${input_path}" \
--output_simrank "${output_simrank}" \
--output_selected_modifiers "${output_selected_modifiers}" \
--output_interactive_merging "${output_interactive_merging}" \
--SIMILARITY_THRESHOLD "${SIMILARITY_THRESHOLD}" \