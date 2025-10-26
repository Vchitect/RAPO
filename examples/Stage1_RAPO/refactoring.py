import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse

def get_output(prompt):
    template = (
        'Refine the sentence: "{}" to contain subject description, action, scene description. '
        'Transform user-entered text into a concise, detailed description with a specific structure. '
        '(Optional: camera language, light and shadow, atmosphere) and conceive some additional actions to make the sentence more dynamic. '
        'Make sure it is a fluent sentence, not nonsense.'
    )
    prompt_text = template.format(prompt)
    messages = [
        {"role": "system", "content": "You are a caption refiner."},
        {"role": "user",   "content": prompt_text}
    ]

    # prepare inputs
    input_ids = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([input_ids], return_tensors="pt").to(device)

    # generate
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    # strip prompt prefix
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return responses[0]


def get_start_index(txt_path):
    """
    Read existing output file to determine resume index (number of lines).
    """
    if os.path.exists(txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            return len(f.readlines())
    return 0


def main():
    # determine from which line to resume
    start_idx = get_start_index(output_path)

    # read all prompts
    with open(input_path, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]

    # open output file for append
    with open(output_path, 'a', encoding='utf-8') as outf:
        for i in tqdm(range(start_idx, len(prompts)), desc="Refining prompts"):
            prompt = prompts[i]
            try:
                refined = get_output(prompt)
            except Exception as e:
                refined = f"[ERROR] {e}"
            outf.write(refined + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Refine captions and output to text file')
    parser.add_argument(
        '--mode_path', type=str,
        default='llama3_8B_lora_merged_cn',
        help='Model path or identifier'
    )
    parser.add_argument(
        '--input_word_augmentation', type=str,
        default='./output/refactor/merging_reuslts.txt',
        help='Path to input text prompts'
    )
    parser.add_argument(
        '--output_refactoring', type=str,
        default='./output/refactor/refactoring_results.txt',
        help='Path to output text file'
    )
    args = parser.parse_args()

    # setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.mode_path, trust_remote_code=True)
    model     = AutoModelForCausalLM.from_pretrained(
        args.mode_path,
        trust_remote_code=True
    ).to(device).eval()

    input_path  = args.input_word_augmentation
    output_path = args.output_refactoring

    main()
