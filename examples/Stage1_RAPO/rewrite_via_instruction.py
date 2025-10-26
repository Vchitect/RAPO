import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
import argparse
import os

def extract_output(model_output):
    match = re.search(r'Final Output:\s*(.*)', model_output, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return model_output.strip()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process text and generate output.')
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()

    input_file_path = args.input_path
    output_path = args.output_path
    
    model_id = './ckpt/Mistral-7B-Instruct-v0.3/'
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    output_data = []
    with open(input_file_path, "r") as infile:
        for line in infile:
            The_current_input = line.strip()  
            prompts = [f"""Please limit your output to 30 words or less. Suppose you are a Text Aligner, and your role is to transform user-entered text into a concise, detailed description with a specific structure. You should ensure that the output text is coherent, contextually relevant, and follows the same structure as the examples provided. The output should frequently use common phrases like 'She is,' 'appears to,' 'There are,' 'seems to,' 'It appears to,' and 'They are.' The sentence structure should maintain clarity and be specific about locations and actions. The output should incorporate high-frequency words such as: appears, wearing, woman, enjoying, man, view, sitting, group, seems, seem, people, young, standing, time, beautiful, white, closeup, holding, aerial, shirt, video, appear, surrounded, playing, together, peaceful, front, background, focused, using, working, table, good, black, person, serene, others, sky, walking, trees, around, room, city, water, visible, green, blue, captures, camera, something. 
Examples provided:
(1) input: A child plays with toys. output: a young child playing with toys in a room. The child is sitting on the floor surrounded by various toys and appears to be having fun. 
(2) input: A bear explores its surroundings. output: a black bear walking around in a grassy area. The bear appears to be exploring its surroundings and seems to be curious about its environment. 
(3) input: A woman ties a string around an orange. output: a woman sitting at a table and tying a string around an orange. She is wearing a brown robe and appears to be preparing a gift.
(4) input: A doctor performs a procedure on a patient. output: a man wearing a surgical gown and mask standing next to a patient in a hospital room. The man is a doctor who is performing a procedure on the patient. 
(5) input: A monkey looks around. output: a monkey sitting on a tree branch. The monkey appears to be looking around and seems to be curious about its surroundings. 
(6) input: People discuss something at a table. output: a group of people gathered around a table, discussing something together. It appears to be a business meeting or a brainstorming session. The people in the video are engaged in a conversation and seem to be focused on the topic at hand. 
(7) input: A girl holds a cardboard star. output: a young girl wearing a blue dress and holding a cardboard star. She is standing in front of a white background and appears to be smiling. 
(8) input: Young people swim in a pool. output: a group of young people having fun in a swimming pool. They are all wearing swimsuits and enjoying themselves. One of the people in the pool is wearing a bikini. 
(9) input: A woman writes with a pen. output: a close-up shot of a woman's hand holding a pen and writing on a piece of paper. The woman is wearing a ring on her finger and appears to be focused on her work.
The current input: {The_current_input} , Final Output:"""
            ]

            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device).input_ids
            attention_mask = torch.ones(inputs.shape, dtype=torch.long, device=device)
            outputs = model.generate(inputs, max_new_tokens=500, pad_token_id=tokenizer.eos_token_id, attention_mask=attention_mask)
            output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for output in output_text:
                extracted_result = extract_output(output)
                output_data.append({"output": extracted_result})
                with open(output_path, 'a', encoding='utf-8') as txt_file:
                    txt_file.write(extracted_result + '\n')
                    print(f"extracted_result:{extracted_result}")

    print(f"Rewrite outputs saved to {output_path}")
