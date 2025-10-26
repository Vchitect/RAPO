# -*- coding: utf-8 -*-
import csv
import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video
from pathlib import Path
import numpy as np
import cv2
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

# === VLM dependencies ===
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


# ---------------------------
# VLM-based alignment assessment
# ---------------------------
def misalignment_assessment(
    qwen_vl_path: str,
    video_path: str = "",
    prompt: str = "",
    max_new_tokens: int = 256,
    device: str = "cuda"
):
    """
    Use Qwen2.5-VL to assess how well the video aligns with the text description.
    Return the model's response string.
    """
    # Load model and processor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        qwen_vl_path, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(qwen_vl_path)

    # Evaluation template
    eval_template = f"""
Evaluate how well the video aligns with the given text prompt.
Consider whether the objects, actions, and scene described in the prompt are accurately represented in the video.
Provide a brief explanation and assign an alignment score from 1 (completely misaligned) to 5 (perfectly aligned).
(A) PROMPT: \"\"\"{prompt}\"\"\"
    """

    # Build messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": eval_template},
            ],
        }
    ]

    # Prepare inputs
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    # Generate output
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text_list = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    # Return string (take the first item if it's a list)
    return output_text_list[0] if isinstance(output_text_list, list) and len(output_text_list) > 0 else ""


# ---------------------------
# Wan pipeline and video generation
# ---------------------------
def load_model(model_id: str) -> WanPipeline:
    """
    Load WanPipeline with its VAE.
    """
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    return pipe


def generate_single_video(
    pipe: WanPipeline,
    prompt: str,
    output_file_path: Path,
    negative_prompt: str,
    seed: int = 1423
) -> None:
    """
    Generate a single video and save it to disk.
    """
    generator = torch.Generator(device="cuda").manual_seed(seed)

    print(f"▶️ Generating video: {prompt}")

    output = pipe(
        prompt=prompt.strip(),
        negative_prompt=negative_prompt,
        height=480,
        width=832,
        num_frames=81,
        guidance_scale=5.0,
        generator=generator
    ).frames[0]

    export_to_video(output, str(output_file_path), fps=15)
    print(f"✅ Saved: {output_file_path}")


def extract_optical_flow(video_path: str, sample_interval_sec: float = 0.5) -> list:
    """
    Sample frames from the video and compute mean optical flow between adjacent samples.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    sample_every = int(fps * sample_interval_sec) if fps and fps > 0 else 1

    frames = []
    for i in range(0, frame_count, sample_every):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    cap.release()

    flows = []
    for i in range(len(frames) - 1):
        prev_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, next_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        mean_flow_x = float(np.mean(flow[..., 0]))
        mean_flow_y = float(np.mean(flow[..., 1]))
        flows.append((mean_flow_x, mean_flow_y))
    return flows


# ---------------------------
# Physics consistency + VLM alignment fusion + prompt refinement
# ---------------------------
def evaluate_physical_consistency(
    flows: list,
    pysical_rule: str,
    text_prompt: str,
    instruct_llm_path: str,
    vlm_alignment: str = ""
) -> tuple:
    """
    Physics consistency analysis + VLM alignment assessment fusion + prompt refinement.
    Returns: (mismatch_summary, refined_prompt)
    """
    model = AutoModelForCausalLM.from_pretrained(
        instruct_llm_path,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(instruct_llm_path)

    # Phase 1: Physics plausibility check based on optical flow
    physics_check_prompt = (
        "You are an expert in physics and motion analysis. I am providing you with a prompt for generating a video "
        "and the optical-flow motion statistics extracted from that generated video.\n\n"
        f"Prompt for the video: {text_prompt}\n"
        f"Sequence of average optical flow vectors (x, y) per sample: {flows}\n\n"
        "Task: Judge whether the motion is physically plausible, referencing laws such as inertia, conservation of momentum, "
        "buoyancy, and continuous force application. Provide a concise final conclusion only (no process), e.g., "
        "\"Sudden global reversals without external force violate inertia\" or \"No obvious physical inconsistency\".\n"
        "Examples:\n"
        "Response 1: Objects or liquids have sudden reverse motion between adjacent frames; if there is no external force explanation "
        "(such as secondary collision, bounce), this sudden acceleration does not conform to the law of inertia; "
        "in particular, liquids or debris usually do not have overall reverse flow.\n"
        "Response 2: Based on the extracted optical flow, there are no obvious physical inconsistencies in this video. "
        "The motion is smooth, directional, and realistic in magnitude and trend. There are no sudden reversals of direction or unrealistic oscillations."
    )

    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": physics_check_prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=1024)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    physics_response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Phase 2: Fuse VLM alignment + physics issues -> rewrite prompt
    rewrite_prompt = (
        "You are a prompt engineering expert for diffusion-based text-to-video generation. "
        "Refine the prompt so that the next generated video better matches real-world physics and the intended semantics.\n\n"
        f"Related physical rule to obey: {pysical_rule}\n"
        f"Original prompt: {text_prompt}\n"
        "Detected mismatches:\n"
        f"- Optical-flow-based physics analysis: {physics_response}\n"
        f"- VLM alignment assessment (semantic/temporal/object-action alignment): {vlm_alignment}\n\n"
        "Requirements for the refined prompt:\n"
        "- Describe the expected video content directly; do not mention rules, analysis, or this instruction.\n"
        "- Keep it under 120 words.\n"
        "- Preserve the core intent but explicitly constrain motions, forces, object states, timings, and camera if helpful."
    )

    rewrite_messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": rewrite_prompt}
    ]
    text = tokenizer.apply_chat_template(rewrite_messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=1024)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    refined_prompt = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Summarize mismatches for logging
    mismatch_summary = f"[Physics] {physics_response}  ||  [VLM] {vlm_alignment}"
    return mismatch_summary, refined_prompt


# ---------------------------
# Main workflow: unchanged logic + call VLM assessment
# ---------------------------
if __name__ == "__main__":
    # ==== Centralized checkpoint and path configuration ====
    WAN_MODEL_ID = "/mnt/petrelfs/gaobingjie/pretrained/Wan2.1-T2V-1.3B-Diffusers"  # Wan T2V checkpoint
    INSTRUCT_LLM_PATH = "/mnt/petrelfs/gaobingjie/pretrained/Qwen2.5-7B-Instruct"    # Instruction-tuned LLM for physics/rewrite
    QWEN_VL_PATH = "/mnt/petrelfs/gaobingjie/pretrained/qwen2.5-vl-7B-instruct"      # VLM for alignment assessment

    # Output and data
    OUTPUT_DIR = Path("/mnt/petrelfs/gaobingjie/rapo_plus/dataset/chosen_cases/examples_refined/")
    OUTPUT_LOG = Path("/mnt/petrelfs/gaobingjie/rapo_plus/dataset/chosen_cases/examples_refined/refined_prompts.csv")
    CSV_PATH = Path("/mnt/petrelfs/gaobingjie/rapo_plus/dataset/chosen_cases/examples.csv")

    # Negative prompt (not a checkpoint path)
    NEGATIVE_PROMPT = (
        "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
        "images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, "
        "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
        "misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    )

    # Prepare I/O
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_LOG.parent.mkdir(parents=True, exist_ok=True)

    # Load T2V pipeline
    pipe = load_model(WAN_MODEL_ID)

    # Read input CSV
    df = pd.read_csv(CSV_PATH)

    # Number of refinement iterations per prompt
    num_refine_iterations = 5

    # === Load existing log (if any) for resume capability ===
    # Log columns: base_name, iter_idx, video_file, mismatch, refined_prompt
    if OUTPUT_LOG.exists() and OUTPUT_LOG.stat().st_size > 0:
        log_df = pd.read_csv(OUTPUT_LOG, header=None, names=["base_name", "iter_idx", "video_file", "mismatch", "refined_prompt"])
    else:
        log_df = pd.DataFrame(columns=["base_name", "iter_idx", "video_file", "mismatch", "refined_prompt"])

    # Main loop over rows
    for idx, row in df.iterrows():
        base_name = f"{idx + 1:07d}"
        PHYSICAL_RULE = row['phys_law']
        orig_prompt = row['captions']

        # Recover latest refined prompt if previous iterations exist
        done_rows = log_df[log_df["base_name"] == base_name].sort_values("iter_idx")
        if not done_rows.empty:
            last_iter = int(done_rows["iter_idx"].iloc[-1])
            prompt = str(done_rows["refined_prompt"].iloc[-1])
            start_iter = last_iter + 1
            print(f"\n=== {base_name} has {last_iter} recorded iterations, resuming from {start_iter} ===")
        else:
            prompt = orig_prompt
            start_iter = 1
            print(f"\n=== Processing Row {base_name} ===")

        for i in range(start_iter, num_refine_iterations + 1):
            video_file = OUTPUT_DIR / f"{base_name}_r{i}.mp4"

            # Skip if this iteration already exists in the log (ensure the prompt chain stays consistent)
            if not done_rows.empty and i in set(done_rows["iter_idx"].astype(int).tolist()):
                prompt_i = str(done_rows[done_rows["iter_idx"] == i]["refined_prompt"].iloc[0])
                prompt = prompt_i
                print(f"[ Skip ] {base_name} iteration {i} already in log. Skipping.")
                continue

            # If the video exists but no log entry, evaluate and log directly
            if video_file.exists():
                print(f"[ Found ] Existing video: {video_file}, skipping generation and evaluating directly.")
                try:
                    flows = extract_optical_flow(str(video_file))
                    vlm_text = misalignment_assessment(
                        qwen_vl_path=QWEN_VL_PATH,
                        video_path=str(video_file),
                        prompt=orig_prompt,
                        max_new_tokens=256,
                        device="cuda"
                    )
                    mismatch, refined_prompt = evaluate_physical_consistency(
                        flows, PHYSICAL_RULE, orig_prompt, INSTRUCT_LLM_PATH, vlm_alignment=vlm_text
                    )
                except Exception as e:
                    print(f"[ Warn ] Evaluation failed for existing video: {e}. Skipping this iteration.")
                    continue

                print(f"[ Iter {i} ] Mismatch: {mismatch}")
                print(f"[ Iter {i} ] Refined Prompt: {refined_prompt}")

                with open(OUTPUT_LOG, mode='a', newline='', encoding='utf-8') as log_file:
                    writer = csv.writer(log_file)
                    writer.writerow([base_name, i, str(video_file), mismatch, refined_prompt])

                prompt = refined_prompt  # Use for the next iteration
                continue

            # Normal path: generate -> optical flow -> VLM assess -> fuse -> log
            generate_single_video(pipe, prompt, video_file, NEGATIVE_PROMPT)
            flows = extract_optical_flow(str(video_file))
            vlm_text = misalignment_assessment(
                qwen_vl_path=QWEN_VL_PATH,
                video_path=str(video_file),
                prompt=orig_prompt,
                max_new_tokens=256,
                device="cuda"
            )
            mismatch, refined_prompt = evaluate_physical_consistency(
                flows, PHYSICAL_RULE, orig_prompt, INSTRUCT_LLM_PATH, vlm_alignment=vlm_text
            )

            print(f"[ Iter {i} ] Mismatch: {mismatch}")
            print(f"[ Iter {i} ] Refined Prompt: {refined_prompt}")

            with open(OUTPUT_LOG, mode='a', newline='', encoding='utf-8') as log_file:
                writer = csv.writer(log_file)
                writer.writerow([base_name, i, str(video_file), mismatch, refined_prompt])

            prompt = refined_prompt  # Use for the next iteration
