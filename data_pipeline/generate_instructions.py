"""
Stage 5 — edit-instruction generation with VideoChat-Flash.

For datasets without textual manipulation instructions, each episode's frames are turned into a
video and VideoChat-Flash describes the manipulation in two turns: first it lists the objects in
the scene, then it describes the full pick-move-and-place operation. The answer is written to
<dataset_name>/EditInstructions/<episode>/edit_instruction.txt, the layout data_processing_v3.py reads.

Extracted from the (commented-out) instruction-generation block of the research data pipeline
(data_processing_v2.py); the model, prompts and generation settings are unchanged.

Usage (from data_pipeline/):
    python3 generate_instructions.py --dataset_name <dataset_name>
"""

import argparse
import os

import cv2
import torch
from natsort import natsorted
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def frames_to_video(frames_dir, output_path, fps=5):
    # Get list of image files and sort them naturally (frame1, frame2, ...)
    frames = [f for f in os.listdir(frames_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    frames = natsorted(frames)

    if not frames:
        raise ValueError("No image frames found in the directory.")

    # Read first frame to get video dimensions
    first_frame = cv2.imread(os.path.join(frames_dir, frames[0]))
    height, width, _ = first_frame.shape

    # Define video writer (mp4, H.264 codec)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Write each frame into the video
    for frame_name in frames:
        frame_path = os.path.join(frames_dir, frame_name)
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Warning: Skipping unreadable frame {frame_name}")
            continue
        out.write(frame)

    out.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_name", default="Test_Data")
    args = parser.parse_args()

    DATA_DIR = os.path.join(BASE_DIR, args.dataset_name)
    IMAGE_DIR = os.path.join(DATA_DIR, "JPEGImages")
    EDIT_INSTRUCTION_DIR = os.path.join(DATA_DIR, "EditInstructions")

    # Setting up the video captioning model
    video_captioning_model_path = 'OpenGVLab/VideoChat-Flash-Qwen2_5-7B_InternVideo2-1B'
    video_captioning_tokenizer = AutoTokenizer.from_pretrained(video_captioning_model_path, trust_remote_code=True)
    video_captioning_model = AutoModel.from_pretrained(video_captioning_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")

    mm_llm_compress = False # use the global compress or not
    if mm_llm_compress:
        video_captioning_model.config.mm_llm_compress = True
        video_captioning_model.config.llm_compress_type = "uniform0_attention"
        video_captioning_model.config.llm_compress_layer_list = [4, 18]
        video_captioning_model.config.llm_image_token_ratio_list = [1, 0.75, 0.25]
    else:
        video_captioning_model.config.mm_llm_compress = False

    max_num_frames = 512
    generation_config = dict(
        do_sample=False,
        temperature=0.0,
        max_new_tokens=1024,
        top_p=0.1,
        num_beams=1
    )

    os.makedirs(EDIT_INSTRUCTION_DIR, exist_ok=True)
    video_directories = os.listdir(IMAGE_DIR)
    for video_dir in tqdm(video_directories, desc="Generating Edit Instructions"):
        video_dir_path = os.path.join(IMAGE_DIR, video_dir)
        if not os.path.isdir(video_dir_path):
            continue

        # Consolidating the frames into a temporary video file
        video_path = os.path.join(EDIT_INSTRUCTION_DIR, 'temp.mp4')
        frames_to_video(video_dir_path, output_path=video_path, fps=5)

        # Generating the edit instruction
        question1 = (
            "List down all the different types of objects shown in this video."
            "The following examples are only for reference to understand the expected detail level and description style.\n\n"
            "Examples:\n"
            "1.\n"
            "  - Red plastic cup: Placed slightly behind and to the right of the metallic spoon.\n"
            "  - Metallic spoon: Positioned near the front-center of the table, angled slightly toward the red cup.\n"
            "  - White ceramic bowl: Located further back and slightly to the left of both the spoon and cup.\n\n"
            "2.\n"
            "  - Small blue cube: Resting between the yellow rubber ball on its left and the black screwdriver on its right.\n"
            "  - Yellow rubber ball: Positioned close to the front edge of the table, just left of the blue cube.\n"
            "  - Black screwdriver: Lying parallel to the table edge, to the right of the blue cube.\n"
            "  - Red circular coaster: Placed a short distance behind the blue cube, aligned roughly with its center.\n\n"
            "3.\n"
            "  - Silver aluminum can: Located behind and slightly to the left of the white paper box.\n"
            "  - White paper box: Centered in the middle of the table, with the can positioned just behind it.\n"
            "  - Marked circular spot: Visible in front of the paper box near the table's center.\n\n"
            "4.\n"
            "  - Green apple: Positioned between the stainless-steel fork on its left and the small orange on its right.\n"
            "  - Small orange: Placed just right of the apple, close to the table's front edge.\n"
            "  - Stainless-steel fork: Lying flat on the table to the left of the apple, handle pointing outward.\n\n"
            "5.\n"
            "  - Rectangular cardboard box: Situated in front of the tall transparent water bottle and slightly to its left.\n"
            "  - Tall transparent water bottle: Standing upright behind the box, aligned near the table's center.\n"
            "  - Folded white towel: Positioned along the back edge of the table, behind both the box and bottle."
        )

        output1, chat_history1 = video_captioning_model.chat(
            video_path=video_path,
            tokenizer=video_captioning_tokenizer,
            user_prompt=question1,
            return_history=True,
            max_num_frames=max_num_frames,
            generation_config=generation_config
        )

        question2 = (
            "Describe the full pick-move-and-place manipulation operation in detail, performed by the robotic arm in this video. Output only the full manipulation operation in detail and no extra text. "
            "If the robotic arm is not able to successfully pick up the intended object, then describe in detail whatever object position changes are caused due to the failed attempt. "
            "Movement direction should be from the viewer's perspective. "
            "Carefully understand which object/objects is/are being manipulated. "
            "You need to be very specific about the type of the object/objects being manipulated. Don't give generic object descriptions. "

            "The following examples are only for reference to understand the expected detail level and description style.\n\n"
            "Examples:\n"
            "1. The robotic arm extends forward toward a red plastic cup positioned slightly behind a metallic spoon and grasps it from the top rim using its gripper. It lifts the cup vertically, moves diagonally to the front-right corner of the workspace, and places the cup beside a white ceramic bowl before retracting to its resting pose.\n"
            "2. The robotic arm reaches toward a small blue cube resting between a yellow rubber ball on its left and a black screwdriver on its right. It grips the cube firmly from above, lifts it upward, then moves backward and slightly left to place it near a red circular coaster, maintaining the cube's upright orientation throughout the movement.\n"
            "3. The robotic arm moves leftward toward a silver aluminum can located behind a white paper box. It tilts its wrist downward, clamps the can around its midsection, and carefully lifts it over the box. It then rotates its end-effector 45 degrees clockwise before lowering the can onto a marked circular spot on the table.\n"
            "4. The robotic arm approaches a green apple situated between a small orange and a stainless-steel fork. As it tries to grasp the apple from above, the gripper slips, causing the apple to roll forward and slightly to the right, coming to rest against the fork's handle. The arm then retracts without completing the pickup.\n"
            "5. The robotic arm slides rightward toward a rectangular cardboard box placed in front of a tall transparent water bottle. It positions its gripper under the box's left edge, lifts the box carefully, shifts it to the back-left corner of the table, and releases it next to a folded white towel before returning to its home position."
        )

        edit_instruction, chat_history2 = video_captioning_model.chat(
            video_path=video_path,
            tokenizer=video_captioning_tokenizer,
            user_prompt=question2,
            chat_history=chat_history1,
            return_history=True,
            max_num_frames=max_num_frames,
            generation_config=generation_config
        )

        # Writing the edit instruction to the required text file
        dest_dir = os.path.join(EDIT_INSTRUCTION_DIR, video_dir)
        os.makedirs(dest_dir, exist_ok=True)
        dest_file_path = os.path.join(dest_dir, "edit_instruction.txt")
        with open(dest_file_path, 'w') as f:
            f.write(edit_instruction)

        # Removing the temporary video file
        os.remove(video_path)


if __name__ == "__main__":
    main()
