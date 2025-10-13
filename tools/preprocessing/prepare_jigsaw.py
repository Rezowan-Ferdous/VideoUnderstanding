import os 
import sys
import json 
import yaml

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


from pathlib import Path

import numpy as np
import networkx as nx
from networkx.algorithms.community import louvain_communities

from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from Datasets.preprocessor import DatasetPreprocessor, JIGSAWS_GESTURE_MAP, JIGSAW_INTERACTION_MAP, VideoProcessor, PromptGenerator, DefaultPromptGenerator
import os 
from pathlib import Path
from tqdm import tqdm
import argparse 

verb_ids= ["reaching","positioning","pushing", "transferring", "moving_center","pulling","G6","G7","G8","G9","G10","G11","G12","G13","G14","G15"]

# action_ids= {"G1" :{
#                 "clip_uid": "G1_reaching_needle_rh",
#                 "action":{
#                 "verb": ["reaching"],
#                 "objects": ["needle"],
#                 "descriptions":"reaching for the needle with right hand"},
#           "hands":[
#         {
#           "hand_uid": "JIGSAWS_Knot_Tying_B001_0_rh",
#           "side": "right",  
#           "state": "holding",
#           "contact_object_uid": ["needle"],
#           "bounding_box": [150, 200, 50, 60]
#         },
#         {
#           "hand_uid": "JIGSAWS_Knot_Tying_B001_0_lh",
#           "side": "left", 
#           "state": "inactive",
#           "contact_object_uid": [],
#           "bounding_box": []
#         }
#       ],
#           "objects": [{"object_id": "needle", "type": "tool", "bounding_box": [100, 150, 30, 40]}, {
#             "object_id": "tissue", "type": "tissue", "bounding_box": [200, 250, 100, 120]
#           },],
#           "chirality": ["right"],
#           "description": "reaching for the needle with right hand"
#         },
#         "G2" :{
#           "clip_uid": "positioning_needle_",
#           "objects": ["needle"],
#           "verb": ["positioning"],
#           "hands":[],
#           "chirality": [],
#           "description": "positioning the tip of the needle"
#         },
#         "G3" :{
#           "clip_uid": "pushing_needle_",
#           "objects": ["needle"],
#           "verb": ["positioning"],
#           "hands":[],
#           "chirality": [],
#           "description": "positioning the tip of the needle"
#         ,}

# JIGSAW_INTERACTION_MAP = {
#     "G1": ["needle"],
#     "G2": ["needle"],
#     "G3": ["needle", "tissue"],
#     "G4": ["needle"],
#     "G5": ["needle"],
#     "G6": ["suture"],  
#     "G7": ["suture"],
#     "G8": ["needle"],  
#     "G9": ["suture"],
#     "G10": ["suture"],
#     "G11": ["needle"],
#     "G12": ["needle"],
#     "G13": ["suture"],
#     "G14": ["suture"],
#     "G15": ["suture"]   }

def parse_meta_file(meta_file_path, task_name):
    trials ={}
    try:
        with open(meta_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    print(f"Skipping empty line in {meta_file_path}")
                    continue # skip empty lines
                trial_id = parts[0]
                skill_level_self_proclaimed = parts[1]
                skill_level_GRS = int(parts[2])
                grs_scores_raw = [int(score) for score in parts[3:9]]
                # Map GRS scores to their names based on readme.txt order
                grs_score_names = [
                    "Respect_for_tissue",
                    "Suture_needle_handling", # Note: For Needle_Passing/Knot_Tying, this is Needle Handling
                    "Time_and_motion",
                    "Flow_of_operation",
                    "Overall_performance",
                    "Quality_of_final_product"
                ]
                grs_scores = dict(zip(grs_score_names, grs_scores_raw))

                trials[trial_id] = {
                    "skill_level_self_proclaimed": skill_level_self_proclaimed,
                    "skill_level_GRS": skill_level_GRS,
                    "GRS_scores": grs_scores,
                    "gesture_sequence": [] # Will be populated later
                }
    except FileNotFoundError:
        print(f"Warning: Meta file not found: {meta_file_path}. Skipping {task_name} metadata.")
    except Exception as e:
        print(f"Error parsing meta file {meta_file_path}: {e}")
    return trials


def export_jigsaws_summary(json_data, output_file_path):
    """
    Exports JIGSAWS trial summaries from JSON data to a text file.

    Each line in the output file will contain:
    TrialID\tTask\tSkillLevel_Self\tSkillLevel_GRS\tGRS_Scores_CommaSep\tGestureSequence_CommaSep

    Args:
        json_data (dict): The loaded JIGSAWS JSON data.
        output_file_path (str): The path to the output text file.
    """
    try:
        with open(output_file_path, 'w') as f:
            # Write a header line (optional)
            f.write("TrialID\tTask\tSkillLevel_Self\tSkillLevel_GRS\tGRS_Scores\tGestureSequence\n")

            tasks = json_data.get("tasks", {})
            for task_name, task_info in tasks.items():
                trials = task_info.get("trials", {})
                for trial_id, trial_data in trials.items():
                    # Extract data
                    skill_self = trial_data.get("skill_level_self_proclaimed", "N/A")
                    skill_grs = trial_data.get("skill_level_GRS", "N/A")

                    # Get GRS scores in a specific order
                    grs_scores_dict = trial_data.get("GRS_scores", {})
                    grs_score_names = [
                        "Respect_for_tissue",
                        "Suture_needle_handling", # Note: Name varies by task in readme
                        "Time_and_motion",
                        "Flow_of_operation",
                        "Overall_performance",
                        "Quality_of_final_product"
                    ]
                    # Create a list of score strings, using 'N/A' if missing
                    grs_scores_list = [str(grs_scores_dict.get(name, "N/A")) for name in grs_score_names]
                    grs_scores_str = ",".join(grs_scores_list)

                    # Get gesture sequence tokens
                    gesture_sequence = trial_data.get("gesture_sequence", [])
                    gesture_tokens = [seg.get("gesture_id", "UNK") for seg in gesture_sequence]
                    gesture_seq_str = ",".join(gesture_tokens)

                    # Write the line to the file
                    line = f"{trial_id}\t{task_name}\t{skill_self}\t{skill_grs}\t{grs_scores_str}\t{gesture_seq_str}\n"
                    f.write(line)

        print(f"JIGSAWS summary successfully exported to: {output_file_path}")

    except Exception as e:
        print(f"An error occurred while exporting: {e}")




def parse_annotation_file(annotation_file_path):
    gesture_sequence = []
    try:
        with open(annotation_file_path, 'r') as f:
            for line in f:
                 parts = line.strip().split()
                 if len(parts) == 3:
                     try:
                         start_frame = int(parts[0])
                         end_frame = int(parts[1])
                         gesture_id_raw = parts[2]
                         # Standardize gesture ID format to GXX
                         if gesture_id_raw.startswith('G'):
                             gesture_id = gesture_id_raw
                         else:
                             gesture_id = f"G{gesture_id_raw}"

                         gesture_sequence.append({
                             "start_frame": start_frame,
                             "end_frame": end_frame,
                             "gesture_id": gesture_id
                         })
                     except ValueError:
                         # Handle lines that don't parse correctly (e.g., headers)
                         print(f"Warning: Skipping invalid line in {annotation_file_path}: {line.strip()}")
                         continue
    except FileNotFoundError:
        print(f"Warning: Transcript file not found: {annotation_file_path}")
    except Exception as e:
        print(f"Error parsing transcript file {annotation_file_path}: {e}")
    return gesture_sequence



def find_jigsaw_video_file(video_root,relative_path):
    base_path = Path(video_root) / relative_path
    # check if the path exist 
    if base_path.exists():
        return str(base_path)
    
    parent = base_path.parent
    stem = base_path.stem

    ext = base_path.suffix

    for i in range(1,3):
        variant_path = parent / f"{stem}_capture{i}{ext}"
        if variant_path.exists():
            return str(variant_path)
    return None


def main(args):
    tasks_info = {
        "Needle_Passing": "Needle_Passing",
        "Knot_Tying": "Knot_Tying",
        "Suturing": "Suturing"
    }
    
    video_list = []
    print("Starting JIGSAWS dataset preprocessing...")
    for task_name, task_info in tasks_info.items():
        print(f"Processing task: {task_name}")
        # a. Parse Meta File
        trials_data = parse_meta_file(task_info["meta_file"], task_name)
        # print(trials_data)
        if not trials_data:
            print(f"  No trial data found for {task_name}. Skipping.")
            continue

        # b. Parse Transcript Files
        transcript_folder = task_info["transcript_folder"]
        if os.path.exists(transcript_folder):
            for filename in os.listdir(transcript_folder):
                
                if filename.endswith(".txt"):
                    file,file_ext = filename.split('.')
                    if trials_data[file]:
                        print(f" Processing transcript file: {filename}")
                    trial_id_from_file = os.path.splitext(filename)[0] # e.g., Knot_Tying_B001
                    transcript_path = os.path.join(transcript_folder, filename)
                    print('trial_id_from_file:  Transcript path:',trial_id_from_file, transcript_path)
                    # Check if this trial exists in our meta data
                    if trial_id_from_file in trials_data:
                        gesture_seq = parse_annotation_file(transcript_path)
                        trials_data[trial_id_from_file]["gesture_sequence"] = gesture_seq
                    else:
                        print(f"  Warning: Transcript file {filename} does not correspond to a known trial in meta data. Skipping.")
        else:
            print(f"  Warning: Transcript folder not found: {transcript_folder}")

