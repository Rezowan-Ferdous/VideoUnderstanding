'''
# Cholec80-train	CholecT50-train	16	[1, 2, 4, 5, 13, 15, 18, 22, 23, 25, 26, 27, 31, 35, 36, 40]
# Cholec80-train	CholecT50-val	3	[8, 12, 29]
# Cholec80-train	CholecT50-test	4	[6, 10, 14, 32]
# Cholec80-val	CholecT50-train	3	[43, 47, 48]
# Cholec80-val	CholecT50-test	1	[42]	-
# Cholec80-test	CholecT50-train	12	[49, 52, 56, 57, 60, 62, 65, 66, 68, 70, 75, 79]	-
# Cholec80-test	CholecT50-val	2	[50, 78]	-
# Cholec80-test	CholecT50-test	4	[51, 73, 74, 80]	-
# [1, 2, 4, 5, 6, 8, 10, 12, 13, 14, 15, 18, 22, 23, 25, 26, 27, 29, 31, 32, 35, 36, 40, 42, 43, 47, 48, 49, 50, 51, 52, 56, 57, 60, 62, 65, 66, 68, 70, 73, 74, 75, 78, 79, 80]
'''

import json 
import os 
from itertools import combinations
from collections import defaultdict,Counter 
import statistics
from dataset_mapping import triplet_dict, verb_dict, instrument_dict, target_dict, phase_dict




null_verb_target= [-1,94,95,96,97,98,99]
CHIRALITY_VERB_PAIRS = {
    "0": "10",  # grasp <-> release (release is hypothetical ID 10)
    "11":"12", # pick drop
    "16":"17", # push pull
    "18": "19", # tie <-> untie (from your example, hypothetical IDs)
    "21": "20", # "tighten": "loosen",
    # "4": "20",  # clip <-> unclip (unclip is hypothetical ID 20)
    # "5": "21",  # cut <-> suture (suture is hypothetical ID 21)
}

def find_videos_with_overlapping_actions(data):
    annotations = data.get('annotations', {})
    for frame_id, instances in annotations.items():
        # Count only valid instances (triplet ID is not -1)
        valid_instances = [inst for inst in instances if inst[0] not in null_verb_target]
        if len(valid_instances) > 1:
            print("valid_instances", valid_instances)
            return True
    return False

def find_videos_with_chiral_actions(data, chirality_map):
    # Create a set of all verb IDs that are part of a chiral pair for quick lookup.
    chiral_verb_ids = set(chirality_map.keys()) | set(chirality_map.values())

    annotations = data.get('annotations', {})
    for frame_id, instances in annotations.items():
        for instance in instances:
            # Verb ID is the 8th item (index 7) in the instance vector.
            verb_id = str(instance[7])
            if verb_id in chiral_verb_ids:
                print("verb_id",verb_id)
                return True
    return False

def find_videos_with_bounding_boxes(data):
    annotations = data.get('annotations', {})
    for frame_id, instances in annotations.items():
        for instance in instances:
            # A triplet instance is a vector of 15 items.
            # BBox for instrument: indices 3, 4, 5, 6 (x, y, w, h)
            # BBox for target: indices 10, 11, 12, 13 (x, y, w, h)
            instrument_bbox_coords = instance[3:7]
            target_bbox_coords = instance[10:14]

            # The value is -1.0 for null/absence. Check if any value is not -1.0.
            if any(coord != -1.0 for coord in instrument_bbox_coords):
                print("instrument_bbox_coords",instrument_bbox_coords)
                return True
            if any(coord != -1.0 for coord in target_bbox_coords):
                print("target_bbox_coords",target_bbox_coords)
                return True
    return False


def find_non_concurrent_pairs(directory_path):
    """
    Analyzes the entire dataset to find all pairs of triplets that NEVER
    occur together in the same frame.
    """
    print("\n--- Starting Analysis for Non-Concurrent Triplets ---")
    if not os.path.exists(directory_path):
        print(f"Error: Directory not found at '{directory_path}'")
        return

    all_triplets = set()
    concurrent_pairs = set()
    triplet_map = {}
    master_map_loaded = False

    files_to_process = sorted([f for f in os.listdir(directory_path) if f.endswith('.json')])

    # Step 1: Find all unique triplets and all observed concurrent pairs
    for filename in files_to_process:
        filepath = os.path.join(directory_path, filename)
        with open(filepath, 'r') as f:
            data = json.load(f)

        if not master_map_loaded and 'categories' in data and 'triplet' in data['categories']:
            triplet_map = data['categories']['triplet']
            master_map_loaded = True

        annotations = data.get('annotations', {})
        for frame_id, instances in annotations.items():
            valid_triplets = {
                triplet_map.get(str(inst[0]), f"ID_{inst[0]}")
                for inst in instances if inst[0] not in [-1, 94, 95, 96, 97, 98, 99]
            }

            # Add all found triplets to our master set
            all_triplets.update(valid_triplets)

            # If there's an overlap, record the pairs
            if len(valid_triplets) >= 2:
                for pair in combinations(sorted(list(valid_triplets)), 2):
                    concurrent_pairs.add(pair)

    print(f"Found {len(all_triplets)} unique triplets across the dataset.")
    print(f"Found {len(concurrent_pairs)} unique concurrent pairs.")

    # Step 2: Generate all theoretically possible pairs
    all_possible_pairs = set(combinations(sorted(list(all_triplets)), 2))

    # Step 3: Find the difference
    non_concurrent_pairs = all_possible_pairs - concurrent_pairs

    # Step 4: Print the report
    print("\n--- Report on Non-Concurrent Triplet Pairs ---")
    print(f"Found {len(non_concurrent_pairs)} pairs that NEVER occurred together:\n")
    # if not non_concurrent_pairs:
    #     print("No non-concurrent pairs were found.")
    # else:
    #     # Sort for consistent output
    #     for i, (triplet1, triplet2) in enumerate(sorted(list(non_concurrent_pairs))):
    #         print(f"  {i+1:04d}: '{triplet1}'  ---  '{triplet2}'")
    print("\n--------------------------------------------")


def generate_consolidated_sequences(directory_path, output_folder):
    """
    Analyzes all JSON files and writes each video's action sequence
    to a single line in a consolidated text file.
    """
    if not os.path.exists(directory_path):
        print(f"Error: Directory not found at '{directory_path}'")
        return
    triplet_files = os.path.join(output_folder,'triplets.txt')
    action_files= os.path.join(output_folder,'actionss.txt')
    instrument_files = os.path.join(output_folder,'instruments.txt')
    print(f"--- Generating consolidated sequence file at '{output_folder}' ---")

    # Use a list to hold all the lines before writing to the file
    all_sequence_lines = []
    all_action_lines =[]
    all_instrument_lines =[]


    files_to_process = sorted([f for f in os.listdir(directory_path) if f.endswith('.json')])

    for filename in files_to_process:
        print(f"Processing file: {filename}")
        filepath = os.path.join(directory_path, filename)
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Generate the single-line sequence string for the current video
        sequence_string , actions_str, instrument_str = get_single_video_sequence_string(data)

        # print("actions_str, instrument_str ",actions_str, instrument_str )
        if sequence_string:
            all_sequence_lines.append(sequence_string)
        if actions_str:
            all_action_lines.append(actions_str)
        if instrument_str:
            all_instrument_lines.append(instrument_str)

    # Write all collected sequences to the output file
    with open(triplet_files, 'w') as f:
        for line in all_sequence_lines:
            f.write(line + "\n")
    with open(action_files, 'w') as f:
        for line in all_action_lines:
            f.write(line + "\n")

    with open(instrument_files, 'w') as f:
        for line in all_instrument_lines:
            f.write(line + "\n")

    print(f"Successfully created sequence file with {len(all_sequence_lines)} video sequences in {output_folder}.")


def get_single_video_sequence_string(data):
    """
    Processes data for a single video and returns its action sequence as a
    formatted single-line string with frame segments and durations.
    e.g., "[ActionA](start-end) -> [ActionB](start-end)"
    """
    annotations = data.get('annotations', {})
    if not annotations:
        return ""

    categories = data.get('categories', {})
    instrument_map = categories.get('instrument', {})
    verb_map = categories.get('verb', {})
    target_map = categories.get('target', {})

    if not all([instrument_map, verb_map, target_map]):
        print(f"Warning: Category maps incomplete in file. Skipping.")
        return ""

    sorted_frame_ids = sorted([int(k) for k in annotations.keys()])
    # print("sorted_frame_ids",sorted_frame_ids)
    action_sequence_blocks = []
    action_blocks=[]
    instrument_blocks =[]
    target_blocks =[]
    last_action_set = None
    last_action =None 
    last_target= None
    last_instrument= None
    start_of_sequence_frame = sorted_frame_ids[0]

    for i, frame_id in enumerate(sorted_frame_ids):
        instances = annotations.get(str(frame_id), [])
        current_action_set = set()
        cur_action =set()
        cur_target= set()
        cur_instrument= set()

        for inst in instances:
            if inst[0] == -1: continue

            tool_id, verb_id, target_id = str(inst[1]), str(inst[7]), str(inst[8])
            instrument_name = instrument_map.get(tool_id)
            verb_name = verb_map.get(verb_id)
            target_name = target_map.get(target_id)

            if verb_name and target_name and instrument_name and 'null' not in verb_name and 'null' not in target_name:
                act= f"{verb_name}"

                intrument =f"{instrument_name}"
                action_str = f"{verb_name}_{target_name}_with_{instrument_name}"
                current_action_set.add(action_str)
                cur_action.add(act)
                cur_target.add(target_name)
                cur_instrument.add(intrument)

        if cur_action != last_action:
          if i > 0:
                end_of_sequence_frame = sorted_frame_ids[i-1]

                action_blocks.append({
                    "actions": sorted(list(cur_action)),
                    "instrument": sorted(list(cur_instrument)),
                    "target": sorted(list(cur_target)),
                    "start_frame": start_of_sequence_frame,
                    "end_frame": end_of_sequence_frame
                })



        if cur_instrument != last_instrument:
          if i > 0:
                end_of_sequence_frame = sorted_frame_ids[i-1]
                instrument_blocks.append({
                    "actions": sorted(list(cur_action)),
                    "instrument": sorted(list(cur_instrument)),
                    "target": sorted(list(cur_target)),
                    "start_frame": start_of_sequence_frame,
                    "end_frame": end_of_sequence_frame
                })


        if current_action_set != last_action_set:
            if i > 0:
                end_of_sequence_frame = sorted_frame_ids[i-1]

                action_sequence_blocks.append({
                    "actions": sorted(list(last_action_set)),
                    "start_frame": start_of_sequence_frame,
                    "end_frame": end_of_sequence_frame
                })


                target_blocks.append({
                    "actions": sorted(list(last_action_set)),
                    "instrument": sorted(list(cur_instrument)),
                    "target": sorted(list(cur_target)),
                    "start_frame": start_of_sequence_frame,
                    "end_frame": end_of_sequence_frame
                })
            start_of_sequence_frame = frame_id
            last_action_set = current_action_set
            last_action= cur_action
            last_instrument= cur_instrument
            last_target= cur_target

    # Add the final action sequence block after the loop
    action_sequence_blocks.append({
        "actions": sorted(list(last_action_set)),
        "start_frame": start_of_sequence_frame,
        "end_frame": sorted_frame_ids[-1]
    })

    action_blocks.append({
        "actions": sorted(list(cur_action)),
        "instrument": sorted(list(cur_instrument)),
        "target": sorted(list(cur_target)),
        "start_frame": start_of_sequence_frame,
        "end_frame": end_of_sequence_frame
        })
    instrument_blocks.append({
        "actions": sorted(list(cur_action)),
        "instrument": sorted(list(cur_instrument)),
        "target": sorted(list(cur_target)),
        "start_frame": start_of_sequence_frame,
        "end_frame": end_of_sequence_frame
    })


    # Now, format the sequence of blocks into a single string with durations
    triplet_formatted_blocks = []
    for block in action_sequence_blocks:
        duration_sec = (block['end_frame'] - block['start_frame']) + 1

        if not block['actions']:
            block_str = "preparation"
        else:
            block_str = " AND ".join(block['actions'])

        triplet_formatted_blocks.append(f"[{block_str}]({duration_sec}s)")

    triplet_blocks= " -> ".join(triplet_formatted_blocks)

    action_formatted_block =[]
    for block in action_blocks:
        duration_sec = (block['end_frame'] - block['start_frame']) + 1

        if not block['actions']:
            block_str = "preparation"
        else:
            block_str = " AND ".join(block['actions'])

        action_formatted_block.append(f"[{block_str}]({duration_sec}s)")

    actionblocks= " -> ".join(action_formatted_block)

    instrument_formatted_block =[]
    for block in instrument_blocks:
        duration_sec = (block['end_frame'] - block['start_frame']) + 1

        if not block['actions']:
            block_str = "preparation"
        else:
            block_str = " AND ".join(block['instrument'])

        instrument_formatted_block.append(f"[{block_str}]({duration_sec}s)")

    instrumentblocks= " -> ".join(instrument_formatted_block)

    return triplet_blocks,actionblocks,instrumentblocks

def extract_sequences_with_durations(directory_path):
    all_video_sequences = []
    triplet={}
    master_map_loaded = False

    for filename in sorted(os.listdir(directory_path)):
        if not filename.endswith('.json'): continue
        filepath = os.path.join(directory_path, filename)
        with open(filepath, 'r') as f: data = json.load(f)

        if not master_map_loaded and 'categories' in data and 'triplet' in data['categories']:
            triplet_map = data['categories']['triplet']
            master_map_loaded = True

        annotations = data.get('annotations', {})
        if not annotations or not triplet_map: continue
        sorted_frame_ids = sorted([int(k) for k in annotations.keys()])
        video_sequence, last_action_set, start_frame = [], None, -1

        for i, frame_id in enumerate(sorted_frame_ids):
            instances = annotations.get(str(frame_id), [])
            current_triplets = {
                triplet_map.get(str(inst[0]), f"ID_{inst[0]}")
                for inst in instances if inst[0] not in [-1,94, 95, 96, 97, 98, 99]
            }
            current_action_set = tuple(sorted(list(current_triplets)))

            if current_action_set != last_action_set:
                if last_action_set is not None:
                    duration = (sorted_frame_ids[i-1] - start_frame) + 1
                    video_sequence.append({"actions": last_action_set, "duration": duration})
                start_frame = frame_id
                last_action_set = current_action_set

        if last_action_set is not None:
            duration = (sorted_frame_ids[-1] - start_frame) + 1
            video_sequence.append({"actions": last_action_set, "duration": duration})

        all_video_sequences.append(video_sequence)

    return all_video_sequences

def analyze_and_save_concurrent_segments(all_video_sequences, output_filepath):
    """
    Analyzes sequences to find and quantify continuous segments of concurrent actions,
    calculates preparation time, and writes the full report to a text file.
    """
    concurrent_segment_stats = defaultdict(lambda: {'count': 0, 'durations': []})
    total_preparation_duration = 0

    # Step 1: Aggregate segment data
    for video_sequence in all_video_sequences:
        for block in video_sequence:
            actions = block['actions']
            duration = block['duration']
            
            if len(actions) >= 1:
                concurrent_segment_stats[actions]['count'] += 1
                concurrent_segment_stats[actions]['durations'].append(duration)
            elif len(actions) == 0:
                total_preparation_duration += duration

    # Step 2: Calculate final statistics
    report_data = []
    for group, data in concurrent_segment_stats.items():
        durations = data['durations']
        report_data.append({
            'group': group, 'count': data['count'],
            'total_duration_s': sum(durations),
            'mean_duration_s': statistics.mean(durations),
            'std_dev_s': statistics.stdev(durations) if len(durations) > 1 else 0.0
        })

    # Step 3: Sort the report by total duration
    sorted_report = sorted(report_data, key=lambda x: x['total_duration_s'], reverse=True)

    # Step 4: Write the detailed report to the specified file
    with open(output_filepath, 'w') as f:
        f.write("="*80 + "\n")
        f.write("      CONCURRENT ACTION SEGMENT REPORT (Duration-Based)\n")
        f.write("="*80 + "\n")
        
        if not sorted_report:
            f.write("No concurrent action segments were found in the dataset.\n")
        else:
            f.write(f"Found {len(sorted_report)} unique types of concurrent action segments.\n\n")
            f.write(f"{'Count':<8} | {'Total Time':<12} | {'Avg Time':<12} | {'Concurrent Action Group'}\n")
            f.write("-" * 120 + "\n")

            for item in sorted_report:
                group_str = " AND ".join(item['group'])
                count_str = str(item['count'])
                total_time_str = f"{item['total_duration_s']}s"
                avg_time_str = f"{item['mean_duration_s']:.1f}s"
                f.write(f"{count_str:<8} | {total_time_str:<12} | {avg_time_str:<12} | {group_str}\n")
        
        f.write("-" * 120 + "\n")
        f.write(f"\nTotal time spent in [Preparation] (no valid actions): {total_preparation_duration} seconds.\n")
        f.write("="*80 + "\n")

    print(f"Analysis complete. Report successfully saved to:\n{output_filepath}")








