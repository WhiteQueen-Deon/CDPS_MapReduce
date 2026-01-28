import pandas as pd
from itertools import groupby
from master import MapReduceMaster

# ============================================================
# ROUND 2: Calculate Recommendation Strength
# ============================================================

# Detect number of workers from Round 1 outputs
def detect_num_workers():
    """
    Auto-detect number of workers from Round 1 output files
    """
    import os
    worker_count = 0
    while os.path.exists(f"round1_output_worker_{worker_count}.txt"):
        worker_count += 1
    
    if worker_count == 0:
        print("Error: No Round 1 output files found!")
        print("Please run Round 1 first.")
        exit(1)
    
    print(f"Detected {worker_count} workers from Round 1 outputs")
    return worker_count


# First, merge Round 1 outputs
def merge_round1_outputs(num_workers):
    """
    Merge all Round 1 worker outputs into single file
    """
    merged_file = "round1_merged.csv"
    
    with open(merged_file, 'w') as outf:
        # Write header
        outf.write("user1,user2,common_friend\n")
        
        # Merge worker outputs
        for worker_id in range(num_workers):
            input_file = f"round1_output_worker_{worker_id}.txt"
            try:
                with open(input_file, 'r') as inf:
                    for line in inf:
                        parts = line.strip().split()
                        if len(parts) == 3:
                            outf.write(f"{parts[0]},{parts[1]},{parts[2]}\n")
                print(f"  Merged {input_file}")
            except FileNotFoundError:
                print(f"  Warning: {input_file} not found")
    
    print(f"Merged Round 1 outputs into: {merged_file}")
    return merged_file


NUM_WORKERS = detect_num_workers()
FILENAME = merge_round1_outputs(NUM_WORKERS)


def user_mapper(record):
    """
    Round 2 Map: Convert to counting format
    Input: pandas Series with ['user1', 'user2', 'common_friend']
    Output: [((user1, user2), 1)]
    
    Example:
    Input: user1=323, user2=325, common_friend=21
    Output: [(('323', '325'), 1)]
    """
    user1 = str(int(record['user1']))
    user2 = str(int(record['user2']))
    
    return [((user1, user2), 1)]


def user_shuffler(mapped_data, num_workers):
    """
    Round 2 Shuffle: Partition by friend pair
    
    Same pair goes to same worker for counting
    """
    buckets = {i: [] for i in range(num_workers)}
    
    for key, value in mapped_data:
        # key is tuple (user1, user2)
        pair_str = f"{key[0]}_{key[1]}"
        target = hash(pair_str) % num_workers
        buckets[target].append((key, value))
    
    return buckets


def user_reducer(received_data):
    """
    Round 2 Reduce: Count occurrences of each pair
    Input: [((user1, user2), 1), ((user1, user2), 1), ...]
    Output: [(user1, user2, count), ...]
    
    Example:
    Input: [(('323','325'), 1), (('323','325'), 1), (('323','325'), 1)]
    Output: [('323', '325', 3)]
    
    Meaning: User 323 and 325 have 3 common friends
    """
    sorted_data = sorted(received_data, key=lambda x: x[0])
    results = []
    
    for pair, group in groupby(sorted_data, key=lambda x: x[0]):
        counts = [item[1] for item in group]
        strength = sum(counts)
        
        user1, user2 = pair
        results.append((user1, user2, strength))
    
    return results


def user_result_handler(results, worker_id):
    """
    Save final recommendations with strength
    """
    output_file = f"final_recommendations_worker_{worker_id}.txt"
    
    # Sort by strength descending
    results_sorted = sorted(results, key=lambda x: x[2], reverse=True)
    
    with open(output_file, 'w') as f:
        f.write("# Friend Recommendations\n")
        f.write("# Format: user_a user_b strength\n")
        f.write("# Strength = number of common friends\n\n")
        
        for user1, user2, strength in results_sorted:
            f.write(f"{user1} {user2} {strength}\n")
    
    print(f"[Worker {worker_id}] Round 2 complete. Generated {len(results)} recommendations.")
    print(f"[Worker {worker_id}] Saved to {output_file}")
    
    if results_sorted:
        print(f"\n[Worker {worker_id}] Top 10 recommendations:")
        for user1, user2, strength in results_sorted[:10]:
            print(f"  Recommend {user2} to {user1}, strength={strength}")


if __name__ == "__main__":
    master = MapReduceMaster(
        filename=FILENAME,
        num_workers=NUM_WORKERS,
        mapper=user_mapper,
        shuffler=user_shuffler,
        reducer=user_reducer,
        result_handler=user_result_handler
    )
    master.start()
