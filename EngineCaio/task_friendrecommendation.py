import pandas as pd
from itertools import groupby, combinations
from master import MapReduceMaster

FILENAME = "social_network.csv"
NUM_WORKERS = 2

# ============================================================
# ROUND 1: Generate Friend Pairs from Common Friends
# ============================================================

def user_mapper(record):
    """
    Round 1 Map: Transform friendship edge into bidirectional relationships
    Input: pandas Series with columns ['user1', 'user2']
    Output: [(user1, user2), (user2, user1)]
    """
    user1 = str(int(record['user1']))
    user2 = str(int(record['user2']))
    return [(user1, user2), (user2, user1)]


def user_shuffler(mapped_data, num_workers):
    """
    Round 1 Shuffle: Partition by user ID
    """
    buckets = {i: [] for i in range(num_workers)}
    for key, value in mapped_data:
        target = hash(key) % num_workers
        buckets[target].append((key, value))
    return buckets


def user_reducer(received_data):
    """
    Round 1 Reduce: Generate friend pairs with common friend
    Input: [(user, friend), ...]
    Output: [((friend1, friend2), common_user), ...]
    """
    sorted_data = sorted(received_data, key=lambda x: x[0])
    results = []
    
    for user, group in groupby(sorted_data, key=lambda x: x[0]):
        friends = [item[1] for item in group]
        
        if len(friends) < 2:
            continue
        
        # Generate all friend pairs
        for friend1, friend2 in combinations(friends, 2):
            results.append(((friend1, friend2), user))
            results.append(((friend2, friend1), user))
    
    return results


def user_result_handler(results, worker_id):
    """
    Save Round 1 results
    """
    output_file = f"round1_output_worker_{worker_id}.txt"
    
    with open(output_file, 'w') as f:
        for pair, common_friend in results:
            f.write(f"{pair[0]} {pair[1]} {common_friend}\n")
    
    print(f"[Worker {worker_id}] Round 1 complete. Saved {len(results)} pairs to {output_file}")


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
