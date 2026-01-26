import socket
import struct
import pandas as pd
import time
import cloudpickle
from itertools import groupby


class MapReduceMaster:
    def __init__(self, filename, num_workers, mapper, shuffler, reducer, result_handler):
        self.filename = filename
        self.num_workers = num_workers
        self.functions = {
            'mapper': mapper,
            'shuffle': shuffler,
            'reducer': reducer,
            'result': result_handler
        }
        self.registered_workers = [] # [(IP, Port), ...]

    def calculate_splits(self):
        header = pd.read_csv(self.filename, nrows=0)
        cols = header.columns.tolist()
        with open(self.filename, 'r') as f: lines = sum(1 for _ in f) - 1
        
        chunk = lines // self.num_workers
        tasks = []
        curr = 1
        for i in range(self.num_workers):
            r = chunk + (lines % self.num_workers if i == self.num_workers - 1 else 0)
            tasks.append({'filename': self.filename, 'start_row': curr, 'num_rows': r, 'headers': cols})
            curr += r
        return tasks

    def start(self):
        tasks = self.calculate_splits()
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(('0.0.0.0', 5000)) # Listen at port 5000
        server.listen(self.num_workers)
        
        print(f"[MASTER] Waiting {self.num_workers} workers...")

        # PHASE 1: REGISTRY (Find workers IP/Ports)
        worker_connections = []
        for i in range(self.num_workers):
            conn, addr = server.accept()
            worker_port = struct.unpack('!I', conn.recv(4))[0]
            self.registered_workers.append((addr[0], worker_port))
            worker_connections.append(conn)
            print(f"[MASTER] Worker {i} registered at {addr[0]}:{worker_port}")

        # PHASE 2: Send task + code
        for i, conn in enumerate(worker_connections):
            payload = {
                'worker_id': i,
                'task': tasks[i],
                'peers': self.registered_workers, 
                'functions': self.functions        
            }
            
            data_bytes = cloudpickle.dumps(payload)
            conn.sendall(struct.pack('!I', len(data_bytes)) + data_bytes)
            conn.close()

        print("[MASTER] Tasks and codes sent. Initializing Sync Barrier...")
        
        # PHASE 3: BARRIER
        arrived = 0
        sync_connections = []
        while arrived < self.num_workers:
            conn, _ = server.accept()
            conn.recv(4) 
            sync_connections.append(conn)
            arrived += 1
            print(f"[MASTER] Worker {arrived}/{self.num_workers} ready.")

        for conn in sync_connections:
            conn.sendall(b'GO')
            conn.close()
            
        print("[MASTER] Job Concluded.")

'''
if __name__ == "__main__":
    master = MapReduceMaster("infections_timeseries.csv", num_workers=2)
    master.start()
'''