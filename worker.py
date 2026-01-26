import socket
import threading
import json
import struct
import pickle
import time
import pandas as pd
import numpy as np
import multiprocessing
from datetime import datetime
from itertools import groupby
import cloudpickle

class DistributedWorker:
    def __init__(self, host, port, master_host, master_port):
        self.host = host
        self.port = port
        self.master_host = master_host
        self.master_port = master_port
        
        # Starts empty
        self.mapper_func = None
        self.reducer_func = None
        self.shuffle_func = None
        self.result_func = None
        
        # State
        self.mapped_data = []
        self.received_data = []
        self.peers = []
        self.lock = threading.Lock()

    # ================== SHUFFLE SERVER ==================
    def start_shuffle_server(self):
        # Listen to other workers
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind((self.host, self.port))
        server.listen(10)
        self.is_listening = True
        print(f"[Worker {self.worker_id}] at port {self.port} ready to receive...")

        while self.is_listening:
            try:
                server.settimeout(1.0)
                conn, _ = server.accept()
                
                # Protocol: [4 bytes size] + [CloudPickle Data]
                header = conn.recv(4)
                if header:
                    msg_len = struct.unpack('!I', header)[0]
                    data = b''
                    while len(data) < msg_len:
                        data += conn.recv(msg_len - len(data))
                    
                    received_batch = cloudpickle.loads(data)

                    print(f"[NET] Received package from PEER: {len(received_batch)} elements ({msg_len / 1024:.2f} KB).")

                    with self.lock:
                        self.received_data.extend(received_batch)
                conn.close()
            except socket.timeout:
                continue

    # ================== TASK RECEIVER ==================
    def fetch_task_and_code(self):
        # Conect to master and receives the task
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self.master_host, self.master_port))
        
        # 1. Sends shuffle port to master
        s.sendall(struct.pack('!I', self.port))
        
        # 2. Receives package (Code + Task + Peers)
        header = s.recv(4)
        msg_len = struct.unpack('!I', header)[0]
        data = b''
        while len(data) < msg_len:
            data += s.recv(msg_len - len(data))
            
        payload = cloudpickle.loads(data)
        s.close()
        
        # 3. Install functions
        self.mapper_func = payload['functions']['mapper']
        self.shuffle_func = payload['functions']['shuffle']
        self.reducer_func = payload['functions']['reducer']
        self.result_func = payload['functions']['result']
        
        # Configuration
        self.peers = payload['peers']
        self.task_info = payload['task']
        self.worker_id = payload['worker_id']
        
        print(f"[Worker {self.worker_id}] Task received!")

    # ================== PIPELINE ==================
    def run(self):
        # 1. Run server and search for code
        self.fetch_task_and_code()
        threading.Thread(target=self.start_shuffle_server).start()
        

        # 2. Read data
        overall_start = time.time()
        io_start = time.time()
        df = pd.read_csv(self.task_info['filename'], 
                         skiprows=self.task_info['start_row'], 
                         nrows=self.task_info['num_rows'], 
                         header=None, names=self.task_info['headers'])
        df = df.dropna(subset=['Combined_Key'])
        io_time = time.time() - io_start
        print(f"I/O: Read {len(df)} rows in {io_time:.4f} seconds.")

        # 3. MAP
        print(f"[Worker {self.worker_id}] MAPPING...")
        map_start = time.time()
        for _, row in df.iterrows():
            self.mapped_data.extend(self.mapper_func(row))
        map_time = time.time() - map_start
        print(f"Worker {self.worker_id} MAP: {len(df)} rows -> {len(self.mapped_data)} tuples in {map_time:.4f} seconds.")

        # 4. SHUFFLE (Send to peers)
        print(f"[Worker {self.worker_id}] SHUFFLING...")
        shuffle_start = time.time()
        buckets = self.shuffle_func(self.mapped_data, len(self.peers))
        
        for target_idx, payload in buckets.items():
            if target_idx == self.worker_id:
                self.received_data.extend(payload)
            else:
                target_host, target_port = self.peers[target_idx]
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect((target_host, target_port))
                data_bytes = cloudpickle.dumps(payload)
                s.sendall(struct.pack('!I', len(data_bytes)) + data_bytes)
                s.close()
        shuffle_time = time.time() - shuffle_start
        print(f"Worker {self.worker_id} SHUFFLE and transmission: {shuffle_time:.4f} seconds.")

        # 5. BARRIER (Sync)
        self.sync_with_master()

        # 6. REDUCE 
        print(f"[Worker {self.worker_id}] REDUCING...")
        self.is_listening = False
        reduce_start = time.time() 
        results = self.reducer_func(self.received_data)
        reduce_time = time.time() - reduce_start
        print(f"Worker {self.worker_id} REDUCE: Handled {len(self.received_data)} tuples -> {len(results)} results in {reduce_time:.4f} seconds.")

        # 7. RESULTS
        self.result_func(results, self.worker_id)

        total_time = time.time() - overall_start
        print(f"WORKER {self.worker_id} OVERALL TASK TIME: {total_time:.4f} SECONDS.")

    def sync_with_master(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self.master_host, self.master_port))
        s.sendall(struct.pack('!I', self.worker_id))
        s.recv(2) # Wait for 'GO'
        s.close()

if __name__ == "__main__":
    import sys
    # Ex: python worker.py 6000
    my_port = int(sys.argv[1])
    # Assumes Master at localhost (127.0.0.1) port 5000
    w = DistributedWorker('127.0.0.1', my_port, '127.0.0.1', 5000)
    w.run()