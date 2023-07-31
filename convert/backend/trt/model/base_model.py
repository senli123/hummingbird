from abc import ABCMeta, abstractmethod

import numpy as np
import struct
import os
class BaseModel(metaclass = ABCMeta):
    @abstractmethod
    def create_engine(self,builder,
                      config, 
                      wts_path,
                      dt,
                      input_blob_name,
                    output_blob_name,
                    max_batch_size, 
                    input_h,
                    input_w):
        pass
    @staticmethod
    def load_weights(wts_path):
        
        print(f"Loading weights: {wts_path}")

        assert os.path.exists(wts_path), 'Unable to load weight file.'

        weight_map = {}
        with open(wts_path, "r") as f:
            lines = [line.strip() for line in f]
        count = int(lines[0])
        assert count == len(lines) - 1
        for i in range(1, count + 1):
            splits = lines[i].split(" ")
            name = splits[0]
            cur_count = int(splits[1])
            assert cur_count + 2 == len(splits)
            values = []
            for j in range(2, len(splits)):
                # hex string to bytes to float
                values.append(struct.unpack(">f", bytes.fromhex(splits[j])))
            weight_map[name] = np.array(values, dtype=np.float32)

        return weight_map

