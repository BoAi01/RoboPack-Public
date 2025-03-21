import os
import pdb

import numpy as np
import threading
from tqdm import tqdm


class AutosaveList(object):
    """
    A list that automatically saves itself to disk when it reaches a certain size.
    It uses a lock to ensure that the list is not modified while it is being saved.
    """
    def __init__(self, save_dir, file_prefix, max_size=10000):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        print('autosave list saving to ', save_dir)

        self.file_prefix = file_prefix
        self.max_size = max_size
        self.data = []
        self.total_count = 0
        self.save_count = 0
        self.lock = threading.Lock()
        self.is_close = False

    def append(self, item):
        assert not self.is_close, 'cannot append to a closed AutosaveList'
        with self.lock:
            self.data.append(item)
            self.total_count += 1
            if len(self.data) >= self.max_size:
                data_to_save = self.data  # Get the data to save while holding the lock
                self.data = []  # Reset self.data to an empty array
                self.save(data_to_save)  # Save the data outside the lock

    def save(self, data_to_save):
        np.save(os.path.join(self.save_dir,
                             f'{self.file_prefix}_{self.total_count}_{self.save_count}.npy'),
                data_to_save)
        self.save_count += 1

    def close(self):
        if len(self.data) > 0:
            self.save(self.data)
            self.data = []
            self.is_close = True

    @staticmethod
    def read_all_data(save_dir, file_prefix):
        data_list = []
        file_list = [filename for filename in os.listdir(save_dir) if
                     filename.startswith(file_prefix) and filename.endswith(".npy")]
        file_list.sort(key=lambda x: int(x[:-4].split("_")[2]))  # Sort the file names in ascending order
        for filename in file_list:
            file_path = os.path.join(save_dir, filename)
            try:
                data = np.load(file_path)
            except:
                print(f'reading {file_path} failed, stopping at here')
                break
            data_list.append(data)
        return np.concatenate(data_list, axis=0)

    def __len__(self):
        with self.lock:
            return len(self.data)


if __name__ == '__main__':
    autosave_list = AutosaveList('test_autosave_list', 'test_autosave_list', max_size=100)
    for i in tqdm(range(1000)):
        to_save = np.zeros((300, 100)) + i
        autosave_list.append(to_save)

    data = np.load('test_autosave_list/test_autosave_list_100_0.npy')
    print(data)
    pdb.set_trace()
