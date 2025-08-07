import os
import numpy as np
from glob import glob
from tqdm import tqdm
from glob import glob
import tarfile
import pickle
from io import BytesIO


class TarRandomAccess():
    def __init__(self, data_dir):
        tar_files = sorted(glob(os.path.join(data_dir, '*.tar')))
        if len(tar_files) == 0:
            raise ValueError('No tar files found in data directory: {}'.format(data_dir))
        self.index = {}
        index_cache = os.path.join(data_dir, 'tar_files_index')
        if os.path.exists(index_cache):
            print('Loading tar files index from cache: {}'.format(index_cache))
            with open(index_cache, 'rb') as file:
                self.index = pickle.load(file)
        else:
            for tar_file in tqdm(tar_files, desc='Indexing tar files'):
                with tarfile.open(tar_file) as tar:
                    for info in tar.getmembers():
                        self.index[info.name] = (tar.name, info)
            print('Saving tar files index to cache: {}'.format(index_cache))
            with open(index_cache, 'wb') as file:
                pickle.dump(self.index, file)
        self.file_names = list(self.index.keys())

    def __getitem__(self, file_name):
        d = self.index.get(file_name)
        if d is None:
            return None
        tar_file, tar_member = d
        with tarfile.open(tar_file) as tar:
            data = BytesIO(tar.extractfile(tar_member).read())
        return data


class WebDataset():
    def __init__(self, data_dir, decode_func=None):
        self.tars = TarRandomAccess(data_dir)
        if decode_func is None:
            self.decode_func = self.decode
        else:
            self.decode_func = decode_func
        
        self.index = {}
        self.prefixes = []
        for file_name in self.tars.file_names:
            p = file_name.split('.', 1)
            if len(p) == 2:
                prefix, postfix = p
                if prefix not in self.index:
                    self.index[prefix] = []
                    self.prefixes.append(prefix)
                self.index[prefix].append(postfix)

    def decode(self, data, file_name):
        if file_name.endswith('.npy'):
            data = np.load(data)
        else:
            raise ValueError('Unknown data type for file: {}'.format(file_name))    
        return data
        
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, index):
        if isinstance(index, str):
            prefix = index
        elif isinstance(index, int):
            prefix = self.prefixes[index]
        else:
            raise ValueError('Expecting index to be int or str')
        sample = self.index.get(prefix)
        if sample is None:
            return None
        
        data = {}
        data['__prefix__'] = prefix
        for postfix in sample:
            file_name = prefix + '.' + postfix
            d = self.decode(self.tars[file_name], file_name)
            data[postfix] = d
        return data
