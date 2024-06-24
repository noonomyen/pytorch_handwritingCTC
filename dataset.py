# HandwritingDataset
# Datset class
from typing import Optional, Dict
import pandas as pd
# import numpy as np
from io import BytesIO
from skimage import io
import os
import pickle
from hashlib import md5 as hash_md5
from torch.utils.data import Dataset

class CTCData(Dataset):
    """Handwriting dataset Class."""

    def __init__(self, csv_file, root_dir, transform=None, get_char=True, char_dict=None,
                 parquet=False, in_memory_pretransform=None, in_memory=False,
                 col_map: Optional[Dict[str, str]] = None, in_memory_serialized_cache=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            im_memory (bool): Load all images in memory
        """
        self.from_parquet = parquet
        self.in_memory_serialized_cache = in_memory_serialized_cache
        
        if self.from_parquet:
            self.word_df = pd.read_parquet(os.path.join(root_dir, csv_file))
        else:
            self.word_df = pd.read_csv(os.path.join(root_dir, csv_file))

        if col_map:
            self.word_df.rename(columns=col_map, inplace=True)

        if get_char and char_dict is None:
            chars = []
            self.word_df.loc[:, "word"].apply(lambda x: chars.extend(list(x)))
            chars = sorted(list(set(chars)))
            self.char_dict = {c:i for i, c in enumerate(chars, 1)}
        else:
            self.char_dict = char_dict
            
        self.root_dir = root_dir
        self.transform = transform
        self.max_len = self.word_df.loc[:, "word"].apply(lambda x: len(x)).max() 
        
        self.in_memory = in_memory
        self.size = len(self.word_df)
        self.cache_image = [None] * self.size
        self.cache_word = [None] * self.size

        if self.in_memory:
            req_cache_dump = False
            fn_path: str

            if self.in_memory_serialized_cache:
                with open(os.path.join(root_dir, csv_file), "rb") as f:
                    h = hash_md5()
                    h.update(f.read())
                    fn = csv_file + h.hexdigest() + "-serialized.bin"

                fn_path = os.path.join(root_dir, fn)

                if os.path.isfile(fn_path):
                    self.cache_image, self.cache_word = pickle.load(open(fn_path, "rb"))
                else:
                    req_cache_dump = True

            if self.from_parquet:
                if not self.in_memory_serialized_cache or (self.in_memory_serialized_cache and req_cache_dump):
                    for idx, bytes_ in zip(self.word_df.index, self.word_df.loc[:, "image"]):
                        img = io.imread(BytesIO(bytes_["bytes"]))
                        self.cache_image[idx]
                        if in_memory_pretransform:
                            tmp = in_memory_pretransform({ "image": img, "word": self.word_df["word"].iloc[idx] })
                            self.cache_image[idx] = tmp["image"]
                            self.cache_word[idx] = tmp["word"]
                        else:
                            self.cache_image[idx] = img
                            self.cache_word[idx] = self.word_df["word"].iloc[idx]
            elif not self.in_memory_serialized_cache or (self.in_memory_serialized_cache and req_cache_dump):
                for idx, img_name in zip(self.word_df.index, self.word_df.loc[:, "file"]):
                    img = io.imread(os.path.join(self.root_dir, self.get_folder(img_name), img_name))
                    if in_memory_pretransform:
                        tmp = in_memory_pretransform({ "image": img, "word": self.word["word"].iloc[idx] })
                        self.cache_image[idx] = tmp["image"]
                        self.cache_word[idx] = tmp["word"]
                    else:
                        self.cache_image[idx] = img
                        self.cache_word[idx] = self.word_df["word"].iloc[idx]

            if req_cache_dump:
                pickle.dump((self.cache_image, self.cache_word), open(fn_path, "wb"), pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.word_df)

    def __getitem__(self, idx):
        if self.in_memory:
            image = self.cache_image[idx]
            word = self.cache_word[idx]
        elif self.from_parquet:
            image = io.imread(BytesIO(self.word_df["image"].iloc[idx]))
            word = self.word_df["word"].iloc[idx]
        else:
            img_name = self.word_df["file"].iloc[idx]
            folder_name = self.get_folder(img_name)
            img_filepath = os.path.join(self.root_dir, folder_name, img_name)
            image = io.imread(img_filepath)
            word = self.word_df["word"].iloc[idx]
            
        sample = {'image': image, 'word': word}

        return self.transform(sample) if self.transform else sample
    
    def get_folder(self, im_nm):
        
        im_nm_split = im_nm.split('-')
        start_folder = im_nm_split[0]
        src_folder = '-'.join(im_nm_split[:2])
        
        return os.path.join(start_folder, src_folder)