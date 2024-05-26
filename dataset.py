# HandwritingDataset
# Datset class
import pandas as pd
import numpy as np
from skimage import io
import os
from torch.utils.data import Dataset

class CTCData(Dataset):
    """Handwriting dataset Class."""

    def __init__(self, csv_file, root_dir, transform=None, get_char=True, char_dict=None,
                 word_col=-1, in_memory=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            im_memory (bool): Load all images in memory
        """
        assert isinstance(word_col, (int, str))
        self.word_df = pd.read_csv(os.path.join(root_dir, csv_file))
        
        if get_char and char_dict is None:
            chars = []
            self.word_df.iloc[:, word_col].apply(lambda x: chars.extend(list(x)))
            chars = sorted(list(set(chars)))
            self.char_dict = {c:i for i, c in enumerate(chars, 1)}
        else:
            self.char_dict = char_dict
            
        self.root_dir = root_dir
        self.transform = transform
        self.word_col = word_col
        self.max_len = self.word_df.iloc[:, word_col].apply(lambda x: len(x)).max() 
        
        self.in_memory = in_memory
        self.cache = [None] * len(self.word_df)

        if self.in_memory:
            for idx, img_name in zip(self.word_df.index, self.word_df.iloc[:, 0]):
                self.cache[idx] = io.imread(os.path.join(self.root_dir, self.get_folder(img_name), img_name))

    def __len__(self):
        return len(self.word_df)

    def __getitem__(self, idx):
        img_name = self.word_df.iloc[idx, 0]
        
        if self.in_memory:
            image = self.cache[idx]
        else:
            folder_name = self.get_folder(img_name)
            img_filepath = os.path.join(self.root_dir, folder_name, img_name)
            image = io.imread(img_filepath)
        
        if type(self.word_col) == int:
            word = self.word_df.iloc[idx, self.word_col]
        else:
            word = self.word_df[self.word_col].iloc[idx]
            
        sample = {'image': image, 'word': word}

        return self.transform(sample) if self.transform else sample
    
    def get_folder(self, im_nm):
        
        im_nm_split = im_nm.split('-')
        start_folder = im_nm_split[0]
        src_folder = '-'.join(im_nm_split[:2])
        
        return os.path.join(start_folder, src_folder)