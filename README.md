# Handwritten Text Recognition (modify)

Short demo of a CTC handwriting model for words and line-level handwriting recognition

Source: [jc639/pytorch-handwritingCTC](https://github.com/jc639/pytorch-handwritingCTC)

[Original document](./README.old.md)

## Change list

- Add a DataSource layer to make it easier to use data from multiple sources or formats.
- Use [githubharald/DeslantImg](https://github.com/githubharald/DeslantImg) for Deslant Algorithm.
- Add type declarations.
- Rename the `CTCData` class to `CTCDataset`.
- `CTCDataset` uses data sources from `DataSource`.
- `CTCDataset` supports ConcatDataset (`CTCConcatDataset`) by using addition operators.

## DataSource to Dataset

You can write custom data access classes. By inheriting the `DataSource` class.

### Example for Parquet file

```py
from htr_crnn_ctc.datasource import ParquetDataSource
from htr_crnn_ctc.dataset import CTCDataset
from htr_crnn_ctc.transforms import Deslant, Rescale, ToRGB, ToTensor, Normalise

import torch

# Parquet file structure, columns 'image', 'text'
# image -- dict[Literal['bytes'], bytes]
# text  -- str

pds = ParquetDataSource(
    file="tmp\\dataset\\IAM-line\\data\\train.parquet", # Parquet file name
    map_columns=None                                    # Column name mapping
)

ds = CTCDataset(
    data_source=pds,
    char_dict=None,
    transform=Compose([
        Deslant(),
        Rescale(
            output_size=(64, 800),
            random_pad=True,
            border_pad=(10, 40), 
            random_rotation=2,
            random_stretch=1.2
        ),
        ToRGB(),
        ToTensor(rgb=True),
        Normalise(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
)
```

### Examples

- [Deslant Image - CSV IAM word dataset](./deslant-csv-iam.ipynb)
- [Deslant Image - Parquet file to In-memory serialized](./deslant-parquet-to-in-memory.ipynb)
- [Concat Dataset](./concat-dataset.ipynb)
- [Line - In-memory](./train-line-in-memory.ipynb)
