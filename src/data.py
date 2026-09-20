"""PyTorch Dataset and collate function for ShapeScenes.

Loads the image + word pairs written by generate_data.py and returns
(image tensor, token id tensor) pairs, batched with a collate function
that pads to the longest word in the batch.

Understand Dataset, DataLoader and collate_fn in: 
    - https://medium.com/geekculture/pytorch-datasets-dataloader-samplers-and-the-collat-fn-bbfc7c527cf1
    - https://medium.com/@ajayanandverma/pytorch-dataset-and-dataloader-a89a34c47993
"""

import torch
from torch.utils.data import DataLoader, Dataset

from tokenizer import Tokenizer

tokenizer = Tokenizer()

class ShapeScenesDataset(Dataset):
    """
        It's considered the object to encapsulate a data source 
        and how to access the item in the data source.
    """
    def __init__(self, path):
        data = torch.load(path)

        self.images = data["images"]
        self.words = data["words"]

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        image = self.images[idx]
        word = self.words[idx]

        return image, word


def collate_fn(batch):
    """Merges a list of samples to form a mini-batch of Tensor"""

    images, words = zip(*batch) # images = (C, H, W)

    # Stacks images
    images = torch.stack(images) #  images = (B, C, H, W)

    # Encode and pad words then put them into one batch of tensor
    tokens = tokenizer.batch_encode(words)

    return images, tokens

def create_dataloader(path, batch_size, shuffle=False):
    """This is main vehicle to help us to sample data from our data source (Medium)"""

    dataset = ShapeScenesDataset(path)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )

if __name__ == "__main__":
    train_loader = create_dataloader(
        "data/train.pt",
        batch_size=32,
        shuffle=True
    )

    for images, tokens in train_loader:
        print("Images shape:", images.shape)
        print("Tokens shape:", tokens.shape)
        print("First word tokens:", tokens[0])
        print("First word:", tokenizer.decode(tokens[0]))
        break
