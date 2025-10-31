import os 
import cv2 as cv 
import torch 
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import hydra
from omegaconf import DictConfig


import os
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
import hydra
from omegaconf import DictConfig

# class CLIPDataset(torch.utils.data.Dataset):
#     def __init__(self, image_path: str, max_length: int, image_size: int,
#                  image_filenames, captions, tokenizer, mode="train"
#         ):
#         self.image_path = image_path
#         self.max_length = max_length
#         self.image_size = image_size
        
#         # These are passed from main.py
#         self.image_filenames = image_filenames
#         self.captions = list(captions)
#         print(f"  [CLIPDataset] Tokenizing {len(self.captions)} captions with max_length={max_length}...")
#         self.encoded_captions = tokenizer(
#             list(captions), 
#             padding="max_length",  # Use "max_length"
#             truncation=True, 
#             max_length=max_length
#         )
#         self.transforms = self.get_transforms(mode=mode)
#         # self.transforms = A.Compose(
#         #     [
#         #         A.Resize(224, 224, always_apply=True),
#         #         A.Normalize(max_pixel_value=255.0, always_apply=True),
#         #         ToTensorV2(always_apply=True),
#         #     ]
#         # )
#     def __len__(self):
#         return len(self.captions)

#     def __getitem__(self, idx):
#         item = {
#             key: torch.tensor(values[idx])
#             for key, values in self.encoded_captions.items()
#         }

#         # --- STEP 3: Use the stored 'self.image_path' ---
#         img_full_path = os.path.join(self.image_path, self.image_filenames[idx])
        
#         image = cv.imread(img_full_path)
#         image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        
#         # self.transforms was already created in __init__
#         image = self.transforms(image=image)['image'] 
        
#         item['image'] = image  # Already a tensor
#         item['caption'] = self.captions[idx]

#         return item
#     def get_transforms(self, mode="train"):
#         if mode == "train":
#             return A.Compose(
#                 [
#                     # It can now access self.image_size
#                     A.Resize(self.image_size, self.image_size, always_apply=True),
#                     A.Normalize(max_pixel_value=255.0, always_apply=True),
#                     ToTensorV2(always_apply=True), # Don't permute later
#                 ]
#             )
#         else:
#             return A.Compose(
#                 [
#                     A.Resize(self.image_size, self.image_size, always_apply=True),
#                     A.Normalize(max_pixel_value=255.0, always_apply=True),
#                     ToTensorV2(always_apply=True),
#                 ]
#             )
        

# class CLIPDataModule(pl.LightningDataModule):
#     """
#     A LightningDataModule for the CLIPDataset.
#     It handles loading data, train/val splits, and creating DataLoaders.
#     It includes a custom collate_fn to handle string-based image IDs.
#     """
#     def __init__(self,name: str, image_path: str, captions_path: str, 
#                  tokenizer_config: DictConfig,
#                  batch_size: int, num_workers: int, image_size: int,
#                  max_length: int, val_split_pct: float = 0.1, **kwargs):
#         super().__init__()
#         # Save hyperparams
#         self.name = name
#         self.image_path = image_path
#         self.captions_path = captions_path
#         self.tokenizer = hydra.utils.instantiate(tokenizer_config) # Instantiate tokenizer
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.image_size = image_size
#         self.max_length = max_length
#         self.val_split_pct = val_split_pct

#         self.train_df = None
#         self.val_df = None

#     def setup(self, stage: str = None):
#         """
#         Loads the main dataframe and creates train/val splits.
#         """
#         try:
#             full_df = pd.read_csv(self.captions_path)
#         except FileNotFoundError:
#             print(f"\n\nERROR: Captions file not found at {self.captions_path}")
#             print("Please run 'python scripts/data_prep/download_flickr8k.py' first.\n")
#             raise
            
#         # Group by image and split images to keep all captions together
#         image_names = full_df['image'].unique()
#         train_images, val_images = train_test_split(
#             image_names, 
#             test_size=self.val_split_pct,
#             random_state=42 # for reproducibility
#         )
        
#         self.train_df = full_df[full_df['image'].isin(train_images)]
#         self.val_df = full_df[full_df['image'].isin(val_images)]
#         print(f"Data setup complete: {len(self.train_df)} train, {len(self.val_df)} val items.")

#     def _build_dataset(self, df, mode="train"):
#         """Helper to instantiate the CLIPDataset."""
#         return CLIPDataset(
#             image_path=self.image_path,
#             captions_df=df,
#             tokenizer=self.tokenizer,
#             image_size=self.image_size,
#             max_length=self.max_length,
#             mode=mode
#         )

#     def collate_fn(self, batch):
#         """
#         Custom collate function to handle image_file strings.
#         """
#         item_keys = batch[0].keys()
#         collated = {}

#         for key in item_keys:
#             if key in ('image_file', 'caption'):
#                 # Just gather the strings into a list
#                 collated[key] = [item[key] for item in batch]
#             else:
#                 # Stack tensors
#                 collated[key] = torch.stack([item[key] for item in batch])
        
#         return collated

#     def train_dataloader(self):
#         dataset = self._build_dataset(self.train_df, mode="train")
#         return DataLoader(
#             dataset,
#             batch_size=self.batch_size,
#             shuffle=True,
#             num_workers=self.num_workers,
#             pin_memory=True,
#             collate_fn=self.collate_fn # <-- USE THE CUSTOM COLLATE FN
#         )

#     def val_dataloader(self):
#         dataset = self._build_dataset(self.val_df, mode="val")
#         return DataLoader(
#             dataset,
#             batch_size=self.batch_size,
#             shuffle=False,
#             num_workers=self.num_workers,
#             pin_memory=True,
#             collate_fn=self.collate_fn # <-- USE THE CUSTOM COLLATE FN
#         )


class CLIPDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset for loading image-caption pairs.
    Reads from a DataFrame and loads images from a directory.
    """
    def __init__(self, image_path: str, captions_df: pd.DataFrame, tokenizer, 
                 image_size: int, max_length: int, mode="train"):
        """
        Args:
            image_path (str): Path to the directory containing images.
            captions_df (pd.DataFrame): DataFrame with 'image' and 'caption' columns.
            tokenizer: An *already instantiated* Hugging Face tokenizer.
            image_size (int): The size to resize images to (image_size x image_size).
            max_length (int): Max sequence length for the tokenizer.
            mode (str): 'train' or 'val' to determine transforms.
        """
        self.image_path = image_path
        self.captions_df = captions_df.reset_index(drop=True) # Ensure clean integer index
        self.tokenizer = tokenizer # Tokenizer object is passed in
        self.image_size = image_size
        self.max_length = max_length
        self.transforms = self.get_transforms(mode)

        # Pre-tokenize captions
        print(f"Tokenizing {len(self.captions_df)} captions...")
        self.encoded_captions = self.tokenizer(
            self.captions_df['caption'].tolist(),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        print("Tokenization complete.")

    def __len__(self):
        return len(self.captions_df)

    def __getitem__(self, idx):
        # Get pre-tokenized caption
        item = {
            "input_ids": self.encoded_captions["input_ids"][idx],
            "attention_mask": self.encoded_captions["attention_mask"][idx],
        }

        # Load and transform image
        # Use .iloc[idx] for robust integer-based row selection
        image_file = self.captions_df.iloc[idx]["image"] 
        img_full_path = os.path.join(self.image_path, image_file)
        
        try:
            image = cv2.imread(img_full_path)
            if image is None: raise FileNotFoundError()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception:
            # Fallback for corrupt or missing image
            print(f"Warning: Could not read image {img_full_path}. Using blank image.")
            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        image = self.transforms(image=image)['image']
        
        item['image'] = image
        item['caption'] = self.captions_df.iloc[idx]["caption"]
        item['image_file'] = image_file # Pass the filename string
        return item

    def get_transforms(self, mode="train"):
        """Gets the augmentation pipeline for train or val."""
        if mode == "train":
            return A.Compose(
                [
                    A.Resize(self.image_size, self.image_size, always_apply=True),
                    A.Normalize(max_pixel_value=255.0, always_apply=True),
                    ToTensorV2(always_apply=True),
                ]
            )
        else: # val
            return A.Compose(
                [
                    A.Resize(self.image_size, self.image_size, always_apply=True),
                    A.Normalize(max_pixel_value=255.0, always_apply=True),
                    ToTensorV2(always_apply=True),
                ]
            )
        

class CLIPDataModule(pl.LightningDataModule):
    """
    This is the LightningDataModule.
    Hydra will instantiate THIS class.
    """
    def __init__(self, image_path: str, captions_path: str, 
                 tokenizer, # <--- This argument matches the YAML
                 batch_size: int, num_workers: int, image_size: int,
                 max_length: int, val_split_pct: float = 0.1, **kwargs):
        super().__init__()
        self.image_path = image_path
        self.captions_path = captions_path
        
        # --- **THE FIX IS HERE** ---
        # The 'tokenizer' argument is ALREADY an instantiated object
        # because Hydra instantiates nested _target_s first.
        # We just assign it directly.
        self.tokenizer = tokenizer
        # --- **END OF FIX** ---
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.max_length = max_length
        self.val_split_pct = val_split_pct

        self.train_df = None
        self.val_df = None

    def setup(self, stage: str = None):
        try:
            full_df = pd.read_csv(self.captions_path)
        except FileNotFoundError:
            print(f"\n\nERROR: Captions file not found at {self.captions_path}")
            print("Please run 'python scripts/data_prep/download_flickr8k.py' first.\n")
            raise
            
        image_names = full_df['image'].unique()
        train_images, val_images = train_test_split(
            image_names, 
            test_size=self.val_split_pct,
            random_state=42 # for reproducibility
        )
        
        self.train_df = full_df[full_df['image'].isin(train_images)].reset_index(drop=True)
        self.val_df = full_df[full_df['image'].isin(val_images)].reset_index(drop=True)
        print(f"Data setup complete: {len(self.train_df)} train, {len(self.val_df)} val items.")

    def _build_dataset(self, df, mode="train"):
        # This is where we use the CLIPDataset class
        return CLIPDataset(
            image_path=self.image_path,
            captions_df=df,
            tokenizer=self.tokenizer, # Pass the instantiated tokenizer
            image_size=self.image_size,
            max_length=self.max_length,
            mode=mode
        )

    def collate_fn(self, batch):
        """
        Custom collate function to handle image_file strings.
        """
        item_keys = batch[0].keys()
        collated = {}

        for key in item_keys:
            if key in ('image_file', 'caption'):
                collated[key] = [item[key] for item in batch]
            else:
                collated[key] = torch.stack([item[key] for item in batch])
        
        return collated

    def train_dataloader(self):
        dataset = self._build_dataset(self.train_df, mode="train")
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        dataset = self._build_dataset(self.val_df, mode="val")
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn
        )