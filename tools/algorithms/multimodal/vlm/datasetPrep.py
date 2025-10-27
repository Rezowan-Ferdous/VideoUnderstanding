import os 
import cv2 as cv 
import torch 
import albumentations as A




import conf.data.vlm as config
# import config as cfg

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoTokenizer

class VisionLanguageDataset(Dataset):
    def __init__(self, image_paths, captions, tokenizer_name="bert-base-uncased"):
        self.image_paths = image_paths
        self.captions = captions
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        caption = self.captions[idx]
        tokens = self.tokenizer(caption, return_tensors="pt", padding=True, truncation=True)
        return image, tokens

class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_filenames, captions, tokenizer, dataset, transform=None):
        self.image_filenames = image_filenames
        self.captions = list(captions)
        self.encoded_captions = tokenizer(
            list(captions), padding=True, truncation=True, max_length=config.max_length
        )
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }

        image = cv.imread(f"{config.image_path}/{self.image_filenames[idx]}")
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        item['caption'] = self.captions[idx]

        return item
    
    def get_transforms(mode="train"):
        if mode == "train":
            return A.Compose(
                [
                    A.Resize(config.size, config.size, always_apply=True),
                    A.Normalize(max_pixel_value=255.0, always_apply=True),
                ]
            )
        else:
            return A.Compose(
                [
                    A.Resize(config.size, config.size, always_apply=True),
                    A.Normalize(max_pixel_value=255.0, always_apply=True),
                ]
            )
        
