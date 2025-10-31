import os 
import cv2 as cv 
import torch 
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_path: str, max_length: int, image_size: int,
                 image_filenames, captions, tokenizer, mode="train"
        ):
        self.image_path = image_path
        self.max_length = max_length
        self.image_size = image_size
        
        # These are passed from main.py
        self.image_filenames = image_filenames
        self.captions = list(captions)
        print(f"  [CLIPDataset] Tokenizing {len(self.captions)} captions with max_length={max_length}...")
        self.encoded_captions = tokenizer(
            list(captions), 
            padding="max_length",  # Use "max_length"
            truncation=True, 
            max_length=max_length
        )
        self.transforms = self.get_transforms(mode=mode)
        # self.transforms = A.Compose(
        #     [
        #         A.Resize(224, 224, always_apply=True),
        #         A.Normalize(max_pixel_value=255.0, always_apply=True),
        #         ToTensorV2(always_apply=True),
        #     ]
        # )
    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }

        # --- STEP 3: Use the stored 'self.image_path' ---
        img_full_path = os.path.join(self.image_path, self.image_filenames[idx])
        
        image = cv.imread(img_full_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        
        # self.transforms was already created in __init__
        image = self.transforms(image=image)['image'] 
        
        item['image'] = image  # Already a tensor
        item['caption'] = self.captions[idx]

        return item
    def get_transforms(self, mode="train"):
        if mode == "train":
            return A.Compose(
                [
                    # It can now access self.image_size
                    A.Resize(self.image_size, self.image_size, always_apply=True),
                    A.Normalize(max_pixel_value=255.0, always_apply=True),
                    ToTensorV2(always_apply=True), # Don't permute later
                ]
            )
        else:
            return A.Compose(
                [
                    A.Resize(self.image_size, self.image_size, always_apply=True),
                    A.Normalize(max_pixel_value=255.0, always_apply=True),
                    ToTensorV2(always_apply=True),
                ]
            )