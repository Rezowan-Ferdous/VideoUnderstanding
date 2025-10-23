import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import xml.etree.ElementTree as ET
from tqdm import tqdm



class CustomToCOCO:
    """Base converter class to COCO format"""
    
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        category_names: List[str],
        dataset_name: str = "custom_dataset"
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.category_names = category_names
        self.dataset_name = dataset_name
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize COCO structure
        self.coco_data = {
            "info": self._create_info(),
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": self._create_categories()
        }
        
        self.image_id = 1
        self.annotation_id = 1
        
    def _create_info(self) -> Dict:
        """Create dataset info"""
        return {
            "description": f"{self.dataset_name} Dataset",
            "url": "",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "",
            "date_created": datetime.now().strftime("%Y/%m/%d")
        }
    
    def _create_categories(self) -> List[Dict]:
        """Create category list"""
        categories = []
        for idx, name in enumerate(self.category_names, 1):
            categories.append({
                "id": idx,
                "name": name,
                "supercategory": "object"
            })
        return categories
    
    def add_image(
        self,
        file_name: str,
        width: int,
        height: int
    ) -> int:
        """Add image to COCO dataset"""
        image_info = {
            "id": self.image_id,
            "file_name": file_name,
            "width": width,
            "height": height,
            "license": 0,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": ""
        }
        self.coco_data["images"].append(image_info)
        current_id = self.image_id
        self.image_id += 1
        return current_id
    
    def add_annotation(
        self,
        image_id: int,
        category_id: int,
        bbox: List[float],
        segmentation: Optional[List] = None,
        area: Optional[float] = None,
        iscrowd: int = 0
    ):
        """Add annotation to COCO dataset"""
        if area is None:
            area = bbox[2] * bbox[3]
        
        annotation = {
            "id": self.annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": bbox,
            "area": area,
            "iscrowd": iscrowd
        }
        
        if segmentation is not None:
            annotation["segmentation"] = segmentation
        
        self.coco_data["annotations"].append(annotation)
        self.annotation_id += 1
    
    def save(self, split: str = "train"):
        """Save COCO format JSON"""
        output_path = self.output_dir / f"instances_{split}.json"
        with open(output_path, 'w') as f:
            json.dump(self.coco_data, f, indent=2)
        print(f"Saved: {output_path}")
    
    def convert(self):
        """Override this method in subclasses"""
        raise NotImplementedError


class YOLOToCOCO(CustomToCOCO):
    """Convert YOLO format to COCO"""
    
    def convert(self, split: str = "train"):
        """Convert YOLO annotations to COCO format"""
        images_dir = self.input_dir / split / "images"
        labels_dir = self.input_dir / split / "labels"
        
        if not images_dir.exists():
            raise ValueError(f"Images directory not found: {images_dir}")
        
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        
        for img_path in tqdm(image_files, desc=f"Converting {split}"):
            # Read image to get dimensions
            img = cv2.imread(str(img_path))
            height, width = img.shape[:2]
            
            # Add image
            image_id = self.add_image(img_path.name, width, height)
            
            # Read YOLO labels
            label_path = labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        # YOLO format: class_id x_center y_center width height (normalized)
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0]) + 1  # COCO is 1-indexed
                            x_center, y_center, w, h = map(float, parts[1:5])
                            
                            # Convert to COCO bbox format (x, y, width, height in pixels)
                            x = (x_center - w/2) * width
                            y = (y_center - h/2) * height
                            bbox_w = w * width
                            bbox_h = h * height
                            
                            self.add_annotation(
                                image_id=image_id,
                                category_id=class_id,
                                bbox=[x, y, bbox_w, bbox_h]
                            )
        
        self.save(split)


class PascalVOCToCOCO(CustomToCOCO):
    """Convert Pascal VOC XML format to COCO"""
    
    def convert(self, split: str = "train"):
        """Convert VOC annotations to COCO format"""
        images_dir = self.input_dir / split / "JPEGImages"
        annotations_dir = self.input_dir / split / "Annotations"
        
        if not annotations_dir.exists():
            raise ValueError(f"Annotations directory not found: {annotations_dir}")
        
        xml_files = list(annotations_dir.glob("*.xml"))
        
        # Create category name to ID mapping
        cat_name_to_id = {cat['name']: cat['id'] for cat in self.coco_data['categories']}
        
        for xml_path in tqdm(xml_files, desc=f"Converting {split}"):
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Get image info
            filename = root.find('filename').text
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            
            # Add image
            image_id = self.add_image(filename, width, height)
            
            # Process objects
            for obj in root.findall('object'):
                category_name = obj.find('name').text
                if category_name not in cat_name_to_id:
                    print(f"Warning: Unknown category '{category_name}' in {xml_path}")
                    continue
                
                category_id = cat_name_to_id[category_name]
                
                # Get bounding box
                bndbox = obj.find('bndbox')
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)
                
                bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
                
                # Check for segmentation polygon
                segmentation = None
                polygon = obj.find('polygon')
                if polygon is not None:
                    points = []
                    for pt in polygon.findall('pt'):
                        x = float(pt.find('x').text)
                        y = float(pt.find('y').text)
                        points.extend([x, y])
                    segmentation = [points]
                
                self.add_annotation(
                    image_id=image_id,
                    category_id=category_id,
                    bbox=bbox,
                    segmentation=segmentation
                )
        
        self.save(split)


class LabelMeToCOCO(CustomToCOCO):
    """Convert LabelMe JSON format to COCO"""
    
    def convert(self, split: str = "train"):
        """Convert LabelMe annotations to COCO format"""
        data_dir = self.input_dir / split
        
        json_files = list(data_dir.glob("*.json"))
        
        # Create category name to ID mapping
        cat_name_to_id = {cat['name']: cat['id'] for cat in self.coco_data['categories']}
        
        for json_path in tqdm(json_files, desc=f"Converting {split}"):
            with open(json_path, 'r') as f:
                labelme_data = json.load(f)
            
            # Get image info
            filename = labelme_data['imagePath']
            height = labelme_data['imageHeight']
            width = labelme_data['imageWidth']
            
            # Add image
            image_id = self.add_image(filename, width, height)
            
            # Process shapes
            for shape in labelme_data['shapes']:
                label = shape['label']
                if label not in cat_name_to_id:
                    print(f"Warning: Unknown category '{label}' in {json_path}")
                    continue
                
                category_id = cat_name_to_id[label]
                points = shape['points']
                
                # Calculate bounding box
                points_np = np.array(points)
                xmin, ymin = points_np.min(axis=0)
                xmax, ymax = points_np.max(axis=0)
                bbox = [float(xmin), float(ymin), float(xmax - xmin), float(ymax - ymin)]
                
                # Create segmentation
                segmentation = [float(coord) for point in points for coord in point]
                
                self.add_annotation(
                    image_id=image_id,
                    category_id=category_id,
                    bbox=bbox,
                    segmentation=[segmentation]
                )
        
        self.save(split)


class CSVToCOCO(CustomToCOCO):
    """Convert CSV format to COCO (for detection only)"""
    
    def convert(self, csv_path: str, images_dir: str, split: str = "train"):
        """
        Convert CSV annotations to COCO format
        
        Expected CSV format:
        filename,width,height,class,xmin,ymin,xmax,ymax
        """
        import pandas as pd
        
        df = pd.read_csv(csv_path)
        images_dir = Path(images_dir)
        
        # Create category name to ID mapping
        cat_name_to_id = {cat['name']: cat['id'] for cat in self.coco_data['categories']}
        
        # Group by filename
        for filename, group in tqdm(df.groupby('filename'), desc=f"Converting {split}"):
            # Get image dimensions
            row = group.iloc[0]
            width = int(row['width'])
            height = int(row['height'])
            
            # Add image
            image_id = self.add_image(filename, width, height)
            
            # Process annotations
            for _, row in group.iterrows():
                category_name = row['class']
                if category_name not in cat_name_to_id:
                    continue
                
                category_id = cat_name_to_id[category_name]
                
                xmin = float(row['xmin'])
                ymin = float(row['ymin'])
                xmax = float(row['xmax'])
                ymax = float(row['ymax'])
                
                bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
                
                self.add_annotation(
                    image_id=image_id,
                    category_id=category_id,
                    bbox=bbox
                )
        
        self.save(split)

def create_yolo_yaml(
    output_dir: str,
    dataset_name: str,
    category_names: List[str],
    train_path: str = "train/images",
    val_path: str = "val/images",
    test_path: str = "test/images"
):
    data_yaml = {
        'path':
            output_dir,}
    

def create_yolo_yaml(
    output_dir: str,
    dataset_name: str,
    category_names: List[str],
    train_path: str = "train/images",
    val_path: str = "val/images",
    test_path: str = "test/images"
):
    """Create YOLO data.yaml file"""
    import yaml
    
    data_yaml = {
        'path': str(Path(output_dir).absolute()),
        'train': train_path,
        'val': val_path,
        'test': test_path,
        'nc': len(category_names),
        'names': category_names
    }
    
    output_path = Path(output_dir) / f"{dataset_name}.yaml"
    with open(output_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"Created YOLO config: {output_path}")
    return output_path


# Example usage
if __name__ == "__main__":
    # Example 1: Convert YOLO to COCO
    converter = YOLOToCOCO(
        input_dir='data/yolo_dataset',
        output_dir='data/coco_dataset',
        category_names=['balloon', 'person']
    )
    converter.convert(split='train')
    converter.convert(split='val')
    
    # Example 2: Convert Pascal VOC to COCO
    converter = PascalVOCToCOCO(
        input_dir='data/voc_dataset',
        output_dir='data/coco_dataset',
        category_names=['balloon']
    )
    converter.convert(split='train')
    
    # Example 3: Convert LabelMe to COCO
    converter = LabelMeToCOCO(
        input_dir='data/labelme_dataset',
        output_dir='data/coco_dataset',
        category_names=['balloon']
    )
    converter.convert(split='train')
    
    # Example 4: Create YOLO config
    create_yolo_yaml(
        output_dir='data/balloon',
        dataset_name='balloon',
        category_names=['balloon']
    )