from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import json
import os
from PIL import Image, ImageDraw, ImageOps
import numpy as np

class ClimbingHoldDataset(Dataset):
    def __init__(self, annotations_dir, images_dir, output_size=(128, 128)):
        """
        annotations_dir: Directory containing multiple JSON annotation files
        images_dir: Directory where images are stored
        transform: Transformations to apply to the image (e.g., resize, normalization)
        output_size: Size to resize cropped images (width, height)
        """
        self.images_dir = images_dir
        self.transform = transforms.Compose([
                                    transforms.Resize((224, 224)),  # Resize to a standard size
                                    transforms.ToTensor(),          # Convert PIL image to PyTorch tensor
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization (ImageNet mean and std)
                                    ])
        self.output_size = output_size

        # Initialize an empty list to store all holds from all JSON files
        self.holds = []

        # Iterate over all JSON files in the annotations directory
        for json_file in os.listdir(annotations_dir):
            if json_file.endswith(".json"):
                json_path = os.path.join(annotations_dir, json_file)

                # Load the JSON file
                with open(json_path, 'r') as file:
                    data = json.load(file)

                # Extract information from the JSON file
                images = {img['id']: img['file_name'] for img in data.get('images', [])}
                annotations = data.get('annotations', [])

                for annotation in annotations:
                    file_name = images.get(annotation.get("image_id"))
                    hold_data = {
                        "image_id": file_name,
                        "type": annotation["attributes"].get("Type"),
                        "route_id": annotation["attributes"].get("Route ID"),
                        "orientation": annotation["attributes"].get("Orientation"),
                        "bbox": annotation.get("bbox"),
                        "segmentation": annotation.get("segmentation"),
                    }
                    # Store each hold as a separate entry
                    self.holds.append(hold_data)

    def __len__(self):
        return len(self.holds)

    def __getitem__(self, idx):
        # Get the hold data for the current index
        hold_data = self.holds[idx]
        image_id = hold_data["image_id"]
        image_path = os.path.join(self.images_dir, image_id)
        image = Image.open(image_path)

        # Apply rotation if required (optional)
        rotated_image = ImageOps.exif_transpose(image)

        # Extract the bounding box (x_min, y_min, width, height)
        bbox = hold_data["bbox"]
        x_min, y_min, width, height = bbox

        # Crop the image using the bounding box
        cropped_image = rotated_image.crop((x_min, y_min, x_min + width, y_min + height))

        # Resize the cropped image to the fixed output size (e.g., 128x128 or 224x224)
        cropped_image = cropped_image.resize(self.output_size)

        # Apply any additional transformations (e.g., normalization) if defined
        if self.transform:
            cropped_image = self.transform(cropped_image)

        # Map labels to indices for classification tasks
        hold = {
            "image": cropped_image,
            "parent_image_id": hold_data["image_id"],
            "type": self._map_type(hold_data["type"]),
            "orientation": self._map_orientation(hold_data["orientation"]),
        }

        return hold

    def _map_type(self, type_label):
        types = ['Jug', 'Sloper', 'Crimp', 'Jib', 'Pinch', 'Pocket', 'Edge']
        return types.index(type_label) if type_label in types else -1

    def _map_orientation(self, orientation_label):
        orientations = ['Up', 'Down', 'Side', 'UpAng', 'DownAng']
        return orientations.index(orientation_label) if orientation_label in orientations else -1
    
    def add_color(self, colors):
        for hold, color in zip(self.holds, colors):
            hold['color'] = color[0]

    def add_pred_indices_from_arrays(self, pred_type_path, pred_orient_path=None):
       
        pred_types = np.load(pred_type_path)
        if len(pred_types) != len(self.holds):
            raise ValueError(
                f"Length mismatch: {len(pred_types)} predictions vs {len(self.holds)} holds"
            )

        for hold, t in zip(self.holds, pred_types):
            hold["pred_type_idx"] = int(t)

        if pred_orient_path is not None:
            pred_orients = np.load(pred_orient_path)
            if len(pred_orients) != len(self.holds):
                raise ValueError(
                    f"Length mismatch: {len(pred_orients)} orient predictions vs {len(self.holds)} holds"
                )

            for hold, o in zip(self.holds, pred_orients):
                hold["pred_orient_idx"] = int(o)


class ClimbingHoldDatasetPred(Dataset):
    def __init__(self, annotations_dir, images_dir, output_size=(128, 128)):
        """
        annotations_dir: Directory containing multiple JSON annotation files
        images_dir: Directory where images are stored
        transform: Transformations to apply to the image (e.g., resize, normalization)
        output_size: Size to resize cropped images (width, height)
        """
        self.images_dir = images_dir
        self.transform = transforms.Compose([
                                    transforms.Resize((224, 224)),  # Resize to a standard size
                                    transforms.ToTensor(),          # Convert PIL image to PyTorch tensor
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization (ImageNet mean and std)
                                    ])
        self.output_size = output_size

        # Initialize an empty list to store all holds from all JSON files
        self.holds = []

        # Iterate over all JSON files in the annotations directory
        for json_file in os.listdir(annotations_dir):
            if json_file.endswith(".json"):
                json_path = os.path.join(annotations_dir, json_file)

                # Load the JSON file
                with open(json_path, 'r') as file:
                    data = json.load(file)

                # Extract information from the JSON file
                images = {img['id']: img['file_name'] for img in data.get('images', [])}
                annotations = data.get('annotations', [])

                for annotation in annotations:
                    file_name = images.get(annotation.get("image_id"))
                    hold_data = {
                        "image_id": file_name,
                        "bbox": annotation.get("bbox"),
                        "segmentation": annotation.get("segmentation"),
                    }
                    # Store each hold as a separate entry
                    self.holds.append(hold_data)

    def __len__(self):
        return len(self.holds)

    def __getitem__(self, idx):
        # Get the hold data for the current index
        hold_data = self.holds[idx]
        image_id = hold_data["image_id"]
        image_path = os.path.join(self.images_dir, image_id)
        image = Image.open(image_path)

        # Apply rotation if required (optional)
        rotated_image = ImageOps.exif_transpose(image)

        # Extract the bounding box (x_min, y_min, width, height)
        bbox = hold_data["bbox"]
        x_min, y_min, width, height = bbox

        # Crop the image using the bounding box
        cropped_image = rotated_image.crop((x_min, y_min, x_min + width, y_min + height))

        # Resize the cropped image to the fixed output size (e.g., 128x128 or 224x224)
        cropped_image = cropped_image.resize(self.output_size)

        # Apply any additional transformations (e.g., normalization) if defined
        if self.transform:
            cropped_image = self.transform(cropped_image)

        # Map labels to indices for classification tasks
        hold = {
            "image": cropped_image,
            "parent_image_id": hold_data["image_id"],
        }

        return hold

    def _map_type(self, type_label):
        types = ['Jug', 'Sloper', 'Crimp', 'Jib', 'Pinch', 'Pocket', 'Edge']
        return types.index(type_label) if type_label in types else -1

    def _map_orientation(self, orientation_label):
        orientations = ['Up', 'Down', 'Side', 'UpAng', 'DownAng']
        return orientations.index(orientation_label) if orientation_label in orientations else -1
    
    def add_color(self, colors):
        for hold, color in zip(self.holds, colors):
            hold['color'] = color[0]

    def add_pred_indices_from_arrays(self, pred_type_path, pred_orient_path=None):
       
        pred_types = np.load(pred_type_path)
        if len(pred_types) != len(self.holds):
            raise ValueError(
                f"Length mismatch: {len(pred_types)} predictions vs {len(self.holds)} holds"
            )

        # attach predicted type indices
        for hold, t in zip(self.holds, pred_types):
            hold["pred_type_idx"] = int(t)

        if pred_orient_path is not None:
            pred_orients = np.load(pred_orient_path)
            if len(pred_orients) != len(self.holds):
                raise ValueError(
                    f"Length mismatch: {len(pred_orients)} orient predictions vs {len(self.holds)} holds"
                )

            # attach predicted orientation indices
            for hold, o in zip(self.holds, pred_orients):
                hold["pred_orient_idx"] = int(o)