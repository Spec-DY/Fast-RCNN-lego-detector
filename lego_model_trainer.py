import os
import glob
import random
import xml.etree.ElementTree as ET
import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import cv2
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split


def debug_cuda():
    """
    Debug CUDA setup by printing information about PyTorch, CUDA availability,
    and attempting a simple tensor operation on the GPU.
    """
    print("----- CUDA Debug Information -----")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Python version: {os.sys.version}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDA device count: {torch.cuda.device_count()}")

        # Try CUDA operation
        try:
            a = torch.tensor([1.0, 2.0, 3.0], device='cuda')
            b = torch.tensor([4.0, 5.0, 6.0], device='cuda')
            c = a + b
            print(f"CUDA tensor test: {c} (successful)")
        except Exception as e:
            print(f"CUDA tensor test failed: {e}")

        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode == 0:
                print("NVIDIA driver is properly installed and working")
            else:
                print("nvidia-smi command failed")
        except Exception as e:
            print(f"Error checking NVIDIA driver: {e}")
    else:
        print("CUDA is not available.")

    print("-----------------------------------")


debug_cuda()

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f"Found {gpu_count} GPU(s)!")
    for i in range(gpu_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  Memory: {props.total_memory / 1e9:.2f} GB")
        print(f"  CUDA Capability: {props.major}.{props.minor}")

    # use gpu
    device = torch.device('cuda:0')
    print(f"Using GPU: {device}")
else:
    print("No GPU available, using CPU instead.")
    device = torch.device('cpu')
    print(f"Using CPU: {device}")


DATASET_PATH = 'dataset'
IMAGE_PATH = os.path.join(DATASET_PATH, 'images')
ANNOTATION_PATH = os.path.join(DATASET_PATH, 'annotations')
OUTPUT_PATH = 'output'

os.makedirs(OUTPUT_PATH, exist_ok=True)


def parse_voc_xml(xml_file):
    """
    Parse a Pascal VOC format XML annotation file.

    Args:
        xml_file (str): Path to the XML annotation file

    Returns:
        dict: Dictionary containing image width, height, and object annotations
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        size = root.find('size')
        if size is None:
            print(f"Warning: No size element found in {xml_file}")
            width, height = 300, 300
        else:
            width_elem = size.find('width')
            height_elem = size.find('height')
            if width_elem is None or height_elem is None:
                print(f"Warning: Missing width or height in {xml_file}")
                width, height = 300, 300
            else:
                try:
                    width = int(width_elem.text)
                    height = int(height_elem.text)
                except (ValueError, TypeError):
                    print(f"Warning: Invalid width or height in {xml_file}")
                    width, height = 300, 300

        # obtain all objects
        objects = []
        for obj in root.findall('object'):
            name = 'lego'

            difficult_elem = obj.find('difficult')
            difficult = 0
            if difficult_elem is not None:
                try:
                    difficult = int(difficult_elem.text)
                except (ValueError, TypeError):
                    difficult = 0

            if difficult:
                continue

            bbox = obj.find('bndbox')
            if bbox is None:
                print(
                    f"Warning: No bounding box found for object in {xml_file}")
                continue

            try:
                xmin_elem = bbox.find('xmin')
                ymin_elem = bbox.find('ymin')
                xmax_elem = bbox.find('xmax')
                ymax_elem = bbox.find('ymax')

                if None in (xmin_elem, ymin_elem, xmax_elem, ymax_elem):
                    print(
                        f"Warning: Missing bounding box coordinates in {xml_file}")
                    continue

                xmin = float(xmin_elem.text)
                ymin = float(ymin_elem.text)
                xmax = float(xmax_elem.text)
                ymax = float(ymax_elem.text)
            except (ValueError, TypeError, AttributeError) as e:
                print(
                    f"Warning: Invalid bounding box values in {xml_file}: {e}")
                continue

            if xmin >= xmax or ymin >= ymax:
                print(
                    f"Warning: Invalid box dimensions in {xml_file}: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")
                continue

            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(width, xmax)
            ymax = min(height, ymax)

            # if its too small then skip
            if xmax - xmin < 5 or ymax - ymin < 5:
                print(f"Warning: Box too small after clipping in {xml_file}")
                continue

            objects.append({
                'name': name,
                'bbox': [xmin, ymin, xmax, ymax]
            })

        return {
            'width': width,
            'height': height,
            'objects': objects
        }
    except Exception as e:
        print(f"Error parsing XML file {xml_file}: {e}")

        return {
            'width': 300,
            'height': 300,
            'objects': []
        }


def is_valid_annotation(annotation_data):
    """
    Check if annotation data is valid (filter bad data).

    Args:
        annotation_data (dict): Annotation data parsed from XML

    Returns:
        bool: True if valid, False otherwise
    """
    if len(annotation_data['objects']) == 0:
        return False

    return True


def get_dataset_files(limit=None):
    """
    Collect all valid image and annotation files from the dataset.

    Args:
        limit (int, optional): Maximum number of images to use

    Returns:
        tuple: Lists of valid image paths and annotation paths
    """
    all_images = []

    for ext in ['*.jpg', '*.jpeg', '*.png']:
        all_images.extend(glob.glob(os.path.join(IMAGE_PATH, ext)))
        all_images.extend(glob.glob(os.path.join(IMAGE_PATH, ext.upper())))

    all_annotations = []
    valid_images = []

    print(f"Found {len(all_images)} total images")

    if limit and limit < len(all_images):
        all_images = random.sample(all_images, limit)
        print(f"Sampled {limit} images")

    for img_path in tqdm(all_images, desc="Validating annotations"):
        img_filename = os.path.basename(img_path)
        img_id = os.path.splitext(img_filename)[0]

        xml_path = os.path.join(ANNOTATION_PATH, f"{img_id}.xml")

        if os.path.exists(xml_path):
            try:
                annotation_data = parse_voc_xml(xml_path)
                if is_valid_annotation(annotation_data):
                    valid_images.append(img_path)
                    all_annotations.append(xml_path)
            except Exception as e:
                print(f"Error processing {xml_path}: {e}")
                continue

    print(f"Found {len(valid_images)} valid image-annotation pairs")

    if len(valid_images) == 0:
        print("WARNING: No valid images found. Creating a dummy dataset for testing...")
        dummy_dir = "dummy_dataset"
        os.makedirs(os.path.join(dummy_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(dummy_dir, "annotations"), exist_ok=True)

        for i in range(10):

            img = np.ones((300, 300, 3), dtype=np.uint8) * 255

            for _ in range(3):
                x1, y1 = random.randint(10, 150), random.randint(10, 150)
                x2, y2 = x1 + \
                    random.randint(50, 100), y1 + random.randint(50, 100)
                color = (random.randint(0, 255), random.randint(
                    0, 255), random.randint(0, 255))
                cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

            img_path = os.path.join(dummy_dir, "images", f"dummy_{i}.jpg")
            cv2.imwrite(img_path, img)

            # Create a dummy XML annotation file for the generated dummy image
            xml_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<annotation>
    <folder>dummy</folder>
    <filename>dummy_{i}.jpg</filename>
    <size>
        <width>300</width>
        <height>300</height>
        <depth>3</depth>
    </size>
    <object>
        <name>lego</name>
        <bndbox>
            <xmin>{x1}</xmin>
            <ymin>{y1}</ymin>
            <xmax>{x2}</xmax>
            <ymax>{y2}</ymax>
        </bndbox>
    </object>
</annotation>'''
            xml_path = os.path.join(dummy_dir, "annotations", f"dummy_{i}.xml")
            with open(xml_path, 'w') as f:
                f.write(xml_content)

            valid_images.append(img_path)
            all_annotations.append(xml_path)

        print(
            f"Created dummy dataset with {len(valid_images)} images in {dummy_dir}")

    return valid_images, all_annotations


class LegoDataset(Dataset):
    """
    Custom dataset for LEGO detection.

    Attributes:
        image_paths (list): List of image file paths
        annotation_paths (list): List of annotation file paths
        transform (callable, optional): Optional transform to be applied on a sample
    """

    def __init__(self, image_paths, annotation_paths, transform=None):
        self.image_paths = image_paths
        self.annotation_paths = annotation_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        annotation_path = self.annotation_paths[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # return a virtual img if cannot be loaded
            image = Image.new("RGB", (300, 300), (255, 255, 255))
            target = {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros(0, dtype=torch.int64),
                'image_id': torch.tensor([idx]),
                'area': torch.zeros(0, dtype=torch.float32),
                'iscrowd': torch.zeros(0, dtype=torch.int64)
            }
            if self.transform:
                image = self.transform(image)
            return image, target

        # Parse annotation
        try:
            annotation_data = parse_voc_xml(annotation_path)
        except Exception as e:
            print(f"Error parsing annotation {annotation_path}: {e}")

            target = {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros(0, dtype=torch.int64),
                'image_id': torch.tensor([idx]),
                'area': torch.zeros(0, dtype=torch.float32),
                'iscrowd': torch.zeros(0, dtype=torch.int64)
            }
            if self.transform:
                image = self.transform(image)
            return image, target

        boxes = []
        labels = []

        for obj in annotation_data['objects']:
            bbox = obj['bbox']
            boxes.append(bbox)
            labels.append(1)  # 1 LEGO, 0 background

        # Handle the case with no boxes
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else torch.zeros(0),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
        }

        if self.transform:
            image = self.transform(image)

        return image, target


def collate_fn(batch):
    """
    Custom collate function for DataLoader to handle variable-sized objects.

    Args:
        batch: Batch of (image, target) tuples

    Returns:
        tuple: Tuple of batched images and targets
    """
    return tuple(zip(*batch))


def get_model(num_classes=2):
    """
    Create a Faster R-CNN model with ResNet-50 backbone for object detection.

    Args:
        num_classes (int): Number of classes (including background)

    Returns:
        torch.nn.Module: The initialized model
    """

    model = fasterrcnn_resnet50_fpn(pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def train_model(model, data_loader, optimizer, num_epochs=10, val_loader=None, accumulation_steps=1):
    """
    Train the model with exception-safe handling.

    Args:
        model (torch.nn.Module): The model to train
        data_loader (DataLoader): Training data loader
        optimizer (torch.optim.Optimizer): Optimizer
        num_epochs (int): Number of epochs to train for
        val_loader (DataLoader, optional): Validation data loader
        accumulation_steps (int): Number of steps to accumulate gradients

    Returns:
        dict: Training metrics
    """
    model.to(device)
    print(f"Model moved to {device}")

    metrics = {'train_loss': [], 'val_loss': []}

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        optimizer.zero_grad()

        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for i, (images, targets) in enumerate(progress_bar):
            try:

                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()}
                           for t in targets]

                # Forward pass
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                # Normalize the loss for gradient accumulation
                loss_value = losses.item() / accumulation_steps

                # Backward pass
                (losses / accumulation_steps).backward()

                # Update weights after the specified accumulation steps
                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                # Update progress bar and accumulated loss
                epoch_loss += loss_value
                progress_bar.set_postfix(loss=loss_value)

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"CUDA out of memory. Skipping batch {i}")
                    torch.cuda.empty_cache()
                    continue

                elif "roi_align" in str(e):
                    print(
                        f"Error in ROI Align operation. Switching to CPU for this batch. Error: {e}")
                    try:
                        cpu_images = [img.cpu() for img in images]
                        cpu_targets = [{k: v.cpu() for k, v in t.items()}
                                       for t in targets]

                        model.cpu()
                        loss_dict = model(cpu_images, cpu_targets)
                        losses = sum(loss for loss in loss_dict.values())
                        loss_value = losses.item() / accumulation_steps
                        (losses / accumulation_steps).backward()

                        if (i + 1) % accumulation_steps == 0:
                            optimizer.step()
                            optimizer.zero_grad()
                        epoch_loss += loss_value
                        progress_bar.set_postfix(loss=loss_value)

                        # back to GPU
                        model.to(device)
                    except Exception as inner_e:
                        print(
                            f"Failed to process batch on CPU as well. Skipping batch. Error: {inner_e}")
                        model.to(device)
                        continue
                else:
                    print(f"Runtime error: {e}")
                    continue
            except Exception as e:
                print(f"Error in batch {i}: {e}")
                continue

        if len(data_loader) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        avg_epoch_loss = epoch_loss / len(data_loader)
        metrics['train_loss'].append(avg_epoch_loss)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.4f}")

        # Validation loop
        if val_loader:
            val_loss = evaluate(model, val_loader)
            metrics['val_loss'].append(val_loss)
            print(f"Validation Loss: {val_loss:.4f}")

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_epoch_loss,
        }, os.path.join(OUTPUT_PATH, f'checkpoint_epoch_{epoch+1}.pth'))

    return metrics


def evaluate(model, data_loader):
    """
    Evaluate the model on a validation or test set.

    Args:
        model (torch.nn.Module): The model to evaluate
        data_loader (DataLoader): Data loader for evaluation

    Returns:
        float: Average loss on the dataset
    """
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            try:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()}
                           for t in targets]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                total_loss += losses.item()
                num_batches += 1
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("CUDA out of memory during evaluation. Skipping batch.")
                    torch.cuda.empty_cache()
                    continue
                elif "roi_align" in str(e):
                    print(
                        f"ROI Align error during evaluation. Trying on CPU. Error: {e}")
                    try:
                        cpu_images = [img.cpu() for img in images]
                        cpu_targets = [{k: v.cpu() for k, v in t.items()}
                                       for t in targets]

                        # Temporarily move the model to CPU
                        model_device = next(model.parameters()).device
                        model.cpu()

                        loss_dict = model(cpu_images, cpu_targets)
                        losses = sum(loss for loss in loss_dict.values())
                        total_loss += losses.item()
                        num_batches += 1

                        # Move back to gpu
                        model.to(model_device)
                    except Exception as inner_e:
                        print(
                            f"Failed on CPU too. Skipping batch. Error: {inner_e}")
                        if device.type == 'cuda':
                            model.to(device)
                        continue
                else:
                    print(f"Runtime error during evaluation: {e}")
                    continue
            except Exception as e:
                print(f"Error during evaluation: {e}")
                continue

    return total_loss / max(1, num_batches)


def calculate_map(model, data_loader, iou_threshold=0.5):
    """
    Calculate mAP@0.5.

    Args:
        model (torch.nn.Module): The model to evaluate
        data_loader (DataLoader): Data loader for evaluation
        iou_threshold (float): IoU threshold for correct detection

    Returns:
        float: mAP value at the specified IoU threshold
    """
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Calculating mAP"):
            try:
                images = [img.to(device) for img in images]

                # Get model predictions
                predictions = model(images)

                # Store predictions and targets
                for pred, target in zip(predictions, targets):
                    target_boxes = target['boxes'].cpu().numpy()
                    target_labels = target['labels'].cpu().numpy()

                    pred_boxes = pred['boxes'].cpu().numpy()
                    pred_scores = pred['scores'].cpu().numpy()
                    pred_labels = pred['labels'].cpu().numpy()

                    all_predictions.append({
                        'boxes': pred_boxes,
                        'scores': pred_scores,
                        'labels': pred_labels
                    })

                    all_targets.append({
                        'boxes': target_boxes,
                        'labels': target_labels
                    })
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("CUDA out of memory during mAP calculation. Skipping batch.")
                    torch.cuda.empty_cache()
                    continue
                elif "roi_align" in str(e):
                    print(f"ROI Align error during mAP calculation. Trying on CPU.")
                    try:

                        model_device = next(model.parameters()).device
                        cpu_images = [img.cpu() for img in images]
                        model.cpu()

                        predictions = model(cpu_images)

                        for pred, target in zip(predictions, targets):
                            target_boxes = target['boxes'].cpu().numpy()
                            target_labels = target['labels'].cpu().numpy()

                            pred_boxes = pred['boxes'].cpu().numpy()
                            pred_scores = pred['scores'].cpu().numpy()
                            pred_labels = pred['labels'].cpu().numpy()

                            all_predictions.append({
                                'boxes': pred_boxes,
                                'scores': pred_scores,
                                'labels': pred_labels
                            })

                            all_targets.append({
                                'boxes': target_boxes,
                                'labels': target_labels
                            })

                        model.to(model_device)
                    except Exception as inner_e:
                        print(
                            f"CPU processing failed too. Skipping batch. Error: {inner_e}")

                        if device.type == 'cuda':
                            model.to(device)
                        continue
                else:
                    print(f"Runtime error during mAP calculation: {e}")
                    continue
            except Exception as e:
                print(f"Error during mAP calculation: {e}")
                continue

    ap = calculate_average_precision(
        all_predictions, all_targets, iou_threshold)

    return ap


def calculate_average_precision(predictions, targets, iou_threshold):
    """
    Calculate average precision for object detection.

    Args:
        predictions (list): List of prediction dictionaries
        targets (list): List of target dictionaries
        iou_threshold (float): IoU threshold for considering a detection correct

    Returns:
        float: Average precision value
    """
    # We only care about the LEGO category (label = 1)
    class_id = 1

    # Collect all detections and ground truths
    all_detections = []
    all_ground_truths = []

    for pred, target in zip(predictions, targets):

        indices = np.where(pred['labels'] == class_id)[0]
        boxes = pred['boxes'][indices]
        scores = pred['scores'][indices]

        # Sort by confidence score
        sorted_indices = np.argsort(-scores)
        boxes = boxes[sorted_indices]
        scores = scores[sorted_indices]

        all_detections.append({
            'boxes': boxes,
            'scores': scores
        })

        gt_indices = np.where(target['labels'] == class_id)[0]
        all_ground_truths.append({
            'boxes': target['boxes'][gt_indices],
            'matched': np.zeros(len(gt_indices), dtype=bool)
        })

    # Calculate how many
    total_gt = sum(len(gt['boxes']) for gt in all_ground_truths)

    if total_gt == 0:
        return 0.0

    all_scores = []
    all_tp = []
    all_fp = []

    for img_idx, dets in enumerate(all_detections):
        for det_idx, (box, score) in enumerate(zip(dets['boxes'], dets['scores'])):
            all_scores.append(score)

            gt_boxes = all_ground_truths[img_idx]['boxes']
            gt_matched = all_ground_truths[img_idx]['matched']

            if len(gt_boxes) == 0:
                all_tp.append(0)
                all_fp.append(1)
                continue

            ious = calculate_iou(box, gt_boxes)
            max_iou_idx = np.argmax(ious)
            max_iou = ious[max_iou_idx]

            if max_iou >= iou_threshold and not gt_matched[max_iou_idx]:
                all_tp.append(1)
                all_fp.append(0)
                all_ground_truths[img_idx]['matched'][max_iou_idx] = True
            else:
                all_tp.append(0)
                all_fp.append(1)

    indices = np.argsort(-np.array(all_scores))
    all_tp = np.array(all_tp)[indices]
    all_fp = np.array(all_fp)[indices]

    cum_tp = np.cumsum(all_tp)
    cum_fp = np.cumsum(all_fp)

    precision = cum_tp / (cum_tp + cum_fp)
    recall = cum_tp / total_gt

    ap = compute_ap(precision, recall)

    return ap


def calculate_iou(box, boxes):
    """
    Calculate Intersection over Union between a box and multiple boxes.

    Args:
        box (numpy.ndarray): Single bounding box [x1, y1, x2, y2]
        boxes (numpy.ndarray): Array of bounding boxes [N, 4]

    Returns:
        numpy.ndarray: IoU values for each box in boxes
    """

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    iou = intersection_area / union_area

    return iou


def compute_ap(precision, recall):
    """
    Compute Average Precision using 11-point interpolation.

    Args:
        precision (numpy.ndarray): Precision values at different recalls
        recall (numpy.ndarray): Recall values

    Returns:
        float: Average precision value
    """

    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11

    return ap


def visualize_detection(model, dataset, num_images=5, output_dir='output'):
    """
    Visualize detection results on random images from the dataset.

    Args:
        model (torch.nn.Module): The trained model
        dataset (Dataset): Dataset containing images and annotations
        num_images (int): Number of images to visualize
        output_dir (str): Directory to save visualizations

    Returns:
        None
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    indices = random.sample(range(len(dataset)), min(num_images, len(dataset)))

    with torch.no_grad():
        for i, idx in enumerate(indices):
            try:
                img, target = dataset[idx]
                img_tensor = img.unsqueeze(0).to(device)

                prediction = model(img_tensor)

                img_np = img.permute(1, 2, 0).cpu().numpy()
                img_np = (img_np * 255).astype(np.uint8)
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                for box in target['boxes'].cpu().numpy():
                    cv2.rectangle(img_np,
                                  (int(box[0]), int(box[1])),
                                  (int(box[2]), int(box[3])),
                                  (0, 255, 0), 2)

                boxes = prediction[0]['boxes'].cpu().numpy()
                scores = prediction[0]['scores'].cpu().numpy()

                # Only display predictions with confidence > 0.5
                high_conf_indices = np.where(scores > 0.5)[0]
                for box in boxes[high_conf_indices]:
                    cv2.rectangle(img_np,
                                  (int(box[0]), int(box[1])),
                                  (int(box[2]), int(box[3])),
                                  (0, 0, 255), 2)

                num_gt = len(target['boxes'])
                num_pred = len(high_conf_indices)

                cv2.putText(img_np, f'GT: {num_gt}, Pred: {num_pred}',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                cv2.imwrite(os.path.join(
                    output_dir, f'detection_{i}.jpg'), img_np)
            except RuntimeError as e:
                if "roi_align" in str(e):
                    print(f"ROI Align error during visualization. Trying on CPU.")
                    try:

                        model_device = next(model.parameters()).device
                        model.cpu()
                        img_tensor = img.unsqueeze(0)

                        prediction = model(img_tensor)

                        img_np = img.permute(1, 2, 0).cpu().numpy()
                        img_np = (img_np * 255).astype(np.uint8)
                        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                        for box in target['boxes'].cpu().numpy():
                            cv2.rectangle(img_np,
                                          (int(box[0]), int(box[1])),
                                          (int(box[2]), int(box[3])),
                                          (0, 255, 0), 2)

                        boxes = prediction[0]['boxes'].cpu().numpy()
                        scores = prediction[0]['scores'].cpu().numpy()

                        high_conf_indices = np.where(scores > 0.5)[0]
                        for box in boxes[high_conf_indices]:
                            cv2.rectangle(img_np,
                                          (int(box[0]), int(box[1])),
                                          (int(box[2]), int(box[3])),
                                          (0, 0, 255), 2)

                        num_gt = len(target['boxes'])
                        num_pred = len(high_conf_indices)

                        cv2.putText(img_np, f'GT: {num_gt}, Pred: {num_pred}',
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                        cv2.imwrite(os.path.join(
                            output_dir, f'detection_{i}.jpg'), img_np)

                        model.to(model_device)
                    except Exception as inner_e:
                        print(
                            f"Failed on CPU too. Skipping image. Error: {inner_e}")

                        if device.type == 'cuda':
                            model.to(device)
                else:
                    print(f"Error visualizing detection for index {idx}: {e}")
            except Exception as e:
                print(f"Error visualizing detection for index {idx}: {e}")


def main():

    DATASET_LIMIT = 10000

    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        if gpu_props.total_memory < 4e9:  # 4GB
            BATCH_SIZE = 2
        elif gpu_props.total_memory > 8e9:  # 8GB
            BATCH_SIZE = 8
        else:
            BATCH_SIZE = 4
        print(f"Using GPU, batch size is {BATCH_SIZE}")
        GRADIENT_ACCUMULATION_STEPS = 1
    else:
        BATCH_SIZE = 2
        GRADIENT_ACCUMULATION_STEPS = 4
        print(
            f"Using CPU, batch size is {BATCH_SIZE}, gradient accumulation steps is {GRADIENT_ACCUMULATION_STEPS}")

    NUM_EPOCHS = 10
    LEARNING_RATE = 0.005

    all_image_paths, all_annotation_paths = get_dataset_files(
        limit=DATASET_LIMIT)

    if len(all_image_paths) == 0:
        print("Error: No image paths found. Unable to continue training.")
        return

    # Split data into training, validation, and test sets
    train_img_paths, temp_img_paths, train_anno_paths, temp_anno_paths = train_test_split(
        all_image_paths, all_annotation_paths, test_size=0.3, random_state=42
    )

    val_img_paths, test_img_paths, val_anno_paths, test_anno_paths = train_test_split(
        temp_img_paths, temp_anno_paths, test_size=0.5, random_state=42
    )

    print(f"Training set: {len(train_img_paths)} images")
    print(f"Validation set: {len(val_img_paths)} images")
    print(f"Test set: {len(test_img_paths)} images")

    with open(os.path.join(OUTPUT_PATH, 'data_splits.json'), 'w') as f:
        json.dump({
            'train': train_img_paths,
            'val': val_img_paths,
            'test': test_img_paths
        }, f)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = LegoDataset(
        train_img_paths, train_anno_paths, transform=transform)
    val_dataset = LegoDataset(
        val_img_paths, val_anno_paths, transform=transform)
    test_dataset = LegoDataset(
        test_img_paths, test_anno_paths, transform=transform)

    # Determine number of workers
    num_workers = 0 if os.name == 'nt' else 4

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    model = get_model(num_classes=2)  # Background + LEGO

    model = model.to(device)
    print(f"Model moved to {device}")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=3, gamma=0.1)

    try:

        print("Starting training...")
        metrics = train_model(
            model,
            train_loader,
            optimizer,
            num_epochs=NUM_EPOCHS,
            val_loader=val_loader,
            accumulation_steps=GRADIENT_ACCUMULATION_STEPS
        )

        lr_scheduler.step()

        plt.figure(figsize=(10, 5))
        plt.plot(metrics['train_loss'], label='Training Loss')
        if 'val_loss' in metrics and metrics['val_loss']:
            plt.plot(metrics['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig(os.path.join(OUTPUT_PATH, 'loss_curve.png'))

        print("Evaluating on test set...")
        test_loss = evaluate(model, test_loader)
        print(f"Test Loss: {test_loss:.4f}")

        # mAp
        print("Calculating mAP@0.5...")
        mAP = calculate_map(model, test_loader, iou_threshold=0.5)
        print(f"mAP@0.5: {mAP:.4f}")

        with open(os.path.join(OUTPUT_PATH, 'results.json'), 'w') as f:
            json.dump({
                'test_loss': test_loss,
                'mAP@0.5': mAP,
                'train_loss': metrics['train_loss'],
                'val_loss': metrics.get('val_loss', [])
            }, f)

        # Save final model
        torch.save(model.state_dict(), os.path.join(
            OUTPUT_PATH, 'lego_detector_final.pth'))

        print("Visualizing detection results...")
        visualize_detection(model, test_dataset, num_images=5)

        print("All done!")

    except Exception as e:
        print(f"Error occurred during training: {e}")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(OUTPUT_PATH, 'emergency_checkpoint.pth'))
        raise e


if __name__ == "__main__":
    main()
