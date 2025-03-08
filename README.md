# LEGO Piece Detection Using Faster R-CNN

## Demo
- **HuggingFace**: [https://huggingface.co/spaces/dionjin/Faster_R-CNN_Lego_Detector](https://huggingface.co/spaces/dionjin/Faster_R-CNN_Lego_Detector)

> I recommend testing it with Blender-generated sources instead of real-world pictures, for example [this dataset](https://www.kaggle.com/datasets/marwin1665/synthetic-lego-images-images22). See details at the end of the Results and Discussion section.

## Research Question
This study addresses the question: How effectively can a deep learning model detect and count LEGO pieces in images without distinguishing between piece types? This question could be relevant for automated inventory systems, sorting mechanisms, and computer vision applications in toy manufacturing.

## Methods

### Data Acquisition and Processing
The research utilized a synthetic dataset containing 168,000 LEGO piece images with annotations in PASCAL VOC format. Each image contained one or more LEGO pieces with corresponding bounding box annotations. The original dataset featured 600 unique LEGO parts, but for this study, all pieces were consolidated under a single "lego" class label, simplifying the task from multi-class to binary object detection.

Data validation was performed to ensure annotation integrity. Images with missing or malformed XML annotations were excluded. The parse_voc_xml function extracted width, height, and object annotations from each XML file, with error handling for missing or invalid entries. The is_valid_annotation function filtered out images without valid object annotations.

The dataset was partitioned using the train_test_split function from scikit-learn with a random state of 42 to ensure reproducibility. The split ratio was 70% training, 15% validation, and 15% testing, creating statistically similar distributions across all sets. To manage computational constraints, a DATASET_LIMIT parameter capped the number of processed images at 10,000.

### Model Architecture
A Faster R-CNN architecture with ResNet-50 backbone was selected for its established effectiveness in object detection tasks. Faster R-CNN operates as a two-stage detector: first generating region proposals, then classifying and refining bounding boxes for these regions.

The model was initialized with weights pre-trained on ImageNet to leverage transfer learning benefits. The final prediction layer was modified to accommodate binary classification (background vs. LEGO):

```python
model = fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
```

### Training Process
Training utilized the following hyperparameters:
- Optimizer: Stochastic Gradient Descent (SGD)
- Momentum: 0.9
- Weight decay: 0.0005
- Initial learning rate: 0.005
- Learning rate scheduler: StepLR with step_size=3, gamma=0.1
- Number of epochs: 10
- Batch size: Dynamic (2-8 based on available GPU memory)

The loss function combined classification loss (cross-entropy) and regression loss (smooth L1) for bounding box coordinates, automatically weighted by the Faster R-CNN implementation:
```python
losses = sum(loss for loss in loss_dict.values())
```

Gradient accumulation was implemented to simulate larger batch sizes on memory-constrained hardware:
```python
(losses / accumulation_steps).backward()
if (i + 1) % accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad()
```

Training dynamics were managed with device-aware processing, allowing computation to fall back to CPU when GPU operations encountered memory limitations or ROI alignment errors.

### Evaluation Metrics
Model performance was quantified using mean Average Precision (mAP) at an Intersection over Union (IoU) threshold of 0.5, a standard metric for object detection tasks.

The IoU is defined as:
```
IoU = Area of Overlap / Area of Union
```

Average Precision (AP) was computed using 11-point interpolation:
```python
ap = 0
for t in np.arange(0, 1.1, 0.1):
    if np.sum(recall >= t) == 0:
        p = 0
    else:
        p = np.max(precision[recall >= t])
    ap += p / 11
```

For inference, a confidence threshold of 0.5 filtered detection results, balancing precision and recall.

## Results and Discussion
The model demonstrated strong performance on the test set, achieving an mAP@0.5 of 0.9089 (90.89%). This high score indicates excellent detection capability within the synthetic dataset environment.

Training loss exhibited consistent decrease across epochs:
- Epoch 1: 0.1414
- Epoch 3: 0.0819
- Epoch 5: 0.0685
- Epoch 7: 0.0594
- Epoch 10: 0.0538

This monotonic decrease suggests successful optimization without overfitting to the training data. However, the validation and test losses were consistently reported as 0.0 across all epochs, which raises concerns about the evaluation implementation. Such values are theoretically impossible for detection models and indicate a potential issue in the loss calculation during evaluation.

Examining the visualization outputs confirms the model's capability to accurately localize LEGO pieces. The bounding boxes closely align with ground truth annotations, and the model successfully detects multiple pieces in complex arrangements.

Several factors likely contributed to the model's high performance. The synthetic nature of the dataset provides consistent lighting, clear object boundaries, and reduced background complexity compared to real-world images. Additionally, the transfer learning approach with ImageNet pre-training provided the model with robust feature extraction capabilities from the start.

However, this study has notable limitations. First, the suspicious validation and test loss values (0.0) suggest implementation issues in the evaluation pipeline that need investigation. Second, the dataset reduction from 168,000 to 10,000 images may have eliminated valuable edge cases that would improve generalization.

The model also lacks ability to distinguish between different LEGO piece types, which might be valuable in certain applications. This limitation was an intentional simplification for the current research focus but represents a clear direction for extension.

Another interesting discovery was that all images in this dataset were Blender-generated. Although they appeared realistic to the human eye, using real-world LEGO photos led to very poor results.

This suggests a complete domain shift: in the machine's view, synthetic images differ markedly from genuine photographs. To explore this suspicion, a different collection of Blender-generated LEGO images from another author was tested, and the model's performance was nearly perfect.

This outcome reinforces the idea that computer-generated worlds and actual photographs can diverge significantly from a model's perspective, even if humans see them as similar.

## Conclusion
This study demonstrates that Faster R-CNN with a ResNet-50 backbone can effectively detect LEGO pieces in synthetic images with high accuracy (90.89% mAP@0.5). The implementation successfully addresses the research question by providing a robust framework for LEGO piece detection and counting.

The project's strengths include adaptive resource utilization, comprehensive error handling, and effective transfer learning implementation. These engineering considerations enable practical deployment in varied computational environments.

However, several limitations warrant acknowledgment. The suspicious evaluation metrics require further investigation to ensure proper model assessment. The synthetic-only training data limits generalization to real-world scenarios. The binary classification approach (LEGO vs. background) represents a simplified version of what could be a more nuanced detection system.

Future research directions should include:
1. Training and evaluating on real-world LEGO images to assess generalization
2. Expanding the model to classify different LEGO piece types in addition to detection
3. Implementing data augmentation to improve robustness to lighting and orientation variations
4. Exploring lightweight models for edge deployment in robotic or mobile applications
5. Resolving the evaluation metric anomalies for more reliable performance assessment

These extensions would address the current limitations while expanding the utility of the LEGO detection system for practical applications in inventory management, automated sorting, and educational robotics.

## Image Resource
[https://www.kaggle.com/datasets/dreamfactor/biggest-lego-dataset-600-parts](https://www.kaggle.com/datasets/dreamfactor/biggest-lego-dataset-600-parts)
