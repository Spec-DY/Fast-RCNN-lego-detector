import torch
import torchvision
import numpy as np
import cv2
from PIL import Image
import gradio as gr
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

MODEL_PATH = "output/lego_detector_final.pth"

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")


def load_model(model_path, num_classes=2):
    """
    Load a trained Faster R-CNN model for LEGO detection.

    Args:
        model_path (str): Path to the saved model weights
        num_classes (int): Number of classes (including background)

    Returns:
        torch.nn.Module: The loaded model in evaluation mode
    """

    model = fasterrcnn_resnet50_fpn(pretrained=False)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model


def detect_legos(model, image, confidence_threshold=0.5):
    """
    Detect LEGO pieces in an image.

    Args:
        model (torch.nn.Module): The detection model
        image (PIL.Image or numpy.ndarray): Input image
        confidence_threshold (float): Confidence threshold for detections

    Returns:
        tuple: Filtered boxes, scores, and labels
    """

    transform = transforms.Compose([transforms.ToTensor()])

    if isinstance(image, np.ndarray):
        image_pil = Image.fromarray(image)
    else:
        image_pil = image

    image_tensor = transform(image_pil).to(device)

    with torch.no_grad():
        prediction = model([image_tensor])

    boxes = prediction[0]['boxes'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()

    high_conf_indices = np.where(scores > confidence_threshold)[0]
    filtered_boxes = boxes[high_conf_indices]
    filtered_scores = scores[high_conf_indices]
    filtered_labels = labels[high_conf_indices]

    return filtered_boxes, filtered_scores, filtered_labels


def visualize_detection(image, boxes, scores, labels=None):
    """
    Visualize detection results on an image.

    Args:
        image (PIL.Image or numpy.ndarray): Input image
        boxes (numpy.ndarray): Detected bounding boxes
        scores (numpy.ndarray): Confidence scores for each box
        labels (numpy.ndarray, optional): Class labels for each box

    Returns:
        numpy.ndarray: Image with detection visualizations
    """

    if not isinstance(image, np.ndarray):
        image = np.array(image)

    img_result = image.copy()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(coord) for coord in box]

        cv2.rectangle(img_result, (x1, y1), (x2, y2), (0, 0, 255), 2)

        score_text = f"{scores[i]:.2f}"
        cv2.putText(img_result, score_text, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.putText(img_result, f"Total: {len(boxes)} LEGO piece(s)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return img_result


def process_image(input_image, confidence_threshold):
    """
    Process an image for the Gradio interface.

    Args:
        input_image (PIL.Image): Input image
        confidence_threshold (float): Confidence threshold for detections

    Returns:
        tuple: Processed image, detection summary, and detailed information
    """
    global model
    if 'model' not in globals():
        model_path = MODEL_PATH
        model = load_model(model_path)

    boxes, scores, labels = detect_legos(
        model, input_image, confidence_threshold)

    result_image = visualize_detection(input_image, boxes, scores, labels)

    detection_summary = f"Detected {len(boxes)} LEGO pieces"

    details = ""
    for i, (box, score) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = [int(coord) for coord in box]
        details += f"Piece #{i+1}: Confidence {score:.2f}, Position [{x1}, {y1}, {x2}, {y2}]\n"

    return result_image, detection_summary, details


def create_gradio_interface():
    """
    Create a Gradio interface for the LEGO detector.

    Returns:
        gr.Interface: Gradio interface object
    """

    title = "LEGO Piece Detector"
    description = """
    
    Upload an image containing LEGO pieces, and the model will detect and mark all LEGO pieces.
    Note: This model is trained on images of LEGO pieces generated using Blender, so LEGO pieces in other contexts may not perform well.
    
    - Use the slider to adjust the detection confidence threshold
    - Results will show each detected piece and its confidence score
    
    Powered by a Faster R-CNN deep learning model.
    """

    input_image = gr.Image(type="pil", label="Upload Image")
    confidence_slider = gr.Slider(minimum=0.1, maximum=0.9, value=0.5, step=0.05,
                                  label="Confidence Threshold", info="Adjust this value to filter detection results")

    output_image = gr.Image(type="numpy", label="Detection Results")
    output_summary = gr.Textbox(label="Detection Summary")
    output_details = gr.Textbox(label="Detailed Information", lines=10)

    iface = gr.Interface(
        fn=process_image,
        inputs=[input_image, confidence_slider],
        outputs=[output_image, output_summary, output_details],
        title=title,
        description=description,
        cache_examples=False
    )

    return iface


if __name__ == "__main__":
    iface = create_gradio_interface()
    iface.launch(share=True)
