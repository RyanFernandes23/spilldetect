# YOLO Model Training and Usage Documentation

## Overview
This project demonstrates how to train and use a YOLO object detection model using the Ultralytics YOLO framework. The process includes preparing your dataset, training the model, and running inference locally on your machine.

---

## 1. Environment Setup

- **GPU Requirement:**
  - For best performance, use a machine with an NVIDIA GPU.
  - Verify GPU availability (in Colab):
    ```python
    !nvidia-smi
    ```

- **Install Dependencies:**
  - Install Ultralytics YOLO and other dependencies:
    ```bash
    pip install ultralytics
    ```
  - For GPU-enabled PyTorch (optional, if you have an NVIDIA GPU):
    ```bash
    pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    ```

---

## 2. Dataset Handling

- **Image Labeling:**
  - 44 images were labeled using Label Studio.
  - The annotations were exported in YOLO format for compatibility with Ultralytics YOLO.

- **Prepare Dataset:**
  - Place your labeled images and label files in a folder (e.g., `data.zip`).
  - Use `train_val_split.py` to split your dataset into training and validation sets:
    ```bash
    python train_val_split.py --datapath="<path_to_unzipped_data>" --train_pct=0.8
    ```
  - This creates the required folder structure for YOLO training.

- **Create `data.yaml`:**
  - Example content:
    ```yaml
    path: /content/data
    train: train/images
    val: validation/images
    nc: 1
    names: [spill]
    ```

---

## 3. Training the YOLO Model

- **Run Training:**
  - Example command (from notebook or terminal):
    ```bash
    yolo detect train data=data.yaml model=yolov8l.pt epochs=80 imgsz=640
    ```
  - Training outputs (weights, metrics, plots) are saved in the respective model's output directory.

---

## 4. Evaluation Results

### Primary Model Evaluation (`primary_model/train3/`)
- **mAP50:** 0.568
- **mAP50-95:** 0.236
- ![results.png](primary_model/train3/results.png)
- ![confusion_matrix.png](primary_model/train3/confusion_matrix.png)
- ![confusion_matrix_normalized.png](primary_model/train3/confusion_matrix_normalized.png)
- ![PR_curve.png](primary_model/train3/PR_curve.png)
- ![F1_curve.png](primary_model/train3/F1_curve.png)
- ![P_curve.png](primary_model/train3/P_curve.png)
- ![R_curve.png](primary_model/train3/R_curve.png)
- ![labels_correlogram.jpg](primary_model/train3/labels_correlogram.jpg)
- ![labels.jpg](primary_model/train3/labels.jpg)
- ![val_batch0_pred.jpg](primary_model/train3/val_batch0_pred.jpg)
- ![val_batch0_labels.jpg](primary_model/train3/val_batch0_labels.jpg)
- ![train_batch0.jpg](primary_model/train3/train_batch0.jpg)
- ![train_batch1.jpg](primary_model/train3/train_batch1.jpg)

### Lightweight Model Evaluation (`lightweight_model/train2/`)
- **mAP50:** 0.67
- **mAP50-95:** 0.227
- ![results.png](lightweight_model/train2/results.png)
- ![confusion_matrix.png](lightweight_model/train2/confusion_matrix.png)
- ![confusion_matrix_normalized.png](lightweight_model/train2/confusion_matrix_normalized.png)
- ![PR_curve.png](lightweight_model/train2/PR_curve.png)
- ![F1_curve.png](lightweight_model/train2/F1_curve.png)
- ![P_curve.png](lightweight_model/train2/P_curve.png)
- ![R_curve.png](lightweight_model/train2/R_curve.png)
- ![labels_correlogram.jpg](lightweight_model/train2/labels_correlogram.jpg)
- ![labels.jpg](lightweight_model/train2/labels.jpg)
- ![val_batch0_pred.jpg](lightweight_model/train2/val_batch0_pred.jpg)
- ![val_batch0_labels.jpg](lightweight_model/train2/val_batch0_labels.jpg)
- ![train_batch0.jpg](lightweight_model/train2/train_batch0.jpg)
- ![train_batch1.jpg](lightweight_model/train2/train_batch1.jpg)

---

## 5. Using the Model Locally

- **Extract Model Weights:**
  - Unzip `primary_model.zip` or `lightweight_model.zip`.

- **Run Inference:**
  - Use the provided `yolo_detect.py` script:
    ```bash
    python yolo_detect.py --model primary_model.pt --source <image_or_folder_or_video> --resolution 1280x720
    ```
  - Example for webcam:
    ```bash
    python yolo_detect.py --model primary_model.pt --source usb0 --resolution 1280x720
    ```
  - Example for image:
    ```bash
    python yolo_detect.py --model primary_model.pt --source test.jpg
    ```

- **Script Features:**
  - Supports images, folders, videos, and webcam streams
  - Draws bounding boxes and class labels
  - Optionally records video output
  - Press `q` to quit, `s` to pause, `p` to save a frame

---

## 6. References
- See `Train_YOLO_Models.ipynb` for step-by-step code and explanations.
- For more details, visit the [Ultralytics YOLO documentation](https://docs.ultralytics.com/).

---

## 7. Model Optimization and Improvement

To further improve your YOLO model's performance, consider the following strategies:

### 1. Data Augmentation
- Use built-in YOLO augmentations (e.g., mosaic, mixup, flipping, scaling, color jitter).
- Increase dataset diversity by collecting more varied images.

### 2. Hyperparameter Tuning
- Adjust learning rate, batch size, optimizer, and augmentation parameters.
- Use the `--lr0`, `--batch`, and other flags in the training command.
- Try the YOLO hyperparameter evolution feature for automated tuning.

### 3. Model Selection
- Experiment with different YOLO model sizes (e.g., yolov8n, yolov8s, yolov8m, yolov8l, yolov8x).
- Lightweight models are faster but may be less accurate; larger models may yield better results if you have enough data and compute.

### 4. Transfer Learning
- Start training from a pretrained checkpoint (e.g., `yolov8l.pt`) instead of training from scratch.
- Fine-tune on your custom dataset for better results.

### 5. More Training Data
- Annotate and add more labeled images, especially for underrepresented classes or edge cases.

### 6. Validation and Early Stopping
- Monitor validation loss and metrics to avoid overfitting.
- Use early stopping or reduce epochs if validation performance plateaus.

### 7. Post-processing Improvements
- Adjust confidence and IoU thresholds for NMS (Non-Maximum Suppression) to balance precision and recall.

### 8. Analyze Failure Cases
- Review incorrect predictions and add similar examples to your training set.
- Use confusion matrices and error analysis plots to guide improvements.

For more advanced tips, see the [Ultralytics YOLO Optimization Guide](https://docs.ultralytics.com/yolov8/tutorials/optimize/).

---

For any issues, please refer to the notebook or open an issue in this repository.
