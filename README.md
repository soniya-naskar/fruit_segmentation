# fruit_segmentation
Fruit Image Segmentation & Classification using YOLOv8
=======================================================

This project performs instance segmentation and classification of fruits — apple, banana, and orange — using YOLOv8. It also generates RGB segmentation masks, trains a model, evaluates using metrics like mAP and IoU, and saves the classified test images.

-------------------------------------------------------
The provided dataset has been unzipped and since it was in coco json format so it has been converted into yolo format with the file structure given below.
Project Structure
-----------------
fruit_dataset/
├── train/
│   ├── images/
│   ├── labels/
│   ├── masks/         <- RGB segmentation masks (auto-generated)
│   └── _annotations.coco.json
├── valid/
├── test/
data.yaml              <- YOLO dataset config file

-------------------------------------------------------
Classes & RGB Colors
--------------------
| Label | Class   | RGB Mask Color         |
|-------|---------|------------------------|
| 0     | none    | (0, 0, 0)     - Black  |
| 1     | apple   | (156, 2, 27)  - Red    |
| 2     | banana  | (237, 214, 62) - Yellow |
| 3     | orange  | (230, 103, 30) - Orange |

-------------------------------------------------------
How to Use
----------

1. Unzipped the file, installed all the dependencies in requirements.txt mentioned below:
   

2. Prepare your dataset:
   - For each of the `train/`, `valid/`, and `test/` folders:
     - Move all images to `images/` subfolder.
     - Ensure YOLO-format `.txt` label files are in `labels/`.
     - Add the COCO-style annotation file (`_annotations.coco.json`) to each folder.
     - If not already present, create the `masks/` folder for storing RGB segmentation masks.

3. Generate RGB Masks:
   - Run the mask generation script to create RGB masks using polygon segmentation:
     - Example:
       ```python
       for split in ['train', 'valid', 'test']:
           create_rgb_masks(f"fruit_dataset/{split}/images", 
                            f"fruit_dataset/{split}/labels", 
                            f"fruit_dataset/{split}/masks")
       ```

4. Create `data.yaml`:
   - Point it to your dataset path and define class names.
     ```yaml
     path: fruit_dataset
     train: train/images
     val: valid/images
     test: test/images
     nc: 3
     names: ["apple", "banana", "orange"]
     ```

5. Train the YOLOv8 Model:
   - Use a pre-trained YOLOv8 segmentation model (e.g., yolov8n-seg.pt).
   - Example:
       ```python
       from ultralytics import YOLO
       model = YOLO("yolov8n-seg.pt")
       model.train(data="data.yaml", epochs=30, imgsz=416,  batch=8,
    project="fruit_detection_1",
    name="yolov8_fruit_detector_3",
    val=True)
       ```
   ***For better result epochs=50 and imgsz=640 can be taken, I was having some issue in my system ,the kernel was dying automatically while training the model with the same epochs and imgsz so I have reduced both the parameter.***

7. Loaded the trained data and evaluated validation data:
   - After training:
       ```python
      model = YOLO("fruit_detection_1/yolov8_fruit_detector_3/weights/best.pt")
      results = model("fruit_dataset/test/images/043_jpg.rf.fa255d6b50bbd925aa2ae76c33c530b0.jpg", save=True)
       ```
8. Evaluated the metrics(precision,recall,f1 score):
     # Segmentation metrics
     print(f"Segmentation mAP@0.5: {metrics.seg.map50:.4f}")
     print(f"Segmentation mAP@0.5:0.95: {metrics.seg.map:.4f}")
     ...

9. Run Inference and Save Predictions:
   - To run inference on all test images and save outputs in csv in test_images folder:
       ```python
       from ultralytics import YOLO
       import os
       model = YOLO("fruit_detection_1/yolov8_fruit_detector_3/weights/best.pt")  # path to your best model
       test_img_dir = "fruit_dataset/test/images"
       output_csv = "test_image.csv"
       results_list = []
       ...

       ```
   - also after running the inference the test images are saved with masks in test_prediction folder:
       ```python
       
       output_dir = "test_predictions"
       os.makedirs(output_dir, exist_ok=True)
       image_files = [f for f in os.listdir(test_img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
       for image_file in tqdm(image_files, desc="Predicting and saving masks"):
       image_path = os.path.join(test_img_dir, image_file)
       results = model(image_path)  # Run inference
       # Save result with masks as PNG
       save_path = os.path.join(output_dir, os.path.splitext(image_file)[0] + ".png")
       results[0].save(filename=save_path)


-------------------------------------------------------
Evaluation Metrics
------------------
Segmentation:
- mAP@0.5
- mAP@0.5:0.95

Classification:
- Precision
- Recall
- F1 Score

-------------------------------------------------------
Dependencies
------------
Listed in requirements.txt:
- os
- zipfile
- json
- shutil
- PIL
- pandas
- ultralytics
- numpy
- opencv-python
- matplotlib
- tqdm
  

-------------------------------------------------------
Author
------
Soniya Naskar  
B.Tech in Computer Science & Engineering  
LinkedIn: https://www.linkedin.com/in/soniya-naskar-8279801a3/



