@echo off
REM This script sets up a DETR-compatible dataset from a YOLOv8 dataset

REM Define variables - CHANGE THESE TO YOUR PATHS
set YOLO_DATASET=D:\Github\safety_gear_detection\safety_gear_detection\safety_gear_detection\css-data
set DETR_DATASET=D:\Github\safety_gear_detection\safety_gear_detection\safety_gear_detection\detr_dataset

REM 1. Create the DETR dataset directory structure
mkdir "%DETR_DATASET%\annotations"
mkdir "%DETR_DATASET%\images\train"
mkdir "%DETR_DATASET%\images\val"
mkdir "%DETR_DATASET%\images\test"

REM 2. Convert YOLOv8 to COCO format for each split
python yolov8_to_coco.py --yolo-path "%YOLO_DATASET%" --output "%DETR_DATASET%\annotations\instances_train.json" --split train
python yolov8_to_coco.py --yolo-path "%YOLO_DATASET%" --output "%DETR_DATASET%\annotations\instances_val.json" --split val
python yolov8_to_coco.py --yolo-path "%YOLO_DATASET%" --output "%DETR_DATASET%\annotations\instances_test.json" --split test

REM 3. Create symbolic links for the images (this requires administrator privileges on Windows)
echo Creating symbolic links for training images...
for %%f in ("%YOLO_DATASET%\train\images\*.*") do (
    mklink "%DETR_DATASET%\images\train\%%~nxf" "%%f"
)

echo Creating symbolic links for validation images...
for %%f in ("%YOLO_DATASET%\val\images\*.*") do (
    mklink "%DETR_DATASET%\images\val\%%~nxf" "%%f"
)

echo Creating symbolic links for test images...
for %%f in ("%YOLO_DATASET%\test\images\*.*") do (
    mklink "%DETR_DATASET%\images\test\%%~nxf" "%%f"
)

echo Done! Your DETR dataset is ready at %DETR_DATASET%
echo You can now train DETR with: python main.py --coco_path %DETR_DATASET%
pause