# Training and Prediction of Crown Dieback and Dead Trees Using the YOLO11s Model in Python

## Introduction

Crown dieback and dead trees are critical phenomena that occur due to environmental stressors (such as drought, pollution, and disease) or non-environmental factors including mechanical damage and improper forest management. Timely and accurate identification of affected trees plays a vital role in sustainable forest management, early intervention, and ecosystem resilience.

In recent years, artificial intelligence algorithms have advanced significantly, enabling the detection and monitoring of such phenomena using remote sensing data (including aerial imagery, UAV images, and satellite observations). Among these, the YOLO11 model stands out as a state-of-the-art object detection framework capable of precisely localizing target objects through bounding-box predictions. In this context, YOLO11 can be effectively employed to identify trees exhibiting crown dieback or full mortality.

Accordingly, this repository presents a structured four-step pipeline for training and applying the YOLO11s model to detect dieback and dead trees in forested areas. From data preparation and labeld to model training, inference, and result interpretation, each stage is explained in detail to support reproducibility and further development.

Ultimately, it is hoped that this work contributes (however modestly) to the broader mission of conserving our invaluable forest resources: the very ecosystems that sustain our breath, enrich our landscapes, and provide countless ecological, economic, and cultural services.


## 1. Preparing train and test samples

To prepare the training samples for dieback and dead tree detection, the original imagery is first preprocessed by slicing it into tiles of size 512×512 pixels. The choice of tile dimensions is informed by the spatial resolution of the imagery and the approximate size of the target objects (i.e., dieback and dead trees). This ensures that each tile captures sufficient contextual information while maintaining compatibility with the YOLO object detection framework. Tiling also facilitates efficient training by standardizing input dimensions and improving object localization across varied spatial contexts.
Following the tiling process, manual annotation is performed using the Sreeni labeling tool. The annotations are exported in YOLO-compatible format, including bounding box coordinates and class identifiers.
This annotated dataset forms the foundation for training a YOLO-based model capable of detecting dieback and dead tree in forested areas.

After the imagery is crop, the data can be imported into the Sreeni environment for annotation.
To use the Serrni annotation tool, follow these steps:
First, a Python environment can be created in your desired drive and folder.  the following steps are performed in the miniconda prompt:


```bash
conda create --prefix D:\D\Python_environments\ digitalAlsreeni python=3.9.21
```

After the environment is created, we activate it.

```bash
conda activate "D:\D\Python_environments\ digitalAlsreeni"
```
The desired package must be installed. Therefore, the following command is executed in the Conda Prompt and then we open it by writing the word **sreeni**.

```bash
pip install digitalsreeni-image-annotator==0.1.7
sreeni
```
According to the interface shown below:


•	Import the images into the annotation tool (Section 1).

•	Create the relevant classes corresponding to the target (Section 2).

•	Select the polygon tool for precise annotation (Section 3). Carefully annotate all trees belonging to the defined classes in each image.

•	Finally, export the annotations in the desired format (Section 4), which in this case is YOLO (v5+).


Upon export, the software will generate and save the YOLO-compatible files in the designated output folder. This includes the training and test datasets, their corresponding label files, and the associated YAML configuration file.

<img width="1368" height="728" alt="sreeni" src="https://github.com/user-attachments/assets/bdedd81b-2027-4582-baf5-183249beddef" />


Finally, the following folders should be present in the designated directory. See the figure below for reference.


<img width="973" height="474" alt="yaml" src="https://github.com/user-attachments/assets/967165ec-ca43-48c4-8066-0c9a5ff9b474" />


Once all the above steps are completed, the training and test samples for detecting crown dieback and dead trees are now prepared. Therefore, we proceed to the next stage that is model training.


## 2. YOLO11s box detection training

To use the YOLOv11 model, the required packages must be installed. First, a dedicated Python environment should be created in conda prompt.

```bash
conda create --prefix D:\D\Python_environments\yolo python=3.9.21
conda activate " D:\D\Python_environments\yolo"
```

After activating the created environment, the required packages are installed.
```bash
Conda install spyder
pip install matplotlib
pip install pillow
pip install numpy
pip install ipython
pip install Ultralytics
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install notebook
```

After successfully installing the required packages, spyder can be launched by typing **spyder** in the conda prompt.

To begin the process of detecting crown dieback and dead trees using YOLO11, the necessary Python packages are first imported:

```python
import os
HOME = os.getcwd()
print(HOME)
from IPython import display
display.clear_output()
import ultralytics
ultralytics.checks()
from ultralytics import YOLO
from IPython.display import display, Image
```
In this step, the paths to the dataset and configuration files are defined:

```python
# Path to the YAML configuration file
data_yaml= "D:/F/512/PostDoctoral/Images/AlangDarreh/YoloProcessing/2016_samplesYoloDataSet/data.yaml"

# Path to the YOLOv11 model weights
model_weights = 'yolo11s.pt'
model = YOLO(model_weights)
```

**data_yaml:** This file contains metadata about the dataset, including class names, and the locations of training and validation images and labels.

**model_weights:** Specifies the pretrained YOLO11s model to be used as the starting point for training or inference.

These paths must be correctly set to ensure the model can access the necessary data and configuration during training.
If you have already downloaded the YOLO11s model, you can directly specify its local path to avoid automatic re-downloading. You might express this in code like:

```python
model_weights = "C:/path/to/yolo11s.pt"  # Use local weights to skip re-downloading
model = YOLO(model_weights)
```

To begin training the YOLO11s model, key hyperparameters are defined as follows:

```python
model.train(
    data=data_yaml,     # Path to the dataset YAML file
    epochs=100,         # Number of training epochs (increased for better convergence)
    imgsz=512,          # Input image size (resized to 512×512)
    lr0=0.003,          # Initial learning rate
    save=True           # Save model checkpoints after training
)
```
These parameters control the core aspects of the training process:

•	**data:** points to the YAML file that defines the dataset structure, class names, and paths to training/validation data.

•	**epochs:** determines how many full passes the model makes over the training data.

•	**imgsz:** sets the resolution of input images; higher values may improve accuracy but increase memory usage.

•	**lr0:** the initial learning rate; tuning this affects how fast the model learns.

•	**save:** ensures that trained weights and logs are saved automatically.

•	Additional parameters such as batch, optimizer, patience, device, cache, and augment can be added for more control (but for now, this configuration is sufficient to begin training).


After executing the above codes, the model begins the training process.

<img width="1368" height="738" alt="yolo_start" src="https://github.com/user-attachments/assets/05510c5d-f31f-47f1-b423-916a586f5da7" />


After training is complete, the model's training outputs can be found in the default Windows installation path under:
C:/Users/<ComputerName>/runs/detect/train/


<img width="1137" height="271" alt="results_yolo" src="https://github.com/user-attachments/assets/94fc2764-fdaf-4d43-917c-ee52b54146d2" />


Inside the weights folder, you will find both the last model and the best-performing model saved during training. These can be used to perform inference on your target classes. It is generally recommended to use the best model (best.pt), as it achieved the highest validation performance during training.
To detect the target classes (e.g., crown dieback and dead trees), the trained model is first loaded and then applied to the desired image(s). The predictions are visualized using matplotlib package.
```python
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Load the best-performing trained model
model = YOLO(r" D:\F\512\PostDoctoral\Images\YoloProcessing\yolo2016\train\weights\best.pt")

# Run inference on a selected image
results = model(r"D:\F\512\PostDoctoral\Images\YoloProcessing\yolo2016\crop_images2016\2016_1194.tif")

# Visualize predictions
for r in results:
    im_array = r.plot()  # Draw bounding boxes and labels on the image (returns NumPy array)
    im = Image.fromarray(im_array[..., ::-1])  # Convert BGR to RGB for display

    plt.figure(figsize=(10, 10))
    plt.imshow(im)
    plt.axis('off')  # Hide axes for cleaner output
    plt.show()
```

<img width="691" height="228" alt="predect" src="https://github.com/user-attachments/assets/9c9c6192-3200-4bcf-9b46-e8a4d60e45ac" />


## Preparing data for Predicting
At this stage, the model is fully trained and ready for inference. If you are working with large forest areas, it's essential to divide the imagery into smaller tiles (preferably 512×512 pixels, which aligns with the model's input configuration).
To ensure accurate detection and seamless reconstruction:

•	Georeferenced tiles are recommended so that predictions can later be merged into a unified output.

•	Apply side overlaps (e.g., 128 pixels) when cropping the images. This helps avoid edge-related prediction errors and improves crown detection for trees located near tile boundaries.


## 3. Predicting on Overlapping Cropped Tiles
We now proceed to run inference on the cropped image tiles with side overlaps, ensuring robust detection across the entire forest extent.

## 3.1. Load the required packages
Before processing cropped forest images with overlaps, we need to import the necessary libraries. If any of these libraries are not yet installed, they can be installed individually in the Conda Prompt (as explained earlier).
```python
import os
import glob
from ultralytics import YOLO
import geopandas as gpd
from shapely.geometry import Polygon
import rasterio
from pyproj import CRS
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from shapely.ops import unary_union
from shapely.geometry import box
```

## 3.2. Define Paths and Directories
First, we set up the input and output directories for the cropped images, prediction results, and shapefiles. These paths ensure that the workflow is organized and that results are saved in the correct locations. If the output or shapefile folders do not already exist, they will be created automatically.
```python
# 1. Path settings
input_dir = r"D:\F\512\PostDoctoral\Images\YoloProcessing\yolo2016\crop_images_prediction"  # Input directory (cropped images)
shapefile_dir = r"D:\F\512\PostDoctoral\Images\YoloProcessing\yolo2016\shp"                            # Directory for shapefiles
model_path = r" D:\F\512\PostDoctoral\Images\YoloProcessing\yolo2016\train\weights\best.pt"  # Path to trained model

os.makedirs(shapefile_dir, exist_ok=True)  # Create shapefile folder if it does not exist
```

## 3.3. Load the YOLO Model
Next, the trained YOLOv11 model is loaded from the specified path. This model will be used to perform predictions on the cropped forest images.
```python
model = YOLO(model_path)   # Load the trained YOLO model from the specified path
```

## 3.4. Define the Coordinate Reference System (CRS)
Finally, we specify the spatial reference system. Here, UTM Zone 40N (EPSG:32640) is chosen, which is suitable for georeferenced data in my study area which is in part of Iran. This ensures that predictions can later be merged and aligned correctly with geographic coordinates.
```python
utm_crs = CRS.from_epsg(32640)   # Define CRS: EPSG:32640 (UTM Zone 40N, WGS84 (my study area))
```

## 3.5. Process Each Image and Collect Detections
The script loops through all .tif images in the input directory. For each image:

•	The geospatial transform and CRS are extracted using rasterio.

•	YOLO performs predictions with a confidence threshold of 0.2.

•	For each detected bounding box, pixel coordinates are converted into UTM coordinates using the image transform.

•	A polygon representing the bounding box in geographic space is created.

•	Detection details (geometry, filename, class label, confidence score, and centroid coordinates) are stored in the all_features list for later use (e.g., exporting to shapefiles or merging results).

```python
all_features = []

for img_path in glob.glob(os.path.join(input_dir, "*.tif")):
    # Read image with rasterio to extract geographic transform
    with rasterio.open(img_path) as src:
        transform = src.transform
        crs = src.crs if src.crs is not None else utm_crs  # Use UTM if CRS is missing
        
    # Run YOLO prediction
    results = model(img_path, conf=0.2)
    filename = os.path.basename(img_path)
    
    for r in results:
        for box in r.boxes:
            # Convert pixel coordinates to geographic coordinates
            x1_px, y1_px, x2_px, y2_px = box.xyxy[0].tolist()
            
            # Transform corner points into UTM coordinates
            x1_utm, y1_utm = transform * (x1_px, y1_px)
            x2_utm, y2_utm = transform * (x2_px, y2_px)
            
            # Create polygon in UTM space
            polygon = Polygon([
                (x1_utm, y1_utm),
                (x2_utm, y1_utm),
                (x2_utm, y2_utm),
                (x1_utm, y2_utm)
            ])
            
            all_features.append({
                'geometry': polygon,
                'filename': filename,
                'class': model.names[int(box.cls)],
                'confidence': float(box.conf),
                'x_utm': (x1_utm + x2_utm) / 2,  # Box centroid X
                'y_utm': (y1_utm + y2_utm) / 2   # Box centroid Y
            })
```

## 3.6. Remove Duplicate Detections with Geographic Non-Maximum Suppression (NMS)
After collecting all detections from the cropped images, some bounding boxes may overlap significantly, representing duplicate detections of the same tree crown. To refine the results, Geographic Non-Maximum Suppression (NMS) is applied:

•	A GeoDataFrame (gdf) is created from all detected features, using the defined UTM coordinate system.

•	Detections are sorted by confidence score in descending order, ensuring that higher-confidence predictions are prioritized.

•	Each polygon is compared with others to check for overlap greater than 50%.

•	If overlap is detected, the polygon with the lower confidence score is removed.

•	This ensures that only the most reliable detection remains.

•	The final set of polygons (final_features) is stored in a new GeoDataFrame (final_gdf).

•	The cleaned results are saved as a shapefile in the specified directory, ready for geospatial analysis or visualization.

•	A summary message confirms the output path and the number of final detections.

```python
if all_features:
    gdf = gpd.GeoDataFrame(all_features, crs=utm_crs)
    
    # Apply NMS based on geographic overlap and confidence score
    final_features = []
    processed_indices = set()
    
    # Sort detections by confidence (descending)
    gdf = gdf.sort_values(by='confidence', ascending=False)
    
    for i in range(len(gdf)):
        if i in processed_indices:
            continue
            
        current_feature = gdf.iloc[i]
        current_poly = current_feature['geometry']
        current_conf = current_feature['confidence']
        to_remove = []
        
        # Check overlap with other polygons
        for j in range(i + 1, len(gdf)):
            if j in processed_indices:
                continue
                
            other_poly = gdf.iloc[j]['geometry']
            other_conf = gdf.iloc[j]['confidence']
            
            # Calculate intersection area
            intersection_area = current_poly.intersection(other_poly).area
            min_area = min(current_poly.area, other_poly.area)
            
            # If overlap exceeds 50%
            if intersection_area / min_area > 0.5:
                # Remove polygon with lower confidence
                if current_conf >= other_conf:
                    to_remove.append(j)
                else:
                    # If the other polygon has higher confidence, remove current one
                    to_remove.append(i)
                    break  # Exit loop since current polygon is removed
        
        # Keep polygon if not removed (highest confidence retained)
        if i not in to_remove:
            final_features.append(current_feature)
            processed_indices.update(to_remove)
            processed_indices.add(i)
    
    # Create final GeoDataFrame
    final_gdf = gpd.GeoDataFrame(final_features, crs=utm_crs)
    
    # Save final shapefile
    output_merged_path = os.path.join(shapefile_dir, "final_merged_2016.shp")
    final_gdf.to_file(output_merged_path, encoding='utf-8')
    print(f" Final detections without duplicates saved at:\n{output_merged_path}")
    print(f"Number of final detections: {len(final_gdf)}")
else:
    print(" No detections were made!")

print(" Processing of all images completed!")
```

## 4. Visualizing Results in an Orthomosaic
Once the processing is completed successfully and the shapefile has been saved, the detection results can be visually inspected on the orthomosaic image. This allows you to evaluate how well the model has identified tree crowns dieback and dead trees across the forest area.

To do this:

•	Open the final shapefile in ArcGIS (or QGIS) alongside the orthomosaic raster.

•	Each detection polygon will be displayed in its correct geospatial location, overlaid on the imagery.

You can explore the attribute table, which contains key information for each detection, such as:

•	The filename of the source tile

•	The predicted class (e.g., crown dieback, dead tree)

•	The confidence score assigned by YOLO11s

•	The centroid coordinates in UTM

<img width="558" height="492" alt="results_presiction" src="https://github.com/user-attachments/assets/4443e0c5-d2d1-4f4b-9d2f-f22db3420eea" />


You can further explore the results through the Layer Properties panel:

•	Open the Properties of the prediction layer.

•	Navigate to the Symbology section.

•	Use the class column (the attribute that stores the predicted class name) to symbolize the detections.

This allows you to visually separate and highlight the bounding boxes corresponding to crown dieback trees and dead trees, each with distinct colors or styles.

You can also inspect the attribute table to see confidence scores and other metadata for each detection, ensuring that the visual representation matches the model’s predictions.

<img width="787" height="545" alt="results_presiction_VISU" src="https://github.com/user-attachments/assets/77ae0b7b-c8a2-40cc-bb49-2d557e818648" />

