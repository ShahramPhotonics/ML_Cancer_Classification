### README for Cancer vs. Healthy Image Classification Script (Second Edition)

**Purpose**:
This script aims to classify images into cancerous and healthy categories. It involves preprocessing images by splitting them into patches, normalizing several feature values (intensity, contrast, entropy, energy, and homogeneity), extracting features, training a machine learning model, and predicting labels for new images. Below, you will find a detailed explanation of each step in the process.

**Dependencies**:
- Python 3.6+
- `numpy` for numerical operations
- `Pillow (PIL)` for image processing
- `sklearn` for machine learning tasks

Ensure you have these packages installed in your environment before running the script.

### Overview of Steps
1. **Directory Setup and Base Paths**
2. **Calculate Maximum Values for All Features**
3. **Image Processing - Split, Normalize, and Save**
4. **Feature Extraction and Model Training**
5. **Prediction on a New Image**

---

### Step-by-Step Explanation

#### 1. Directory Setup and Base Paths
- The script sets up paths for input (`images` folder) and output (`processed_images` folder) relative to the script's location.
- It ensures that the `processed_images` folder is created if it doesn’t already exist by using `Path(output_folder).mkdir(exist_ok=True)`.
- This allows the script to run flexibly regardless of the specific directory as long as the folder structure is consistent.

#### 2. Calculate Maximum Values for All Features
- The script calculates **global maximum values** for each feature across all images in the `images` folder. These global maximums are essential for consistent normalization across all patches.
- **Features Calculated**:
  - **Intensity (Imax)**: Maximum pixel value from all images.
  - **Contrast**: Calculated as the standard deviation of pixel values for each image.
  - **Entropy**: Represents randomness in the pixel values, calculated from the histogram.
  - **Energy**: Sum of squared pixel values, normalized between 0 and 1.
  - **Homogeneity**: Represents the similarity between neighboring pixel values (calculated as the mean).
- These maximums are used to normalize the respective features during patch processing.

#### 3. Image Processing - Split, Normalize, and Save
- **Splitting Images into Patches**:
  - Each image is split into **patches of size `(50, 50)`** pixels using a grid pattern.
  - Only patches that fully fit into the original image without remainder are processed.
- **Normalizing Features**:
  - **Intensity**: Each patch is normalized by mapping pixel values between 0 and `Imax`.
  - **Other Features (Contrast, Entropy, Energy, Homogeneity)**: For each patch, these features are calculated, normalized using their respective global maximum values, and scaled accordingly.
  - Each patch is saved with an appropriate filename indicating its category (prefix `c` for cancerous or `h` for healthy).
- The processed patches are then saved in the `processed_images` folder.

#### 4. Feature Extraction and Model Training
- **Loading Processed Images**:
  - The processed patches are loaded from the `processed_images` folder.
  - Each patch is labeled as **1** for cancerous or **0** for healthy based on its filename.
- **Feature Extraction**:
  - **Five features** are extracted for each patch:
    1. **Intensity (Mean)**: Average pixel intensity of the patch.
    2. **Contrast (Standard Deviation)**: Spread of intensity values.
    3. **Entropy**: Measure of randomness, calculated from the histogram.
    4. **Energy**: Normalized sum of squared pixel values.
    5. **Homogeneity**: Average homogeneity, calculated as the mean of the pixel values.
- **Feature Grouping**:
  - Features are grouped into **10 equally spaced ranges** to reduce complexity and improve generalization.
  - This is done using a `group_subsections` function that helps aggregate feature values into meaningful groups.
- **Flattening and Preparing Dataset**:
  - The grouped features are flattened and used as input features (`X`) for the machine learning model.
  - The dataset (`X` for features and `y` for labels) is split into **training and testing sets** using an 80-20 ratio.
- **Training the Model**:
  - A **Multi-Layer Perceptron (MLP) classifier** with two hidden layers is used for training.
  - The model is trained with up to **500 iterations** or until convergence.

#### 5. Prediction on a New Image
- **Loading and Preprocessing the New Image**:
  - The script loads a file named `new_image.png` from the base directory.
  - The new image is resized to `(500, 500)` pixels, converted to grayscale, and normalized using the previously computed global `Imax`.
- **Feature Extraction and Prediction**:
  - The new image is split into patches, and features are extracted similarly to the training set.
  - The features are grouped and flattened for prediction.
  - The **MLP classifier** predicts whether the image is **cancerous (1)** or **healthy (0)**, and the result is printed to the console.

### Directory Structure
To run the script, ensure the following structure is in place:
```
ML_C_H_second_Edition.py  # The main Python script
images/                   # Folder with original images
processed_images/         # Folder where processed patches are saved
new_image.png             # Image to be predicted
```

### Usage Instructions
1. **Prepare Input Images**:
   - Place your original images in the `images` folder. Use appropriate naming conventions (`c` for cancerous, `h` for healthy).
2. **Run the Script**:
   - Execute the script by running `python ML_C_H_second_Edition.py` from the command line.
3. **View Results**:
   - The console output will show the global maximum feature values, the model accuracy after training, and the prediction for `new_image.png`.

### Notes
- **Environment Setup**: Make sure you are running the script in a Python environment where all dependencies are installed (`Pillow`, `sklearn`, `numpy`). Use Conda or virtualenv as needed.
- **Image Size Requirements**: The input images should be large enough to allow splitting into `(50, 50)` patches. If the images are too small, adjust the `patch_size` parameter to avoid issues.
- **Handling Edge Cases**: If all pixel values in a patch are the same, the script handles it by normalizing to zero to avoid division by zero errors.

### Improvements in This Edition
- **Global Maximum Normalization for Features**: All features (intensity, contrast, entropy, energy, and homogeneity) are now normalized using the global maximum values across the dataset to ensure consistent scaling.
- **Error Handling**: Enhanced normalization function to handle edge cases, such as zero data range, preventing runtime warnings and errors.

gi