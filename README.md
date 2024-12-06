ML_Cancer_Classification
Purpose: This repository contains a project that classifies images into cancerous and healthy categories using machine learning techniques. The project involves processing and extracting features from images, training an artificial neural network, and predicting the category of new images.

Features of This Repository
Image Processing:
Splitting Images: The original images are divided into smaller-sized patches (50x50) pixels.
Normalization: Feature values such as intensity, contrast, entropy, energy, and homogeneity are normalized based on global maximum values across the entire dataset.
Machine Learning Model:
A Multi-Layer Perceptron (MLP) Classifier from sklearn is used to classify patches as either cancerous or healthy.
The classifier is trained using extracted features from processed images.
Prediction:
After training, a new image (new_image.png) can be used for prediction by processing and extracting similar features.
Directory Structure
bash
Copy code
ML_C_H_second_Edition.py          # The main script for training and prediction
Readme_Groupsimilarity_5Features.md # Explanation of feature extraction and grouping
images/                           # Folder containing original images
processed_images/                 # Folder containing processed patches of images
new_image.png                     # Example new image for prediction
README.md                         # README with project overview (this file)
How to Use This Repository
Clone the Repository:

sh
Copy code
git clone https://github.com/ShahramPhotonics/ML_Cancer_Classification.git
Install Dependencies:

You will need Python 3.6+ and the following libraries:
numpy
Pillow (PIL library for Python)
sklearn
Install dependencies using pip:
sh
Copy code
pip install numpy pillow scikit-learn
Run the Script:

Run ML_C_H_second_Edition.py to perform the entire process from image processing to training and prediction.
Ensure the following:
Your input images are placed in the images folder.
The image for prediction is named new_image.png and located in the same directory as the script.
Output:

The processed patches will be saved in the processed_images directory.
The training accuracy of the model will be displayed in the console.
The predicted class for the new image will be printed: 1 for Cancerous and 0 for Healthy.
Detailed Explanation of the Workflow
Image Processing:

The images in the images/ folder are processed as follows:
Splitting into smaller patches to allow focused analysis of different sections.
Normalization of each patch's pixel values using global maximums to ensure consistent comparison.
Saving the patches in the processed_images/ folder for further analysis.
Feature Extraction:

From each patch, five features are extracted:
Intensity: Average intensity of pixels in the patch.
Contrast: Standard deviation of pixel values.
Entropy: Randomness of pixel intensities.
Energy: Sum of squared pixel values.
Homogeneity: Represents similarity between neighboring pixel values.
The features are normalized and grouped to make the analysis more effective.
Training the Model:

A Multi-Layer Perceptron (MLP) with two hidden layers is used for training.
The model uses features extracted from the patches as input.
Training is done on 80% of the dataset, and 20% is used for testing accuracy.
Prediction:

The script can make predictions on a new image (new_image.png) provided in the directory.
The new image is split, normalized, features are extracted, and a prediction is made.
Improvements and Features of This Edition
Global Maximum Normalization: The script calculates global maximum values for features across all images to ensure uniformity in normalization.
Feature Normalization for Prediction: The new image is normalized using the same feature maximum values computed during training to maintain consistency.
Notes
The script requires that input images are large enough to allow for multiple (50x50) patches.
It’s recommended to add more data to improve the accuracy of the model.
Future Improvements
Add cross-validation to further enhance the model's generalization.
Implement other classification models to compare performance with the current MLP classifier.
Provide a web interface to make the tool user-friendly for non-technical users.
Author
Shahram
Photonics engineer specializing in advanced optimization of compositional and structural photonics for optoelectronic devices.

Adding This to Your GitHub Repository
To add this README.md file to your GitHub repository:

Create the README File Locally:

In the ML2 directory, create a new file named README.md:
sh
Copy code
echo "# ML_Cancer_Classification" > README.md
Add Content:

Copy and paste the detailed explanation above into README.md.
Commit and Push to GitHub:

Add the new README.md to Git:
sh
Copy code
git add README.md
Commit the file:
sh
Copy code
git commit -m "Added detailed README file"
Push to GitHub:
sh
Copy code
git push -u origin main
