import os
import numpy as np
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Set the base directory to where the script is located
base_dir = os.path.dirname(os.path.abspath(__file__))

# Define paths based on the script's directory
input_folder = os.path.join(base_dir, "images")  # Folder containing the original images
output_folder = os.path.join(base_dir, "processed_images")  # Folder for processed images
Path(output_folder).mkdir(exist_ok=True)

# Function to normalize an image or feature array with a given max value
def normalize_data(data, global_max):
    data_range = data.max() - data.min()
    if data_range == 0:
        return np.zeros_like(data)  # Handle the case where all values are the same
    normalized = (data - data.min()) / data_range
    scaled = normalized * global_max
    return scaled

# Function to split image into smaller patches
def split_image(image, patch_size=(50, 50)):
    patches = []
    img_array = np.array(image)
    h, w = img_array.shape
    for i in range(0, h, patch_size[0]):
        for j in range(0, w, patch_size[1]):
            patch = img_array[i:i + patch_size[0], j:j + patch_size[1]]
            if patch.shape[0] == patch_size[0] and patch.shape[1] == patch_size[1]:
                patches.append(Image.fromarray(patch))
    return patches

# Step 1: Determine the maximum intensity and feature values across all images before processing
Imax = 0
contrast_max = 0
entropy_max = 0
energy_max = 0
homogeneity_max = 0

for filename in os.listdir(input_folder):
    if filename.endswith(".png"):
        filepath = os.path.join(input_folder, filename)
        with Image.open(filepath) as img:
            img = img.convert("L")  # Convert to grayscale
            img_array = np.array(img)
            
            # Update global maximum for intensity
            current_max = img_array.max()
            if current_max > Imax:
                Imax = current_max
            
            # Calculate contrast (standard deviation)
            contrast = img_array.std()
            if contrast > contrast_max:
                contrast_max = contrast
            
            # Calculate entropy
            hist, _ = np.histogram(img_array, bins=256, range=(0, 256), density=True)
            hist = hist + 1e-9  # Avoid log(0)
            entropy = -np.sum(hist * np.log2(hist))
            if entropy > entropy_max:
                entropy_max = entropy
            
            # Calculate energy
            energy = np.sum((img_array / 255.0) ** 2)
            if energy > energy_max:
                energy_max = energy
            
            # Calculate homogeneity (mean value placeholder)
            homogeneity = np.mean(img_array) / 255.0
            if homogeneity > homogeneity_max:
                homogeneity_max = homogeneity

print(f"Maximum intensity value across all images (Imax): {Imax}")
print(f"Maximum contrast value: {contrast_max}")
print(f"Maximum entropy value: {entropy_max}")
print(f"Maximum energy value: {energy_max}")
print(f"Maximum homogeneity value: {homogeneity_max}")

# Step 2: Process each image
patch_size = (50, 50)  # Adjust patch size as needed
patch_count_per_image = 100  # Number of patches per image

for filename in os.listdir(input_folder):
    if filename.endswith(".png"):
        filepath = os.path.join(input_folder, filename)
        label = "c" if filename.startswith("c") else "h"

        # Open image
        with Image.open(filepath) as img:
            img = img.convert("L")  # Convert to grayscale

            # Split into patches
            patches = split_image(img, patch_size)

            # Normalize each patch for intensity and multiply by Imax
            processed_patches = []
            for patch in patches:
                patch_array = np.array(patch, dtype=np.float32)
                
                # Normalize intensity
                normalized_intensity = normalize_data(patch_array, Imax)
                
                # Calculate and normalize other features (contrast, entropy, energy, homogeneity)
                contrast = patch_array.std()
                normalized_contrast = normalize_data(np.array([contrast]), contrast_max)[0]

                hist, _ = np.histogram(patch_array, bins=256, range=(0, 256), density=True)
                hist = hist + 1e-9  # Avoid log(0)
                entropy = -np.sum(hist * np.log2(hist))
                normalized_entropy = normalize_data(np.array([entropy]), entropy_max)[0]

                energy = np.sum((patch_array / 255.0) ** 2)
                normalized_energy = normalize_data(np.array([energy]), energy_max)[0]

                homogeneity = np.mean(patch_array) / 255.0
                normalized_homogeneity = normalize_data(np.array([homogeneity]), homogeneity_max)[0]

                # Combine normalized intensity with other normalized features
                final_patch_array = normalized_intensity  # Placeholder: In actual implementation, we could add more complex combinations here
                processed_patches.append(Image.fromarray(final_patch_array.astype(np.uint8)))

            # Select a fixed number of patches if more are available
            if len(processed_patches) > patch_count_per_image:
                processed_patches = processed_patches[:patch_count_per_image]

            # Save patches with appropriate labeling
            for i, patch in enumerate(processed_patches):
                patch_filename = f"{label}{str(i).zfill(4)}.png"
                patch.save(os.path.join(output_folder, patch_filename))

# Step 3: Load processed images for training the ML model
# Feature extraction

def extract_features(image, subsection_size=(50, 50)):
    rows, cols = image.shape
    sub_h, sub_w = subsection_size
    all_features = []

    for i in range(0, rows, sub_h):
        for j in range(0, cols, sub_w):
            subsection = image[i:i+sub_h, j:j+sub_w]
            features = []

            # 1. Intensity (Mean)
            intensity = np.mean(subsection)
            features.append(intensity)

            # 2. Contrast (Standard Deviation)
            contrast = np.std(subsection)
            features.append(contrast)

            # 3. Entropy
            hist, _ = np.histogram(subsection, bins=256, range=(0, 256), density=True)
            hist = hist + 1e-9  # Avoid log(0)
            entropy = -np.sum(hist * np.log2(hist))
            features.append(entropy)

            # 4. Energy (Sum of squared pixel values normalized)
            energy = np.sum((subsection / 255.0) ** 2)
            features.append(energy)

            # 5. Homogeneity (Dummy placeholder)
            homogeneity = np.mean(subsection) / 255.0
            features.append(homogeneity)

            all_features.append(features)

    return np.array(all_features)

# Group subsections function to group feature values
def group_subsections(features, num_groups):
    grouped_features = []
    for feature_idx in range(features.shape[1]):
        feature_column = features[:, feature_idx]
        min_val, max_val = np.min(feature_column), np.max(feature_column)
        thresholds = np.linspace(min_val, max_val, num_groups + 1)
        groups = [[] for _ in range(num_groups)]

        for feature in feature_column:
            for idx in range(num_groups):
                if thresholds[idx] <= feature < thresholds[idx + 1]:
                    groups[idx].append(feature)
                    break
        grouped_features.append([np.mean(group) if group else 0 for group in groups])
    return np.array(grouped_features)

# Load images for training
def load_images(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith(".png"):
            filepath = os.path.join(folder, filename)
            image = np.array(Image.open(filepath).convert("L"))
            images.append(image)
            labels.append(1 if filename.startswith("c") else 0)  # 1 = cancerous, 0 = healthy
    return images, labels

# Load processed images for training
images, labels = load_images(output_folder)

subsection_size = (50, 50)
num_groups = 10

# Extract features from images
features = [extract_features(img, subsection_size) for img in images]
grouped_features = [group_subsections(feature, num_groups) for feature in features]

# Flatten grouped features and prepare the dataset
X = np.array([group.flatten() for group in grouped_features])
y = np.array(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a neural network
clf = MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Step 4: Predict on a new image
new_image_path = os.path.join(base_dir, "new_image.png")
if os.path.exists(new_image_path):
    with Image.open(new_image_path) as new_image:
        new_image = new_image.convert("L").resize((500, 500))
        # Normalize the new image using Imax
        new_image_normalized = normalize_data(np.array(new_image), Imax)

        # Extract features and group for prediction
        new_features = extract_features(new_image_normalized, subsection_size)
        new_group = group_subsections(new_features, num_groups).flatten()

        # Make prediction
        prediction = clf.predict([new_group])
        print(f"Prediction for new image (1 = Cancerous, 0 = Healthy): {prediction[0]}")
else:
    print(f"New image for prediction not found at: {new_image_path}")
