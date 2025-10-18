Banana Ripeness Classification using VGG16 (Transfer Learning)

This repository contains the necessary code and materials for training a Convolutional Neural Network (CNN) model to classify the ripeness level of bananas using Transfer Learning with the VGG16 architecture.

The project is implemented in a Jupyter Notebook (Classication 2 pretrained ver.ipynb) and is designed to differentiate between three stages of banana maturity.

🍌 Project Overview

The goal of this project is to build an accurate and efficient image classification model capable of distinguishing between three classes of bananas:

Pisang Mentah (Raw/Unripe)

Pisang SetMatang (Semi-Ripe)

Pisang Matang (Ripe)

To achieve this, we leverage the power of Transfer Learning by fine-tuning the pre-trained VGG16 model on our custom dataset, which significantly reduces training time and resource requirements compared to training a network from scratch.

🛠️ Technology Stack

Language: Python

Frameworks: TensorFlow and Keras

Deep Learning Architecture: VGG16 (Used as a feature extractor)

Data Handling & Processing: NumPy, OpenCV (cv2), Matplotlib, and ImageDataGenerator.

📂 Repository Structure (Inferred)

The Jupyter notebook expects the following file structure for the image data:

.
├── Classication 2 pretrained ver.ipynb  # Main training and prediction notebook
├── train/                               # Training images directory
│   ├── Pisang Mentah/
│   ├── Pisang Matang/
│   └── Pisang SetMatang/
└── valid/                               # Validation images directory
    ├── Pisang Mentah/
    ├── Pisang Matang/
    └── Pisang SetMatang/


Key Notebook Steps

Library Import & Data Setup: Imports essential libraries (TensorFlow, Keras, OpenCV, etc.) and defines the paths for the training and validation datasets.

Data Verification: Confirms the total number of images found in each class for both the train and valid sets.

Data Augmentation: Uses ImageDataGenerator with augmentation techniques (rotation, shifts, zoom, horizontal flip) to increase the size and diversity of the training data, improving model generalization.

Transfer Learning Model Setup:

Loads the VGG16 model with pre-trained weights from ImageNet (weights='imagenet').

The classification layers are excluded (include_top=False) so VGG16 acts solely as a feature extraction base.

All layers of the base VGG16 model are frozen (layer.trainable = False) to prevent the pre-trained weights from being overwritten during initial training.

Custom Top Layers: New custom layers (a Flatten layer, a Dense layer with 512 units and ReLU activation, and a Dropout layer) are added on top of the frozen VGG16 base.

Output Layer: The final output layer uses a single unit with sigmoid activation, optimized with binary_crossentropy loss, suggesting the original notebook was adapted or set up for a binary/multi-class problem simplified for evaluation.

Model Training: The model is trained using the augmented data flow with a low learning rate (RMSprop(learning_rate=0.0001)) to fine-tune the custom layers while keeping the core VGG features stable.

Evaluation & Prediction: The model's performance is evaluated on the validation set, and a simple prediction is demonstrated using a sample image (images 2.jpeg).

🚀 Getting Started

To run this project and train the model locally:

Clone the repository:

git clone [Your-Repo-URL]
cd banana-ripeness-classification


Prepare the Data: Ensure you have the train and valid directories structured as described above, containing the respective banana images.

Install Dependencies: Install the required Python packages (e.g., TensorFlow, NumPy, OpenCV) in your environment.

Run the Notebook: Open and execute the cells in the Classication 2 pretrained ver.ipynb notebook.

jupyter notebook
