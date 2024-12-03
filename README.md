# CNN Based Deforestation Detection from Satellite Imagery
This code developed as part of the CS230 course, leverages U-Net, a state-of-the-art image segmentation model, to detect deforestation using Landsat-8 satellite imagery. This model code improves over the baseline U-Net architecture by integrating a Convolutional Block Attention Module (CBAM), Adam optimization algorithm, and OneCycleLR scheduler. These enhancements improved validation Dice scores from a baseline of 76.71% to 88.19%. Along with the training files, the pretrained models for detecting deforestation on 3-band input (RGB) and 7-band input are also provided. This effort facilitates scalable, near-real-time biodiversity conservation efforts by tracking illicit deforestation.

# Dataset Preparation
If you wish to train your own model using the provided code as the baseline, you can access the satellite data and preprocess it using the [Data_processing.py] Data_processing.py

# Modeling

