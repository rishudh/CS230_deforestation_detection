# CNN Based Deforestation Detection from Satellite Imagery
This code developed as part of the CS230 course, leverages U-Net, a state-of-the-art image segmentation model, to detect deforestation using Landsat-8 satellite imagery. This model code improves over the baseline U-Net architecture by integrating a Convolutional Block Attention Module (CBAM), Adam optimization algorithm, and OneCycleLR scheduler. These enhancements improved validation Dice scores from a baseline of 76.71% to 88.19%. Along with the training files, the trained models for detecting deforestation, AugCBAM5 for 3-band input (RGB) and AugCBAM7 for 7-band input are also provided. This effort facilitates scalable, near-real-time biodiversity conservation efforts by tracking illicit deforestation.

# Dataset Preparation
If you wish to train your own model using the provided code as the baseline, you can access the satellite data and preprocess it using the [data_processing.py](data_processing.py). This code is adopted from [multiearth-challenge](https://github.com/MIT-AI-Accelerator/multiearth-challenge/tree/main), and has been modified to extract Landsat-8 imagery for spectral bands of choice.

# Training
The baseline U-Net model has been adopted from [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet). 

[unet](https://github.com/rishudh/CS230_deforestation_detection/tree/main/unet) contains the modified U-Net architecture integrated with CBAM. 

[train.py](https://github.com/rishudh/CS230_deforestation_detection/blob/main/train.py) contains the code for training the model on satellite dataset, and can be used to iterate over different parameter choices. It leverages the modified U-Net architecture.

[predict.py](https://github.com/rishudh/CS230_deforestation_detection/blob/main/predict.py) can be used to create deforestation masks for input satellite imagery. 

To load all the relevant libraries needed to run this code, install [requirements.txt](https://github.com/rishudh/CS230_deforestation_detection/blob/main/requirements.txt).



# Trained Models
You can also use the provided trained models to readily detect deforestation on satellite images over a region of your choice. These models have been trained on Landsat-8 data, and thus will work best with its data. The models can be accessed [here](https://github.com/rishudh/CS230_deforestation_detection/releases/tag/v1.0).

AugCBAM5 model is trained to detect deforestation on 3-band RGB composites.
AugCBAM7 model is trained to detect deforestation on 7-band composistes.



