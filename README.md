# EfficientAD (Adapted)

This code is adapted from the [unofficial implementation](https://github.com/nelson1425/EfficientAD) of the paper https://arxiv.org/abs/2303.14535

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/efficientad-accurate-visual-anomaly-detection/anomaly-detection-on-mvtec-loco-ad)](https://paperswithcode.com/sota/anomaly-detection-on-mvtec-loco-ad?p=efficientad-accurate-visual-anomaly-detection)


## Setup

### Packages

```
Python==3.10
torch==1.13.0
torchvision==0.14.0
tifffile==2021.7.30
tqdm==4.56.0
scikit-learn==1.2.2
```

### Mvtec AD Dataset

For Mvtec evaluation code install:

```
numpy==1.18.5
Pillow==7.0.0
scipy==1.7.1
tabulate==0.8.7
tifffile==2021.7.30
tqdm==4.56.0
```

Download dataset (if you already have downloaded then set path to dataset (`--mvtec_ad_path`) when calling `efficientad.py`).

```
mkdir mvtec_anomaly_detection
cd mvtec_anomaly_detection
wget https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz
tar -xvf mvtec_anomaly_detection.tar.xz
cd ..
```

Download evaluation code:

```
wget https://www.mydrive.ch/shares/60736/698155e0e6d0467c4ff6203b16a31dc9/download/439517473-1665667812/mvtec_ad_evaluation.tar.xz
tar -xvf mvtec_ad_evaluation.tar.xz
rm mvtec_ad_evaluation.tar.xz
```

## efficientad.py

Training with augmented data, inference, and evalution. Augmented images are saved in mvtec_ad_path. Inference results such as anomaly maps are saved in output_dir. Evaluation results including F1 scores are printed out. Image-wise predictions are saved in evaluation_results.csv. MLflow trackings are saved in mlruns/.

```
python efficientad.py --dataset mvtec_ad --subdataset bottle --img_aug
```

Training without augmented data, inference, and evalution. 

```
python efficientad.py --dataset mvtec_ad --subdataset bottle
```

If trained models are available, only conduct inference and evaluation:

```
python efficientad.py --dataset mvtec_ad --subdataset bottle --stage_inference
```

When evaluation results on individual classes are available, summarize and compute global F1 over all classes:

```
python results_summarization.py
```