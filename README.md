# Res-HAAN: A Lightweight Multimodal Fusion Framework for Online Resistance Spot Weld Quality Monitoring

Official PyTorch implementation of the paper  **"Res-HAAN: A Lightweight Multimodal Fusion Framework for Online Resistance Spot Weld Quality Monitoring"** .

This repository provides the code for multi-modal (Infrared Images, Visible Images, and Temporal Process Parameters) fusion, addressing severe class imbalance and multi-task predictions (Classification & Regression) in Resistance Spot Welding (RSW).

## 🚀 Features

* **Asymmetric Multimodal Fusion** : Proposes Hierarchical Attention Aggregation Network (HAAN) using spatial visual features to guide and filter dynamic temporal sequences.
* **Hybrid Imbalance Learning** : Integrates Feature-SMOTE, Focal Loss, and consistency regularization to solve threshold instability for rare defects (e.g., Expulsion).
* **Multi-task Learning** : Jointly optimizes weld quality classification, nugget diameter regression, and tensile shear force prediction.
* **Lightweight & Real-time** : Achieves ~133 FPS inference speed with only 12.57M parameters based on a ResNet-18 backbone.

## 📂 Repository Structure

```
├── data/                   # Directory for dataset and metadata (Add your downloaded data here)
├── prepare_data.py         # Script to preprocess raw dataset and generate regression_stats.json
├── train.py                # Main script for Stratified 5-Fold training and evaluation
├── requirements.txt        # Dependencies
└── README.md               # This file
```

## 📊 Dataset Preparation

The original multimodal dataset used in this study is publicly available at  **Mendeley Data** .

1. Download the raw dataset from the official repository:
   👉 [**Resistance Spot Welding Insights Dataset**](https://data.mendeley.com/datasets/rwh8kjzdch/2 "null")
2. Extract the dataset and place the raw images and `Data_RSW.csv` into the `./data/` folder.
3. Run the preprocessing script to automatically normalize regression targets and generate `final_data.csv`:

```
python data_preprocess.py
```

## 🏃‍♂️ Training & Evaluation

To start the Stratified 5-Fold cross-validation training, simply run:

```
python train.py
```

### Configuration Options

You can easily reproduce the ablation studies reported in our paper by modifying the `ProjectConfig` class in `train.py`:

* **Modality Ablation** : `MODALITY_MODE = 'all' | 'vision_only' | 'temporal_only'`
* **Fusion Architecture** : `FUSION_MODE = 'haan' | 'concat' | 'symmetric' | 'cmt' | 'healnet'`
* **Training Strategy** : `TRAINING_STRATEGY = 'proposed' | 'baseline' | 'ros' | 'focal' | 'ldam' | 'smote+ldam'`

## 📝 Citation

If you find this code or our paper useful for your research, please consider citing our work:

```
@article{YourName202X,
  title={Res-HAAN: A Lightweight Multimodal Fusion Framework for Online Resistance Spot Weld Quality Monitoring},
  author={Your Name and Co-authors},
  journal={The International Journal of Advanced Manufacturing Technology (Under Review)},
  year={202X}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
