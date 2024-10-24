# Transferable automatic haematological cell classification  <span style="font-size: 18px;">- Overcoming data limitations with self-supervised learning</span>

🩸 **Want to classify your own blood cell images without labeling thousands of cells and spending a fortune on GPU training?** 💻✨

You’re in the right place! With just **50 labels per class**, you can train a reliable classifier right on your laptop—no GPU required! 🎉

Interested? 🤔 Try it out with the **three publicly available blood cell image datasets** we've already set up for you! 📊🔬

![Classification balanced accuracy, blood datasets](imgs/wenderoth_visual_abstract-1.png "Balenced Acc for all Blood datasets")

## 🌟 Features

This repository showcases the power of **self-supervised learning (SSL)** for hematological cell classification. With SSL, you can enjoy:

1. 🔍 **Meaningful Feature Extraction**: Extract valuable features from cell images **without needing any class labels**!
2. 🔄 **Efficient Knowledge Transfer**: Seamlessly transfer knowledge between **bone marrow** and **blood cell** domains.
3. 📊 **Easy Adaptation**: Adapt your models to new datasets with **minimal labeled data**!

## Results

This study highlights the power of self-supervised learning (SSL) for hematological cell classification and cross-domain adaptation with minimal labeled data. For the Matek blood dataset, our SSL model trained on bone marrow cells achieved state-of-the-art results, showing that direct model transfer is possible without dataset-specific training. For the other datasets, we achieved **state-of-the-art results** with significantly fewer labels and much less computational power. The results are shown in the figure below, which displays **balanced accuracy** for the domain adaptation experiments as a function of the number of samples per class used for classifier fitting. The data covers three blood datasets and various ML classifiers: **Support Vector Machine (SVM)**, **Logistic Regression (LR)**, and **k-nearest neighbours (KNN)**. The **SSL feature extractor**, trained on the Matek BM dataset, remained the same throughout. The **benchmark line** represents the accuracy reported in the original dataset-specific publications using supervised end-to-end deep learning.
![Classification balanced accuracy, blood datasets](imgs/balanced_acc_domain_adaption.png "Balenced Acc for all Blood datasets")
The two-dimensional embedding of Acevedo blood cell images, computed using UMAP, shows clear clustering of the eight cell classes. These features were extracted using the **SSL model** trained on the **Matek bone marrow dataset** without class labels. This demonstrates that **SSL training on bm effectively captures general cell features of blood cells**, even across different staining and scanning conditions.
![Classification balanced accuracy, blood datasets](imgs/Fig_6_UMAP_Acevedo.png "Balenced Acc for all Blood datasets")

The following figure shows the **attention heads** of an **SSL-trained (a)** and a **supervised-trained (b)** bone marrow model on a representative cell image. It is evident that the **SSL feature extractor** effectively focuses on the cell, capturing essential features, while the **supervised model** directs its focus to less relevant areas, indicating potential **overfitting** to the training dataset.
![Classification balanced accuracy, blood datasets](imgs/Fig_5_Attention_heads.png "Balenced Acc for all Blood datasets")

## How to use

### 1. Clone the repository

```bash
git clone https://github.com/IPMI-ICNS-UKE/cell-classification.git
```

### 2. Create a Conda environment and install dependencies

   ```bash
   conda create --name bloodcell python=3.10
   conda activate bloodcell
   cd cell-classification
   pip install -r requirements.txt
   ```

### 2. Download Data Sets

The following data sets were used in this study:
![Classification balanced accuracy, blood datasets](imgs/Fig_1_Overview.png "Balenced Acc for all Blood datasets")

- **Bone Marrow (BM) data set:** A highly unbalanced data set of single bone marrow cell microscopy images that can be found [here](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=101941770) and is provided by [Matek et al. (2021)](https://doi.org/10.1182/blood.2020010568).
- **Blood Matek data set:** **Matek Blood dataset:** A single-cell morphological dataset of leukocytes from Acute Myeloid Leukemia (AML) patients and non-malignant controls. The dataset can be accessed [here](https://www.cancerimagingarchive.net/collection/aml-cytomorphology_lmu) and is provided by [Matek et al. (2024)](https://www.cancerimagingarchive.net/collection/aml-cytomorphology_lmu).
- **Blood Acevedo data set:** A dataset of microscopic peripheral blood cell images designed for the development of automatic recognition systems that can be found [here](https://data.mendeley.com/datasets/snkd93bnjr/draft?a=d9582c71-9af0-4e59-9062-df30df05a121) and is provided by [Acevedo et al. (2019)](https://doi.org/10.1016/j.cmpb.2019.105020).
- **Blood Raabin data set:** **Raabin-WBC dataset:** **Raabin-WBC dataset:** A large dataset of white blood cells containing cell locations and types, along with segmented nuclei and cytoplasm covering five types of peripheral blood cells: neutrophil, eosinophil, basophil, lymphocyte, and monocyte. The dataset can be found [here](https://www.raabindata.com/free-data/) and is provided by [Kouzehkanan et al. (2022)](https://doi.org/10.1038/s41598-022-05049-0).

### 3. Download trained SSL Feature Extractor

Download our feature extractor [here](https://zenodo.org/records/13969212?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImM4ZGMyYTQ4LTNkYWQtNDM2Yy1hMTU2LWFkNjk0YzRlN2FjZCIsImRhdGEiOnt9LCJyYW5kb20iOiI3YWYyZTBmOTRiYTRkOGEzNmI0MzQyMzcxZTRhZGM5MyJ9.2sOEGDg1rX2eJVbio8oZdree4Usg5ZH_GbwTAQjOQR0r04xjP49lkO_ZkVINm4S23qyDKQ5BdkDyVG4X71ju2A)from Zenodo! 

### 4.  Fit & Evaluate Your Blood Cell Dataset: Personal or Public

You can use this script to classify your own blood cell images or to evaluate the three open available blood cell image datasets.

```bash
python eval_ssl.py --dataset_fit blood_matek --dataset_eval blood_matek --device cuda:0 --train True --batch_size 4 --teacher_momentum 0.9995 --model xcit_small_12_p8_224_dist --save_path /path/to/save --n_iter 1 --classifier ALL
```

- `--dataset_fit` : Select the dataset for training (options: `blood_acevedo`, `blood_matek`, `blood_raabin`, `bm`). Type: `string`. Default: `blood_matek`.
- `--dataset_eval` : Select the dataset for evaluation (options: `blood_acevedo`, `blood_matek`, `blood_raabin`, `bm`). Type: `string`. Default: `blood_matek`.
- `--device` : Specify which GPU to use (e.g., `cuda:0` for the first GPU). Type: `string`.
- `--train` : Set to `True` to train the model or `False` to evaluate an existing model. Type: `bool`. Default: `True`.
- `--path_to_weights` : Provide the path to the model weights (required if `--train` is `False`). Type: `string`.
- `--batch_size` : Set the batch size for training/evaluation. Type: `int`. Default: `4`.
- `--teacher_momentum` : Set the momentum parameter that influences training dynamics (especially useful for larger batch sizes). Type: `float`. Default: `0.9995`.
- `--model` : Choose the model to use from the timm library. Type: `string`. Default: `xcit_small_12_p8_224_dist`.
- `--save_path` : Specify the directory path where results should be stored. Type: `string`.
- `--max_train_image_number` : Limit the number of images used for training in self-supervised learning (optional). Type: `int`.
- `--n_iter` : Specify how many times the classifier is fitted using random images per class. Type: `int`. Default: `1`.
- `--n_sample_eval` : Specify the number of samples for evaluation; if set to `0`, it evaluates using all available samples (optional). Type: `int`.
- `--classifier` : Choose which classifiers to use: `SVC`, `LR`, `KNN`, or `ALL`. Type: `string`. Default: `ALL`.

#### Experiment Overview 🔬

We conducted several experiments, which can also be replicated using the provided `eval_ssl.py` script. The **SSL feature extractor** was trained on the **Matek BM dataset**, and based on the extracted features, lightweight ML classifiers were trained. Two types of experiments were performed:

1. **Direct Model Transfer**: Using the features from the BM dataset.
2. **Domain Adaptation**: Using a small amount of labeled data from the target dataset.

The classification performance was then evaluated using the test data of the target datasets.

![Classification balanced accuracy, blood datasets](imgs/Fig_3_Experiments_bm.png "Balenced Acc for all Blood datasets")

### 5. Retrain SSL feature extractor

```bash
python train_ssl.py --dataset bm --device cuda:0 --train True --batch_size 4 --teacher_momentum 0.9995 --model xcit_small_12_p8_224_dist --save_path /path/to/save/results
```

- `--dataset`: Select the dataset for training (options: `blood_acevedo`, `blood_matek`, `blood_raabin`, `bm`).
- `--device`: Specify which GPU to use (e.g., `cuda:0` for the first GPU).
- `--train`: Set to `True` to train the model or `False` to evaluate an existing model.
- `--path_to_weights`: Path to pre-trained weights (required if `train=False`).
- `--batch_size`: Batch size for training/evaluation (default: 4).
- `--teacher_momentum`: Momentum for the teacher model (default: 0.9995).
- `--model`: Specify the model architecture from the `timm` library (default: `xcit_small_12_p8_224_dist`).
- `--save_path`: Path to the directory where results will be saved.
- `--max_train_image_number`: Limit the number of images used for training (optional).
- `--n_iter`: Number of iterations for fitting the classifier (optional).
- `--n_sample_eval`: Number of samples used for evaluation (default: 50).
- `--classifier`: Choose the type of classifier for evaluation (options: `SVC`, `LR`, `KNN`, `ALL`).
