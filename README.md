# MM-Retinal: Knowledge-Enhanced Foundational Pretraining with Fundus Image-Text Expertise

Paper [[ArXiv](https://arxiv.org/abs/2405.11793)\] &nbsp; Dataset[[Google Drive](https://drive.google.com/drive/folders/177RCtDeA6n99gWqgBS_Sw3WT6qYbzVmy?usp=drive_link)\] &nbsp; Code[[Github](https://github.com/lxirich/MM-Retinal)\]

by Ruiqi Wu, Chenran Zhang, Jianle Zhang, Yi Zhou, Tao Zhou and Huazhu Fu in **MICCAI 2024**!

We propose **MM-Retinal**, a multi-modal dataset that encompasses high-quality image-text pairs collected from professional fundus diagram books. 

Moreover, we present a novel Knowledge-enhanced foundational pretraining model based on MM-Retinal which incorporates Fundus Image-Text expertise, called **KeepFIT**. 

## :rocket: Updates
* **[2024.5.21]** We are excited to release : :white_check_mark:[dataset](https://drive.google.com/drive/folders/177RCtDeA6n99gWqgBS_Sw3WT6qYbzVmy?usp=drive_link) and :white_check_mark:[data collection code](https://github.com/lxirich/MM-Retinal/tree/main/Dataset_Collection) for ***MM-Retinal***!
* **[2024.5.21]** We are excited to release : :white_check_mark:[training/evaluation code](https://github.com/lxirich/MM-Retinal/tree/main/KeepFIT), :white_check_mark: [new checkpoints](https://drive.google.com/drive/folders/1hPDt9noBnlL75mBpNKtfaYmTr4oQJBjP?usp=drive_link) for ***KeepFIT***!
  
## :sunflower: Data Collection and Statistics

The four professional fundus diagram books are 
* [book1:《图解眼底病》(Diagram of Ocular Fundus Diseases)](https://drive.google.com/file/d/1ml-qdHOVKYXFHfm4UE7sxkQJyJHWmDqU/view?usp=drive_link)
* [book2:《眼底病影像诊断图谱第2版》](https://drive.google.com/file/d/1VZueUH4rW8wTnkKD7bBPMn3OqjnPMiYR/view?usp=drive_link)
* [book3:《眼底病图谱》(Atlas of Ocular Fundus Diseases)](https://drive.google.com/file/d/15KNdgjjeM13EgrVtNzNA_-y_M2GDfnI6/view?usp=drive_link)
* [book4:《Color Atlas and Synopsis of Clinical Ophthalmology》](https://drive.google.com/file/d/1XdOpS2-CbuvdcCZl9O3VJU2yuZlEUA34/view?usp=drive_link)
  
Our designed semi-automatic pipeline of dataset construction contains four steps:
* **Step 1:** Image-Text Pair Collection
* **Step 2**: Image-Text Alignment
* **Step 3**: Modality Classification
* **Step 4**: Text Cleaning and Bilingual Translation

A six-person team took four weeks to get MM-Retinal completed.

<img src=./figures/data.png width="80%">

## :rainbow: Download Pre-training Datasets

* **[MM-Retinal v1(CFP+FFA+OCT)](https://drive.google.com/drive/folders/177RCtDeA6n99gWqgBS_Sw3WT6qYbzVmy?usp=drive_link):** Current version of MM-Retinal dataset includes 2,169 CFP cases, 1,947 FFA cases and 233 OCT cases. Each case is provided with an image and texts in both English and Chinese.
* **[flair(CFP)](https://github.com/jusiro/FLAIR/blob/main/readme.md):** compiles 37 open-access fundus image datasets covering 96 categories with up to 284,660 images. These datasets provide category-level labels for classification.
* **[SynFundus-1M(CFP)](https://github.com/parap1uie-s/SynFundus-1M):** is a synthetic dataset with 1 million images for 14 diseases, created by a diffusion model trained on 1.3 million private fundus images.
* **[FFA-IR(FFA)](https://github.com/mlii0117/FFA-IR):** provides 10,790 reports along with 1,048,584 images from clinical practice. It includes a schema of 46 categories of lesion and bilingual reports.

## :palm_tree: Quick Start
### 1. Environment
Clone the whole repository and install the dependencies.

- Python 3.8.18
- PyTorch 1.13.1
- cuda 12.0

```bash
conda create -n mmretinal python=3.8
conda activate mmretinal

git clone https://github.com/lxirich/MM-Retinal.git
cd MM-Retinal/KeepFIT/KeepFIT-CFP or cd MM-Retinal/KeepFIT/KeepFIT-FFA
pip install -r requirements.txt
```

### 2. Training

For color fundus photography (CFP) modality:

* Define the relative paths for pre-training datasets and dataframes in `./local_data/constants.py`.

* Prepare the pre-training dataset dataframes in `./local_data/prepare_partitions.py`.

```
python main_pretrain.py --epochs 40 --batch_size 24 --num_workers 4
```

For fundus fluorescein angiography (FFA) modality:

* directly run our code by the following.
  
```python
python main.py
```

### 3. Evaluation

For color fundus photography (CFP) modality:

* Define the relative paths for evaluation datasets and dataframes in `./local_data/constants.py`.
  
* Finetune
    ```
    python main_transferability.py --shots_train 80% --shots_test 20% --folds 5 --experiment 08_ODIR200x3 --method lp --domain_knowledge True -- project_features False 
    ```
* Few-shot
    ```
    python main_transferability.py --shots_train 5 --shots_test 20% --folds 5 --experiment 08_ODIR200x3 --method clipAdapter --domain_knowledge True -- project_features True
    ```
* Zero-shot
    ```
    python main_transferability.py --shots_train 0% --shots_test 100% --experiment 08_ODIR200x3 --method zero_shot --domain_knowledge True -- project_features True 
    ```
For fundus fluorescein angiography (FFA) modality:

* Validation and testing are automatically implemented in each epoch.

## :telescope: Results

### 1. Finetune

<img src=./figures/finetune_CFP.png width="80%">

<img src=./figures/finetune_FFA.png width="80%">

### 2. Few-shot and Zero-shot

<img src=./figures/few_zero.png width="80%">

### 3. Ablation Study

<img src=./figures/ablation.png width="80%">

## :dart: Checkpoints
|            Model             |Checkpoint|
|------------------------------|---------:|
| KeepFIT (flair+MM)           |  [Link](https://drive.google.com/file/d/1w6poCkZeqSTHsLYz1R9-z5Ttc0Vw0qSw/view?usp=drive_link)|
| KeepFIT (50%flair+MM)        |  [Link](https://drive.google.com/file/d/1o4EDSifmcN7cKDP5w5qvS_wWiyfEZ6UN/view?usp=drive_link)|
| KeepFIT (FFA-IR+MM)          |  [Link](https://drive.google.com/file/d/1fdCRDbKJKZcqBlmdEdETygZ8EsBu5QlV/view?usp=drive_link)|
| Image_captioning (FFA-IR+MM)          |  [Link](https://drive.google.com/file/d/1FyoSgLKHgm1mRSZp-ZJN5jEhAf6Z4Eal/view?usp=drive_link)|


## :cupid: Acknowledge
FLAIR -- https://github.com/jusiro/FLAIR

FFA-IR -- https://github.com/mlii0117/FFA-IR

SynFundus-1M -- https://github.com/parap1uie-s/SynFundus-1M

## :star2: Citation
If you find this repository useful, please consider citing this paper:
```
@article{
}
```

## :mailbox_with_mail: Contact
If you have any question, please feel free to contact ruiqiwu@seu.edu.cn.