# MM-Retinal: Knowledge-Enhanced Foundational Pretraining with Fundus Image-Text Expertise

Read our arXiv Paper [[ArXiv]()\] &nbsp; Download our Dataset[[:hugs:]()\] &nbsp; Cite our work[[BibTeX]()\]

by [Ruiqi Wu](), Chenran Zhang, Jianle Zhang, Yi Zhou, Tao Zhou and Huazhu Fu in xxx.

We propose **MM-Retinal**, a multi-modal dataset that encompasses high-quality image-text pairs collected from professional fundus diagram books. Moreover, we present a novel Knowledge-enhanced foundational pretraining model based on MM-Retinal which incorporates Fundus Image-Text expertise, called **KeepFIT**. 

Our proposed fundus foundation model achieves **state-of-the-art** performance across **six** unseen downstream tasks. KeepFIT holds the generalization ability in zero-shot and few-shot scenarios. 

## :rocket: Updates
* **[2024.5.13]** We are excited to release : :white_check_mark:[dataset]() for ***MM-Retinal***!
* **[2024.5.13]** We are excited to release : :white_check_mark:[training/evaluation code](), :white_check_mark: [new checkpoints](), and :white_check_mark: [comprehensive readmes]() for ***KeepFIT***!
  
## :sunflower: Data Collection and Statistics

The four professional fundus diagram books are 
* [《图解眼底病》(Diagram of Ocular Fundus Diseases)](https://drive.google.com/file/d/1ml-qdHOVKYXFHfm4UE7sxkQJyJHWmDqU/view?usp=drive_link)
* [《眼底病图谱》(Atlas of Ocular Fundus Diseases)](https://drive.google.com/file/d/15KNdgjjeM13EgrVtNzNA_-y_M2GDfnI6/view?usp=drive_link)
* [《眼底病影像诊断图谱第2版》](https://drive.google.com/file/d/1VZueUH4rW8wTnkKD7bBPMn3OqjnPMiYR/view?usp=drive_link)
* [《Color Atlas and Synopsis of Clinical Ophthalmology》](https://drive.google.com/file/d/1XdOpS2-CbuvdcCZl9O3VJU2yuZlEUA34/view?usp=drive_link)
  
Our designed semi-automatic pipeline of dataset construction contains four steps:
* **Step 1:** Image-Text Pair Collection
* **Step 2**: Image-Text Alignment
* **Step 3**: Modality Classification
* **Step 4**: Text Cleaning and Bilingual Translation

A six-person team took four weeks to get MM-Retinal completed.

<img src=./figures/data.png width="80%">

## :rainbow: Download MM-Retinal

* **[MM-Retinal v1]():** Current version of MM-Retinal dataset includes 2,169 CFP cases, 1,947 FFA cases and 233 OCT cases. Each case is provided with an image and texts in both English and Chinese.

## :palm_tree: Usage
Simply set up the required environment as following:
```bash
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install transformers=4.28.1, sentencepiece, datasets
```

## Quick Start
Check `simple_test.py` for quickly use PMC-LLaMA or you can follow this folowing simple sample.

<!-- ```python
import transformers
import torch
tokenizer = transformers.LlamaTokenizer.from_pretrained('axiong/PMC_LLaMA_13B')
model = transformers.LlamaForCausalLM.from_pretrained('axiong/PMC_LLaMA_13B')
model.cuda()  # move the model to GPU

prompt_input = (
    'Below is an instruction that describes a task, paired with an input that provides further context.'
    'Write a response that appropriately completes the request.\n\n'
    '### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:'
)

example = {
    "instruction": "You're a doctor, kindly address the medical queries according to the patient's account. Answer with the best option directly.",
    "input": (
        "###Question: A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. "
        "She states it started 1 day ago and has been worsening despite drinking more water and taking cranberry extract. "
        "She otherwise feels well and is followed by a doctor for her pregnancy. "
        "Her temperature is 97.7°F (36.5°C), blood pressure is 122/77 mmHg, pulse is 80/min, respirations are 19/min, and oxygen saturation is 98% on room air."
        "Physical exam is notable for an absence of costovertebral angle tenderness and a gravid uterus. "
        "Which of the following is the best treatment for this patient?"
        "###Options: A. Ampicillin B. Ceftriaxone C. Doxycycline D. Nitrofurantoin"
    )
}
input_str = [prompt_input.format_map(example)]

model_inputs = tokenizer(
    input_str,
    return_tensors='pt',
    padding=True,
)
print( f"\033[32mmodel_inputs\033[0m: { model_inputs }" )


topk_output = model.generate(
    model_inputs.input_ids.cuda(),
    max_new_tokens=1000,
    top_k=50
)
output_str = tokenizer.batch_decode(topk_output)
print('model predict: ', output_str[0])
``` -->


## Training

<!-- The training process can be divided as two phases: pretrain and instruction-tuning.


The script for pretraining locates at `Pretrain/training.sh`.

Our pretraining dataset sources from [S2ORC](https://github.com/allenai/s2orc). Only those papers with PubMed IDs are deemed as medical-related and used during pretraining.
<!-- The raw training data can be dowloaded from [S2ORC](https://github.com/allenai/s2orc), filter out the papers with PubmedCentral IDs, and you can get the training data we use.  -->

The book is listed in this repo as [MedicalBook.xlsx](https://github.com/chaoyi-wu/PMC-LLaMA/blob/main/MedicalBook.xlsx), due to licenses, we cannot release raw content. For reproducing, pls buy and process the books.

More details about how to fine-tune LLaMA can refer to [Finetune_LLAMA](https://github.com/chaoyi-wu/Finetune_LLAMA) -->


## Results

### Finetune
<!-- | Method              | Model Size          | USMLE | MedMCQA | PubMedQA |
|---------------------|---------------------|------------------|--------------|------------------|
| Human (pass)        | -                   | 50.0            | --            | 60.0           |
| Human (expert)      | -                   | 87.0            | 90.0         | 78.0           |
| ChatGPT             | 175B                | **57.0**        | 44.7         | 63.9           |
| LLaMA-2             | 13B                 | 42.73           | 37.41        | 68.0           |
| LLaMA-2             | 70B                 | 43.68           | 35.02        | 74.3           |
| Med-Alpaca          | 13B                 | 30.85           | 31.13        | 53.2           |
| Chat-Doctor         | 7B                  | 33.93           | 31.10        | 54.3           |
| PMC_LLaMA_13B ![](./figures/new.gif) | 13B | **56.36**   | **56.04**  | **77.9**  | -->


<!-- Note that, the manual and zero-shot results with * are referred from [LMFLow](https://github.com/OptimalScale/LMFlow/tree/main/src/lmflow). -->

### Few-shot

### Zero-shot

<!-- We demonstrate PMC_LLaMA_13B's responses with out of domain queries. -->



## Acknowledge
FLAIR -- https://github.com/jusiro/FLAIR

## Citation
If you find this repository useful, please consider citing this paper:
```
@article{
}
```

## Contact
If you have any question, please feel free to contact ruiqiwu@seu.edu.cn.