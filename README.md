# In-Model-Merging

The code repository for paper "In-Model Merging for Enhancing the Robustness of Medical Imaging Classification Models".

## Environment Setup
### Create conda env

```
conda create --name in_model_merging python=3.10 -y
conda activate in_model_merging
```

### Install dependencies

```
pip install -r requirements.txt
```

If the cuda is not compatible with the pytorch installation, try either of the following commands:

```
conda install cudatoolkit=11.0 -c pytorch
pip install --pre torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

## Dataset preparation

```
python nih_batch_download_zips.py
```

## Model training and evaluation
### Model training

For model backbone training, the commandline is:

```
bash run.sh [GPU id]
```

For instance:

```commandline
bash run.sh 0
```

For In-Model-Merging finetuning, you would first modify the ```backbone = True``` into False and put the correct path of trained backbone to load_ckpt, then start ```bash run.sh 0```.

### Model evaluation

For model evaluation, you would first modify the ```is_train = True``` into False and put the correct path of trained model to load_ckpt, then start ```bash run.sh 0```.

## Acknowledgement

If you got a chance to use our code, you can cite us!

```
@article{wang2025model,
  title={In-Model Merging for Enhancing the Robustness of Medical Imaging Classification Models},
  author={Wang, Hu and Almakky, Ibrahim and Ma, Congbo and Saeed, Numan and Yaqub, Mohammad},
  journal={arXiv preprint arXiv:2502.20516},
  year={2025}
}
```

Enjoy!!
