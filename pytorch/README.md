# BentoML with PyTorch

---

## Install Package
- Using Package
    - torch==1.10.2 
    - torchvision==0.10.2
    - torchaudio==0.11.3

``` bash
# Create environment
conda create -y -n {ENV_NAME} python=3.8
conda activate {ENV_NAME}

# CPU mode
pip install torch==1.10.2+cpu torchvision==0.11.3+cpu torchaudio==0.10.2+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
pip install -r requirements.txt

# GPU mode
# Please execute after installing cuda11
conda install -y -c pytorch cudatoolkit=11.3 # if using conda
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install -r requirements.txt
```


## Training Example 

### Classification
1. Download example dataset 
    - Flower: https://drive.google.com/file/d/1QQs4L7sud6E5JcCeY7sbdyZJRpnx315-/view?usp=sharing
    - Cats & Dogs: https://drive.google.com/file/d/1uIWfyF8R6-WeumTSU4M0YuYw2X_QIfn8/view?usp=sharing
    - Dog Breed: https://drive.google.com/file/d/14FUyv7TzRq7T0r-ouwB9sdYOTPG22HIX/view?usp=sharing
2. Unzip dataset in `data` directory
3. Check dataset setting
```bash
data
└── {Dataset name}
    ├── train
    │    ├── {class name}
    │    │     └── image files
    │    └── {class name}
    │          └── image files
    └── validation
         ├── {class name}
         │     └── image files
         └── {class name}
               └── image files
   



```
4. Edit `config.yaml`
5. Execute command `python train.py --config config.yaml`

## Serving Model
1. Create `service.py`  

2. Serve model
```bash
bentoml serve service.py:svc --reload
```


## Build for deployment
1. Create `bentofile.yaml`
```yaml
service: "service:svc"
description: "file: ./README.md"
labels:
  owner: jjerry-k
  stage: pytorch example
include:
- "service.py"
- "model"
- "utils"
# log/{DATASET_NAME}/{YYYY_MM_DD}/{hh_mm_ss}/config.yaml
- "log/flower_photos/2022_02_18/18_18_10/config.yaml"

python:
  packages:
    - torch
    - timm
    - Pillow
    - PyYAML
```

2. Build `bento`
``` bash
bentoml build
```

After that, you can serve model this way
```bash
# bentoml serve {Bento Name}:latest --production
bentoml serve flower_photos_service:latest --production
```
## Containerize service

```bash
# bentoml containerize {Bento Name}:latest
bentoml containerize flower_photos_service:latest
```

## Serve using docker
```bash
# In use cpu
# docker run --rm -p 5000:5000 {Bento Name}:{Tag}
docker run --rm -p 5000:5000 flower_photos_service:h6f4gluqys5b7vlm

# If use gpu
# docker run --gpus '"device={number}"' --rm -p 5000:5000 {Bento Name}:{Tag}
docker run --gpus '"device=1"' --rm -p 5000:5000 flower_photos_service:h6f4gluqys5b7vlm
```