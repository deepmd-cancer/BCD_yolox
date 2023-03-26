

## lightning Training

step 1. Install PyTorch2.0.
```shell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

step 2. Install PyTorch_lightning
```shell
pip install https://github.com/Lightning-AI/lightning/archive/refs/heads/master.zip -U
```
step 3. Install my own timm.
```shell
cd ./packages/pytorch-image-models
pip install -v -e .
```

step 4. Install others.
```shell
pip install sentence_transformers
pip install albumentations
pip install pandas
```

step 5. Training
```shell
PYTHONPATH=. python src/train.py --in_base_dir /root/autodl-tmp/DATASET/ --exp_name convnext_xxlarge_102w_stage1_320 --config_path ./config/CLIP_ConvNext-XXLarge_320.yaml --save_checkpoint --wandb_logger

