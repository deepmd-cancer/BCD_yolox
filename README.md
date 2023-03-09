## External Data
* Download CBIS-DDSM from here: https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=22516629
* Download CMMD from here: https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70230508
* Download VinDr-Mammo from here: https://physionet.org/content/vindr-mammo/1.0.0
* Download Mini-DDSM from here: https://www.kaggle.com/datasets/cheddad/miniddsm2
* Run ./BCD/preprocessed/external/preprocessed_*.py to obtain the CSV of external datasets.
* Run ./BCD/preprocessed/external/*_dcm2png.ipynb to convert datasets from DCM to PNG format. Please note that the mini-DDSM dataset is already in PNG format, so there is no need to run the ipynb.
* We have also open-sourced the preprocessed datasets. The final results can be downloaded directly. The link is as follows:
https://www.kaggle.com/datasets/kevin1742064161/bcd-dataset

## YOLOX Training & Infer
* We have open-sourced the training code of yolox. The link is as follows: https://www.kaggle.com/datasets/kevin1742064161/yolo-x
* Download the link at the end of the last section and place it in the './YOLOX/datasets' directory of the yolox project.

step 1. Install YOLOX from source.
```shell
cd YOLOX
pip3 install -v -e .  # or  python3 setup.py develop
```

step 2. Training BCD_1k.
```shell
python tools/train.py -expn m_1k -n yolox-m -f exps/example/yolox_voc/yolox_voc_m.py -d 0 -b 32 --fp16 -c ./weights/yolox_m.pth --logger tensorboard
```

step 3. Infer on official datasets.
```shell
python tools/demo.py image -f exps/example/yolox_voc/yolox_voc_m.py -c YOLOX_outputs/m_1k/best_ckpt.pth --path I:/DATA/BCD_VOC/JPEGImages --conf 0.1 --nms 0.45 --tsize 640 --save_result --device gpu -expn BCD
```

step 4. Use the detection boxes obtained in step 3 for pseudo-label training.
```shell
python tools/train.py -expn BCD_nano -n nano_all -f exps/example/yolox_voc/yolox_voc_nano.py -d 0 -b 64 -c ./weights/yolox_nano.pth --logger tensorboard
python tools/train.py -expn s_all -n yolox-s -f exps/example/yolox_voc/yolox_voc_s.py -d 0 -b 64 --fp16 -c ./weights/yolox_s.pth --logger tensorboard
python tools/train.py -expn x_all -n yolox-x -f exps/example/yolox_voc/yolox_voc_x.py -d 0 -b 64 --fp16 -c ./weights/yolox_x.pth --logger tensorboard
```

step 5. Infer on external datasets.
```shell
python tools/demo.py image -f exps/example/yolox_voc/yolox_voc_x.py -c YOLOX_outputs/x_all/best_ckpt.pth --path I:/DATA/vindr_YOLO --conf 0.1 --nms 0.45 --tsize 640 --save_result --device gpu -expn Vindr
python tools/demo.py image -f exps/example/yolox_voc/yolox_voc_x.py -c YOLOX_outputs/x_all/best_ckpt.pth --path G:/DATA/CMMD/CMMD_yolo --conf 0.1 --nms 0.45 --tsize 640 --save_result --device gpu -expn CMMD
python tools/demo.py image -f exps/example/yolox_voc/yolox_voc_x.py -c YOLOX_outputs/x_all/best_ckpt.pth --path G:/DATA/CBIS-DDSM/CBIS-DDSM_yolo --conf 0.1 --nms 0.45 --tsize 640 --save_result --device gpu -expn CBIS-DDSM
python tools/demo.py image -f exps/example/yolox_voc/yolox_voc_x.py -c YOLOX_outputs/x_all/best_ckpt.pth --path G:/DATA/MINI_DDSM/images --conf 0.1 --nms 0.45 --tsize 640 --save_result --device gpu -expn MINI_DDSM
```
step 6. Rerun ./BCD/preprocessed/external/*_dcm2png.ipynb and incorporate YOLOX prediction results in order to crop ROI.

step 7. We have also open-sourced the official datasets and external datasets after ROI crop. The final results can be downloaded directly. The links are as follows: 
* https://www.kaggle.com/datasets/kevin1742064161/bcd-external-datasets
* https://www.kaggle.com/datasets/kevin1742064161/yolx-nano-fold0-data
* https://www.kaggle.com/datasets/kevin1742064161/yolx-nano-fold1-data
* https://www.kaggle.com/datasets/kevin1742064161/yolx-nano-fold2-data
* https://www.kaggle.com/datasets/kevin1742064161/yolx-nano-fold3-data

## Image-Level Model
step 1. Download and merge above five datasets link.

step 2. Run ./BCD/preprocessed/make_fold_machine_dropnoise.py to get training csv. 

step 3. Model_machine
```shell
cd ./BCD/
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python src/train_machine_onehot_mix.py --in_base_dir /home/ubuntu/Projects/dataset --exp_name 0215_tfeffnetv2_machine_onehot --config_path ./config/0215_tfeffnetv2_machine_onehot.yaml --save_checkpoint --wandb_logger
```

step 4. Model_meta
```shell
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python src/train_meta_onehot_mix.py --in_base_dir /home/ubuntu/Projects/dataset --exp_name 0215_tfeffnetv2_meta --config_path ./config/0215_tfeffnetv2_meta.yaml --save_checkpoint --wandb_logger
```

step 5. Model_convnext
```shell
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python src/train_v1213.py --in_base_dir /home/ubuntu/Projects/dataset --exp_name convnext_nano_1536960 --config_path ./config/convnext_nano_1536960.yaml --save_checkpoint --wandb_logger
```

## CNN+LSTM Model
Run ./BCD/rsna_lstm_models/*.ipynb
