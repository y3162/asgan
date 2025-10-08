## 🚀仮想環境の作成
```
$ conda create --prefix ./asgan_env python=3.10 -y
$ conda activate ./asgan_env
$ conda install pytorch=1.11 torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia -y
$ conda install tensorboard -y
$ conda install -c conda-forge omegaconf hydra-core fairseq librosa matplotlib fastprogress pandas scikit-learn -y
$ pip install deepspeed
$ conda install -c conda-forge libstdcxx-ng -y
$ pip install "pip<24.1"
$ pip install fairseq==0.12.2
```
または
```
$ conda env create --prefix ./asgan_env -f ./environment.yaml
```

## 📁データセットのダウンロード
```
import torchaudio
  dataset = torchaudio.datasets.SPEECHCOMMANDS(
  root="/path/to/database",
  download=True,
)
```

## 📊前処理
```
$ export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
$ python 01_extract_hubert_features.py \
  --dataset-path=/path/to/dataset \
  --output-path=./data/
```

## 🤖モデルの学習
```
$ python train_asgan.py \
model=rp_w \
train_root=./data/ \
n_valid=400 \
data_type=hubert_L6 \
checkpoint_path=./checkpoints/ \
z_dim=512 \
rp_w_cfg.z_dim=512 \
rp_w_cfg.w_layers=1 \
batch_size=16 \
lr=2e-3 \
grad_clip=10 \
aug_init_p=0.2 \
stdout_interval=100 \
validation_interval=2500 \
n_epochs=800 \
c_dim=768 \
rp_w_cfg.c_dim=768 \
d_lr_mult=0.1 \
fp16=True \
preload=False \
num_workers=12 \
betas=[0,0.99] \
rp_w_cfg.equalized_lr=True \
rp_w_cfg.use_sg3_ff=True \
rp_w_cfg.D_kernel_size=5 \
rp_w_cfg.D_block_repeats=[3,3,3,3] \
use_sc09_splits=True \
sc09_train_csv=./splits/train.csv \
sc09_valid_csv=./splits/valid.csv \
rp_w_cfg.r1_gamma=0.1
```
