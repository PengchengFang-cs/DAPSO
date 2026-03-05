# DAPSO
Implementation of DAPSO for MRI reconstruction, built on top of the MambaRecon codebase.

Paper: coming soon

Main figure (PDF): [dapso-main-fixed.drawio.pdf](dapso-main-fixed.drawio.pdf)


## Installation
Clone the repository:
```
git clone git@github.com:PengchengFang-cs/DAPSO.git
```
Create the environment from the environment.yml:
```
conda env create -f environment.yml
```
Activate the environment:
```
conda activate mamba_recon_env
```
Install causal convolution and mamba packages:
```
cd causal-conv1d

python setup.py install
```

```
cd mamba

python setup.py install
```


## Dataset

This project uses two public MRI reconstruction datasets:
- fastMRI
- CC359

Please organize dataset files locally according to the paths expected by the dataloaders in `code/dataloaders/`.


## Run Commands
```
python train.py --exp mamba_unrolled --dataset ixi --model mamba_unrolled --patch_size 2 --batch_size 4 --gpu_id 0
```


## Citation
You are encouraged to modify/distribute this code. However, please acknowledge this code and cite the paper appropriately.
```
@InProceedings{Korkmaz_2025_WACV,
    author    = {Korkmaz, Yilmaz and Patel, Vishal M.},
    title     = {MambaRecon: MRI Reconstruction with Structured State Space Models},
    booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
    month     = {February},
    year      = {2025},
    pages     = {4142-4152}
}
```

## Contact

ykorkma1[at]jhu.edu


## Acknowledgements
We gratefully acknowledge the authors of the following repositories, from which we utilized code in our work:  

- [Mamba-UNet](https://github.com/ziyangwang007/Mamba-UNet/tree/main)  
- [VMamba](https://github.com/MzeroMiko/VMamba)  
