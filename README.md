# Spatio-Temporal Deformable Attention Network for Video Deblurring (ECCV2022)
PyTorch codes for "Spatio-Temporal Deformable Attention Network for Video Deblurring (ECCV2022)"
[Paper](https://arxiv.org/abs/2207.10852) | [Project Page](https://vilab.hit.edu.cn/projects/stdan)

![Overview](https://vilab.hit.edu.cn/projects/stdan/images/STDAN-Overview.png)

## Datasets

We use the [GoPro](https://github.com/SeungjunNah/DeepDeblur_release), [DVD](http://www.cs.ubc.ca/labs/imager/tr/2017/DeepVideoDeblurring/) and [BSD](https://github.com/zzh-tech/ESTRNN) datasets in our experiments, which are available below:

- [GoPro](https://drive.google.com/file/d/1y4wvPdOG3mojpFCHTqLgriexhbjoWVkK/view?usp=sharing)
- [DVD](https://www.cs.ubc.ca/labs/imager/tr/2017/DeepVideoDeblurring/DeepVideoDeblurring_Dataset.zip)
- [BSD](https://drive.google.com/file/d/19cel6QgofsWviRbA5IPMEv_hDbZ30vwH/view?usp=sharing)


## Pretrained Models

You could download the pretrained model from [here](https://drive.google.com/drive/folders/1ysVyLbw_phxAu5bJVNVSMH4ccFLTZhVp?usp=sharing) and put the weights in [weights folder](weights). 

## Prerequisites
#### Clone the Code Repository

```
git clone https://github.com/huicongzhang/STDAN.git
```
### Install Pytorch Denpendencies

```
conda create -n STDAN python=3.7 
conda activate STDAN
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=10.2 -c pytorch
```

#### Install Python Denpendencies and Build PyTorch Extensions

```
cd STDAN
sh install.sh
```

## Get Started


To train STDAN, you can simply use the following command:

```
python runner.py --data_path=/yourpath/DeepVideoDeblurring_Dataset/quantitative_datasets --data_name=DVD --phase=train
```

To test STDAN, you can simply use the following command:
    
```
python runner.py --data_path=/yourpath/DeepVideoDeblurring_Dataset/quantitative_datasets --data_name=DVD --phase=test --weights=./weights/DVD_release.pth 
```

In [here](config.py), there are more settings of testing and training. 

Some video results are shown in [here](https://vilab.hit.edu.cn/projects/stdan)

## Cite this work

```
@inproceedings{zhang2022spatio,
    title={Spatio-Temporal Deformable Attention Network for Video Deblurring},
    author={Zhang, Huicong and Xie, Haozhe and Yao, Hongxun},
    booktitle={ECCV},
    year={2022}
}
```


## License

This project is open sourced under MIT license. 

## Acknowledgement
This project is based on [STFAN](https://github.com/sczhou/STFAN)








