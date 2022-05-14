# VAC_CSLR
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/visual-alignment-constraint-for-continuous/sign-language-recognition-on-rwth-phoenix)](https://paperswithcode.com/sota/sign-language-recognition-on-rwth-phoenix?p=visual-alignment-constraint-for-continuous)

This repo holds codes of the paper: Visual Alignment Constraint for Continuous Sign Language Recognition.(ICCV 2021) [[paper]](https://openaccess.thecvf.com/content/ICCV2021/html/Min_Visual_Alignment_Constraint_for_Continuous_Sign_Language_Recognition_ICCV_2021_paper.html)

<img src=".\framework.png" alt="framework" style="zoom: 80%;" />

---
### Update (2022.05.14)

In recent experiments, we found an implementation improvement about the proposed method. In our early experiments, we adopt `nn.DataParallel` to parallel the visual feature extractor on multiple GPUs. However, only statistic updated on device 0 is kept during training ([Dataparallel](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html)), which leads to unstable training results (results may be different when adopting different numbers of GPUs and batch sizes). Therefore, we adopt [syncBN](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch) in this update, the training schedule can be shorten to 40 epochs, and the relevant results are also provided. Experimental results on other datasets will be provided in our future journal version.

```python
from modules.sync_batchnorm import convert_model

def model_to_device(self, model):
    model = model.to(self.device.output_device)
    if len(self.device.gpu_list) > 1:
        model.conv2d = nn.DataParallel(
            model.conv2d,
            device_ids=self.device.gpu_list,
            output_device=self.device.output_device)
    model = convert_model(model)
    model.cuda()
    return model
```

With the provided code, the updated results are expected as:

| Backbone                | WER on Dev | WER on Test |                       Pretrained model                       |
| :---------------------- | :--------: | :---------: | :----------------------------------------------------------: |
| ResNet18 (baseline)     |    23.8    |    25.4     | [[Baidu]](https://pan.baidu.com/s/17ernd4x3YIAEKpVa1rJqWA?pwd=iccv) [[GoogleDrive]](https://drive.google.com/file/d/1_yPOrVyxO2AJiLC6xOAPiGuPu41ov5Yg/view?usp=sharing) |
| ResNet18+VAC (CTC only) |    21.5    |    22.1     | [[Baidu]](https://pan.baidu.com/s/1vDQyNrKM9Ar2ppvnCcohBA?pwd=VAC0) [[GoogleDrive]](https://drive.google.com/file/d/1etgf94fGvvIvR6c0VCXc8j2aFy5BsrZp/view?usp=sharing) |
| ResNet18+VAC+SMKD       |  **19.8**  |  **20.5**   | [[Baidu]](https://pan.baidu.com/s/1jWT6FhxpD36fQilXZgyW9A?pwd=SMKD) [[GoogleDrive]](https://drive.google.com/file/d/1ULbB4qNdPhDjdKUX3JlgSYkQI2W3Lwm9/view?usp=sharing) |

The VAC result is corresponding to the setting of`loss_weights: SeqCTC: 1.0, ConvCTC: 1.0`. In addition to that, the VAC+SMKD adopt the setting of `model_args: share_classifier: True, weight_norm True`.

If you find this repo useful in your research works, please consider cite our papers [VAC](https://openaccess.thecvf.com/content/ICCV2021/html/Min_Visual_Alignment_Constraint_for_Continuous_Sign_Language_Recognition_ICCV_2021_paper.html) and [SMKD](https://openaccess.thecvf.com/content/ICCV2021/html/Hao_Self-Mutual_Distillation_Learning_for_Continuous_Sign_Language_Recognition_ICCV_2021_paper.html).

---
### Prerequisites

- This project is implemented in Pytorch (>1.8). Thus please install Pytorch first.

- ctcdecode==0.4 [[parlance/ctcdecode]](https://github.com/parlance/ctcdecode)，for beam search decode.

- [Optional] sclite [[kaldi-asr/kaldi]](https://github.com/kaldi-asr/kaldi), install kaldi tool to get sclite for evaluation. After installation, create a soft link toward the sclite:    
  `ln -s PATH_TO_KALDI/tools/sctk-2.4.10/bin/sclite ./software/sclite`
  We also provide a python version evaluation tool for convenience, but sclite can provide more detailed statistics.

- [Optional] [SeanNaren/warp-ctc](https://github.com/SeanNaren/warp-ctc) At the beginning of this research, we adopt warp-ctc for supervision, and we recently find that pytorch version CTC can reach similar results.

### Data Preparation

1. Download the RWTH-PHOENIX-Weather 2014 Dataset [[download link]](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/). Our experiments based on phoenix-2014.v3.tar.gz.

2. After finishing dataset download, extract it to ./dataset/phoenix, it is suggested to make a soft link toward downloaded dataset.   
   `ln -s PATH_TO_DATASET/phoenix2014-release ./dataset/phienix2014`

3. The original image sequence is 210x260, we resize it to 256x256 for augmentation. Run the following command to generate gloss dict and resize image sequence.     

   ```bash
   cd ./preprocess
   python data_preprocess.py --process-image --multiprocessing
   ```

### Inference

​	We provide the pretrained models for inference, you can download them from:

| Backbone | WER on Dev | WER on Test | Pretrained model                                             |
| -------- | ---------- | ----------- | ------------------------------------------------------------ |
| ResNet18 | 21.2%      | 22.3%       | [[Baidu]](https://pan.baidu.com/s/12WSc2Xhy7LSkLojh1XqY6g) (passwd: qi83)<br />[[Dropbox]](https://www.dropbox.com/s/zbas78emfz5m4bp/resnet18_slr_pretrained_distill25.pt?dl=0)     |

​	To evaluate the pretrained model, run the command below：   
`python main.py --load-weights resnet18_slr_pretrained.pt --phase test`

​	(When evaluating the SMKD pretrained model,  please modify the weight_norm and share_classifier in config files as True).

### Training

The priorities of configuration files are: command line > config file > default values of argparse. To train the SLR model on phoenix14, run the command below:

`python main.py --work-dir PATH_TO_SAVE_RESULTS --config PATH_TO_CONFIG_FILE --device AVAILABLE_GPUS`

### Feature Extraction

We also provide feature extraction function to extract frame-wise features for other research purpose, which can be achieved by:

`python main.py --load-weights PATH_TO_PRETRAINED_MODEL --phase features ` 

### To Do List

- [x] Pure python implemented evaluation tools.
- [x] WAR and WER calculation scripts.

### Citation

If you find this repo useful in your research works, please consider citing:

```latex
@InProceedings{Min_2021_ICCV,
    author    = {Min, Yuecong and Hao, Aiming and Chai, Xiujuan and Chen, Xilin},
    title     = {Visual Alignment Constraint for Continuous Sign Language Recognition},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {11542-11551}
}
```

Self-Mutual Distillation Learning for Continuous Sign Language Recognition [[paper]](https://openaccess.thecvf.com/content/ICCV2021/html/Hao_Self-Mutual_Distillation_Learning_for_Continuous_Sign_Language_Recognition_ICCV_2021_paper.html)

```latex
@InProceedings{Hao_2021_ICCV,
    author    = {Hao, Aiming and Min, Yuecong and Chen, Xilin},
    title     = {Self-Mutual Distillation Learning for Continuous Sign Language Recognition},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {11303-11312}
}
```

### Acknowledge

We appreciate the help from Runpeng Cui, Hao Zhou@[Rhythmblue](https://github.com/Rhythmblue) and Xinzhe Han@[GeraldHan](https://github.com/GeraldHan) :)
