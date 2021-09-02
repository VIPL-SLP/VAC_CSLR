# VAC_CSLR
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/visual-alignment-constraint-for-continuous/sign-language-recognition-on-rwth-phoenix)](https://paperswithcode.com/sota/sign-language-recognition-on-rwth-phoenix?p=visual-alignment-constraint-for-continuous)

This repo holds codes of the paper: Visual Alignment Constraint for Continuous Sign Language Recognition.(ICCV 2021) [[paper]](https://arxiv.org/abs/2104.02330)

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
| ResNet18 | 21.2%      | 22.3%       | [[Baidu]](https://pan.baidu.com/s/12WSc2Xhy7LSkLojh1XqY6g) (passwd: qi83)<br />[[Dropbox]](https://www.dropbox.com/s/zbas78emfz5m4bp/resnet18_slr_pretrained_distill25.pt?dl=0)     

​	To evaluate the pretrained model, run the command below：   
`python main.py --load-weights resnet18_slr_pretrained.pt --phase test`

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
@inproceedings{min2021vac,
  title={Visual Alignment Constraint for Continuous Sign Language Recognition},
  author={Yuecong Min, Aiming Hao, Xiujuan Chai, Xilin Chen},
  booktitle={International Conference on Computer Vision (ICCV)},
  year={2021}
}
```

### Acknowledge

We appreciate the help from Runpeng Cui, Hao Zhou@[Rhythmblue](https://github.com/Rhythmblue) and Xinzhe Han@[GeraldHan](https://github.com/GeraldHan) :)
