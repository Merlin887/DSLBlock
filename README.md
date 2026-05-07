## DSLBlock

DSL Block, a denoising block architecture that addresses the resolution-dependent computational cost of the U-Net pipeline by reducing the spatial resolution at which the encoder-decoder operates.

### Overview

This repository contains a PyTorch implementation of the DSL Block. To use DSL Block as drop-in replacement
in LiteDVDNet select ```inference_mode=DownscaleInBlock``` or ```inference_mode=DownscaleInBlockCached```
and provide ```downscale_factor``` and ```ds_feat_ch``` values.

Repository includes various model modifications trained in accordance with the original article.

### Datasets 

Before running tests or training a model please download datasets <b>DAVIS_2017</b> (training) and 
<b>Set8</b> (validation and tests) by using following
[link](https://drive.google.com/file/d/1a809w-YIpt7ksO0eKuauqDFZVmkYbnVo/view?usp=drive_link). 
After downloading just unpack this archive to the repository root.

### Training

To train a model you can run following script:

```
run_training.py
```

To select model setting to train set  ```optionPath``` you need.


### Quality Testing

To compare models quality output you can run following script:

```
run_tests.py
```

Test suite paths are provided in ```test_suite_paths``` list.

### Inference Time Testing

To compare models inference time use following script:

```
analyze_model.py
```




