# HLoRA-MER
Our model is implemented with PyTorch 1.13.0 and Python 3.9. 

## Preparation
To try HLoRA-MER, please download [Dinov2](https://github.com/facebookresearch/dinov2) and follow the next steps.  

1.Install requried packages:
```
    $ pip install -r requirements.txt
```
2.Replace the files in the original Dinov2 project with the files provided in folder dinov2_lora:
    (1) vision_transformer.py -> dinov2-main/dinov2/models/vision_transformer.py
    (2) block.py -> dinov2-main/dinov2/layers/block.py
    (3) attention.py -> dinov2-main/dinov2/layers/attention.py

3.Make training and testing datasets:
./data_lists shows the micro-expressions samples we used in our experiments. CASME2 and SAMM in terms of 3-cls and 5-cls are included. The data.py makes datasets for each subject:
```
    $ python data.py
```
## Training
If you have already made ME datasets, you can simply train HLoRA-MER like this:
```
$ python train.py --dataset SAMM --lora_depth=2
```
You can set the depth of high-level LoRA blocks by changing the argument lora_depth. If you want to test the original Dinov2 for MER without LoRA, just set lora_depth to 0.
Leave-One-Subject-Out (LOSO) cross-validation is applied for MER task. Each subject dataset will be set as test dataset while others are concatenated as the training set. You can check details in log files.

## Testing 
dinov2_mer.py uses Dinov2 (giant version, 1100M parameters) as our basic visual foundation model. You should first download the [pre-trained model](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_reg4_pretrain.pth) The variant "lora_depth" denotes the number of tuned attention blocks, which is the N_LoRA described in our paper.
You can use the following command to test our LoRA-MER network:
```
$ python dinov2_mer.py --lora_depth=2
```

