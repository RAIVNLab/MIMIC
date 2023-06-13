# MIMIC: Masked Image Modeling with Image Correspondences

<p align="center">
  <img src="assets/mimic.png" width="700">
</p>

* The code for the models in this repo is based on [mae](https://github.com/facebookresearch/mae) and [croco](https://github.com/naver/croco).

### Dataset Download
***

<p align="center">
  <img src="assets/dist.png" width="500">
</p>



 you can download the zip file of image pairs with metadata by clicking on the name of each dataset: [HM3D](https://drive.google.com/file/d/1xitNF_vKrx5lqe1eWmEPlFl63l6WxL__/view?usp=sharing), [Gibson](https://drive.google.com/file/d/198KYNLk-9MiJ_4QjbaK_fayPNDAkM_j1/view?usp=sharing), [Matterport](https://drive.google.com/file/d/1mYhuYQxOwEpKT45j1MEifq92DTZV7jOB/view?usp=sharing), [Mannequin](https://drive.google.com/file/d/160rcbEXkpLrDdu13YK6t4cbwrGm-4l3k/view?usp=sharing), [ArkitScenes](https://drive.google.com/file/d/1ifSPHKU9VQ1AeimvXfp_CsJAXqTw9BSX/view?usp=sharing), [CO3D](https://drive.google.com/file/d/1Wszh2dyYEUY2WA-EBcWdk1RIztTcx06H/view?usp=sharing), [Objectron](https://drive.google.com/file/d/1OC5k6zOfOPVD85w74qHK7OEO6QhAi7iF/view?usp=sharing), [3DStreetView](https://drive.google.com/file/d/14eH-5UY0_PCOXYXEeOGl2nekhK31Y8Yq/view?usp=sharing), [DeMoN](https://drive.google.com/file/d/1_1TujxKg22PtdJV4-tMBK08KOU_UktWi/view?usp=sharing), [ScanNet](https://drive.google.com/file/d/1G-lJ7qcGu8HuOzPO22MgaUXJeM1WCLL2/view?usp=sharing).

 The whole dataset can be downloaded [here](https://drive.google.com/drive/folders/1UBCTsAQv5_sfgx1tj8yGbZqKVUu9HIfV?usp=sharing).

### Dataset Curation
***
 The dataset curation instruction is in [DATASET.md](datasets/DATASET.md)




## Pretraining

The code for the model is based on [mae](https://github.com/facebookresearch/mae) codebase, we also added cross attention blocks from [croco](https://github.com/naver/croco). Other modifications to the data loader and model to process image pairs instead of single views can be found in the `model` folder. Pretraining instructions similar to mae is in `model/PRETRAIN.md`. However, you can use the command below to start the finetunning: (make sure your effective batch size (batch_size_per_gpu * nodes * gpus_per_node * accum_iter) is equal to 4096)

```bash
 torchrun --nproc_per_node=8 main_pretrain.py  --multiview  --batch_size 128    --accum_iter 4     --model mae_vit_base_patch16     --norm_pix_loss     --mask_ratio 0.9     --epochs 200  --blr 1.5e-4  --warmup_epochs 20    --train_path_csv /path/to/csv --base_data_path /path/to/MIMIC/data/folder
 ```


 ## Fine-tunning


The following table provides the pre-trained checkpoint on MIMIC3M used in the paper.
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">ViT-Base</th>

<!-- TABLE BODY -->
<tr><td align="left">pre-trained checkpoint</td>
<td align="center"><a href="https://drive.google.com/file/d/1rwaGr-8iH4munfdouNqBDQCU5PwnLpgD/view?usp=sharing">download</a></td>
</tr>
</tbody></table>

We used the [mae](https://github.com/facebookresearch/mae) code for linear probing, [multimae](https://github.com/EPFL-VILAB/MultiMAE) for semantic segmentation and depth estimation finetunning and [vitpose](https://github.com/ViTAE-Transformer/ViTPose) for pose estimation. If you are using multimae finetunnig code, make sure to convert the checkpoint to the multimae format before evalutions.



## Requirements

Pretraining requirements: [mae.yml](model/mae.yml)

Dataset curation requirements: [dataset.yml](datasets/dataset.yml)

## License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.
