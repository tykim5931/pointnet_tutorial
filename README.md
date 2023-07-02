## KAIST Geometric AI Lab - Tutorial 1
https://github.com/KAIST-Geometric-AI-Group/Tutorial_1

## PointNet: Point Cloud Processing Network

## Project Structure
Below shows the overall structure of this repository.

```
pointnet
│ 
├── model.py              <- PointNet models implementation. (classification, segmentation, auto-encoder)
│ 
├── dataloaders 
│   ├── modelnet.py         <- Dataloader of ModelNet40 dataset.
│   └── shapenet_partseg.py <- Dataloader of ShapeNet Part Annotation dataset. 
│
├── utils
│   ├── metrics.py          <- Easy-to-use code to compute metrics.
│   ├── misc.py             <- Point cloud normalization ft. and code to save rendered point clouds. 
│   └── model_checkpoint.py <- Automatically save model checkpoints during training.
│
├── train_cls.py          <- Run classification.
├── train_ae.py           <- Run auto-encoding.
├── train_seg.py          <- Run part segmentation.
├── visualization.ipynb   <- Simple point cloud visualization example code.
│
├── data                  <- Project data.
│   ├── modelnet40_ply_hdf5_2048     <- ModelNet40   
│   └── shapenet_part_seg_hdf5_data  <- ShapeNet Part Annotation
│
└── checkpoints           <- Directory storing checkpoints. 
    ├── classification
    │    └── mm-dd_HH-MM-SS/epoch=16-val_acc=88.6.ckpt
    ├── auto_encoding
    └── segmentation
```

## Task 1. Point Cloud Classification
![image](Figure/cls.png)
On ModelNet40 test set:

|                                | Overall Acc |
| ------------------------------ | ----------- |
| Paper                          | 89.2 %      |
| Implemented (w/ feature trans.)| 89.0 %      | 


## Task 2. Point Cloud Part Segmentation
**_Success condition: You will get the perfect score if you achieve test mIoU over 80%._**
![image](Figure/seg.png)

For segmentation tasks, PointNet concatenates the second transformed feature with the global latent vector to form a point-wise feature tensor, which is then passed through an MLP to produce logits for m part labels.

On ShapeNet Part test set:
|             | ins. mIoU |
| ----------- | --------- |
| Paper       | 83.7 %    |
| Implemented | 80.8 %    | 


## Task 3. Point Cloud Auto-Encoding
**_success condition: You will get the perfect score if you achieve chamfer distance lower than 0.005 * N = 10.24 on the test set._**
![image](Figure/ae.png)

The PointNet Auto-encoder comprises an encoder that inputs point clouds and produces a 1024-sized global feature latent vector, and an MLP decoder that expands this latent vector incrementally until it reaches N*3. This tensor is reshaped into (N, 3), representing N points in 3D coordinates.

Here, we use chamferdist (https://github.com/krrish94/chamferdist)

On ModelNet40 test set:
|        | Chamfer Dist. |
| ------ | ------------- |
| Ours   | 9.7652        |


## Reference
- [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593)
