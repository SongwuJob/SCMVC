## Self-Weighted Contrastive Fusion for Deep Multi-View Clustering
> **Authors:**
Song Wu, Yan Zheng, Yazhou Ren, Jing He, Xiaorong Pu, shudong Huang, Zhifeng Hao, Lifang He.

This repository contains the code and data of our paper published in *IEEE Transactions on Multimedia (TMM)*： [Self-Weighted Contrastive Fusion for Deep Multi-View Clustering](https://ieeexplore.ieee.org/document/10499831).

<!-- > [Self-Weighted Contrastive Fusion for Deep Multi-View Clustering](https://ieeexplore.ieee.org/document/10499831) -->

## 1. Workflow of SCMVC

<img src="https://github.com/SongwuJob/SCMVC/blob/main/figures/workflow.png"  width="900" />

The framework of SCMVC. We propose a hierarchical network architecture to separate the consistency objective from the reconstruction objective. Specifically, the feature learning autoencoders first project the raw data into a low-dimensional latent space $\mathbf{Z}$. Then, two feature MLPs learn view-consensus features $\mathbf{R}$ and global features $\mathbf{H}$, respectively. Particularly, a novel self-weighting method adaptively strengthens useful views in feature fusion, and weakens unreliable views, to implement multi-view contrastive fusion.

## 2.Requirements
- python==3.7.13

- pytorch==1.12.0

- numpy==1.21.5

- scikit-learn==0.22.2.post1

- scipy==1.7.3

## 3.Datasets

- The all datasets could be downloaded from [cloud](https://pan.baidu.com/s/18If7bx2ZOVZhyijtzycjXA). key: data

- Particularly, thanks to the valuable works [MFLVC](https://github.com/SubmissionsIn/MFLVC) and [GCFAggMVC](https://github.com/Galaxy922/GCFAggMVC) for providing these datasets.

## 4.Usage

### Paper:
Self-Weighted Contrastive Fusion for Deep Multi-View Clustering: https://ieeexplore.ieee.org/document/10499831.

### To test the trained model, run:
```bash
python test.py
```

### To train a new model, run:
```bash
python train.py
```

The experiments are conducted on a Windows PC with Intel (R) Core (TM) i5-9300H CPU@2.40 GHz, 16.0 GB RAM, and TITAN X GPU (12 GB caches).


## 5.Experiment Results
we compare our proposed SCMVC with 10 state-of-the-art multi-view clustering methods:
- CGD: [multi-view clustering via cross-view graph diffusion](https://github.com/ChangTang/CGD)
- LMVSC: [large-scale multi-view subspace clustering](https://github.com/sckangz/LMVSC)
- EOMSV: [efficient one-pass multi-view subspace clustering](https://github.com/Tracesource/EOMSC-CA)
- DEMVC: [deep embedded multi-view clustering with collaborative training](https://github.com/SubmissionsIn/DEMVC)
- CoMVC: [contrastive multi-view clustering](https://github.com/DanielTrosten/mvc)
- CONAN: [contrastive fusion networks for multi-view clustering](https://github.com/Guanzhou-Ke/conan)
- MFLVC: [multi-level feature learning for contrastive multi-view clustering](https://github.com/SubmissionsIn/MFLVC)
- DSMVC: [deep safe multi-view clustering](https://github.com/Gasteinh/DSMVC)
- GCFAggMVC: [global and cross-view feature aggregation for multi-view clustering](https://github.com/Galaxy922/GCFAggMVC)
- DealMVC: [dual contrastive calibration for multi-view clustering](https://github.com/xihongyang1999/DealMVC)

<img src="https://github.com/SongwuJob/SCMVC/blob/main/figures/performance.png"  width="900"  />
<img src="https://github.com/SongwuJob/SCMVC/blob/main/figures/view_change.png"  width="900"  />

## 6.Acknowledgments

Our proposed SCMVC are inspired by [MFLVC](https://github.com/SubmissionsIn/MFLVC), [GCFAggMVC](https://github.com/Galaxy922/GCFAggMVC), and [SEM](https://github.com/SubmissionsIn/SEM). Thanks for these valuable works.

## 7.Citation
If you use our code or datasets in this repository for your research, please cite our papers.
```latex
@ARTICLE{10499831,
  author={Wu, Song and Zheng, Yan and Ren, Yazhou and He, Jing and Pu, Xiaorong and Huang, Shudong and Hao, Zhifeng and He, Lifang},
  journal={IEEE Transactions on Multimedia}, 
  title={Self-Weighted Contrastive Fusion for Deep Multi-View Clustering}, 
  year={2024},
  volume={},
  number={},
  pages={1-13},
  doi={10.1109/TMM.2024.3387298}
}

@article{xu2024self,
  title={Self-weighted contrastive learning among multiple views for mitigating representation degeneration},
  author={Xu, Jie and Chen, Shuo and Ren, Yazhou and Shi, Xiaoshuang and Shen, Hengtao and Niu, Gang and Zhu, Xiaofeng},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}

@InProceedings{Xu_2022_CVPR,
    author    = {Xu, Jie and Tang, Huayi and Ren, Yazhou and Peng, Liang and Zhu, Xiaofeng and He, Lifang},
    title     = {Multi-Level Feature Learning for Contrastive Multi-View Clustering},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2022},
    pages     = {16051-16060}
}
```

If you have any problems, please contact me by songwu.work@outlook.com.


