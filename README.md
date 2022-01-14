# TAMC
## Introduction
TAMC (Transcriptional factor binding prediction from ATAC-seq profile at Motif-predicted binding sites using Conventional neural networks) is an open source tool for predicting motif-centric TF binding activity from paired-end ATAC-seq data. TAMC takes advantage of signal processing strategies in HINT-ATAC and TOBIAS to prepare input signals and make predictions of binding probability using a 1D-conventional neural network (1D-CNN) model.

## Prerequisites
-Python (3.6)

-Numpy

-pyBigWig

-pysam

-scipy.stats

-[reg-gen](https://github.com/CostaLab/reg-gen)

## Clone the repository
```
git clone https://github.com/tianqiyy/TAMC.git
