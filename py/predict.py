import os
import shutil
import argparse
import numpy as np
import pandas as pd
from scipy.stats import scoreatpercentile
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score
from scipy.integrate import trapz

import predictsignalprocess # internal

# argparser
parser = argparse.ArgumentParser(description='predicting binding score for each MPBS')

parser.add_argument("--mpbs_bed", type=str, metavar="FILE", required=True, help="mpbs input file (in bed format) location")
parser.add_argument("--outdir", type=str, metavar="PATH", required=True, help="output directory (do not add / at the end)")
parser.add_argument("--prefix", type=str, metavar="String", required=True, help="edit the name of output file")
parser.add_argument("--modelname", type=str, metavar="STRING", required=True, help="type of models based on input format: default, no_footprint, no_strand, no_size, no_cleavage_profile")
parser.add_argument("--bestmodel1", type=str, metavar="FILE", required=True, help="the trained model")
parser.add_argument("--bestmodel2", type=str, metavar="FILE", required=True, help="the trained model")
parser.add_argument("--bestmodel3", type=str, metavar="FILE", required=True, help="the trained model")
parser.add_argument("--TOBIAS_FDS_bw", type=str, metavar="FILE", default=None, help="TOBIAS foorprintscore bigwig file")
parser.add_argument("--atac_bam", type=str, metavar="FILE", default=None, help="ATAC bam file")
parser.add_argument("--refgenome", type=str, metavar="FILE", default=None, help="reference genome fastq file")
parser.add_argument("--biastable_F", type=str, metavar="FILE", default=None, help="forward biastable")
parser.add_argument("--biastable_R", type=str, metavar="FILE", default=None, help="reverse biastable")
parser.add_argument("--forward_shift", type=int, metavar="INT", default=4, help="cut_site = read.pos + forward_shift")
parser.add_argument("--reverse_shift", type=int, metavar="INT", default=-4, help="cut_site = read.aend + reverse_shift - 1")

args = parser.parse_args()

if args.modelname == "default":
    if args.refgenome is None:
        print("refgenome has not been set")
    elif args.atac_bam is None:
        print("atac_bam has not been set")
    elif args.TOBIAS_FPS_bw is None:
        print("TOBIAS_FPS_bw has not been set")
    elif args.mpbs_bed is None:
        print("mpbs_bed has not been set")
    elif args.biastable_F is None:
        print("biastable_F has not been set")
    elif args.biastable_R is None:
        print("biastable_R has not been set")
    elif args.bestmodel1 is None:
        print("bestmodel2 has not been set")
    elif args.bestmodel1 is None:
        print("bestmodel2 has not been set")
    elif args.bestmodel1 is None:
        print("bestmodel2 has not been set")
    else:
        predictsignalprocess.default(args.refgenome, args.atac_bam, args.TOBIAS_FPS_bw, args.mpbs_bed, args.biastable_F, args.biastable_R, args.bestmodel1, args.bestmodel2, args.bestmodel3, args.outdir, args.prefix)
