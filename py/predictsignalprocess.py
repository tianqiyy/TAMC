import os
import shutil
import argparse
import math
import pyBigWig
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pysam import Samfile, Fastafile
from rgt.HINT.signalProcessing import GenomicSignal # HINT-ATAC
from rgt.GenomicRegionSet import GenomicRegionSet # HINT-ATAC
from rgt.HINT.biasTable import BiasTable # HINT-ATAC

import models # internal

def default(refgenome, atac_bam, TOBIAS_FDS_bw, mpbs_bed, biastable_F, biastable_R, bestmodel1, bestmodel2, bestmodel3, outdir, prefix):
    # load genome file
    fasta = Fastafile(refgenome)

    # load ATACseq bam file
    bam = Samfile(atac_bam, "rb")
    reads_file = GenomicSignal(atac_bam) # HINT-ATAC code
    sg_window_size = 9
    reads_file.load_sg_coefs(sg_window_size) # HINT-ATAC code

    # load TOBIAS footprint score bigwig file
    bw = pyBigWig.open(TOBIAS_FPS_bw)

    # load mpbs_bed file
    regions = GenomicRegionSet("regions") # HINT-ATAC code
    regions.read(mpbs_bed)

    # load Tn5 cutting bias table
    bias_table = BiasTable().load_table(table_file_name_F=biastable_F, table_file_name_R=biastable_R) # HINT-ATAC code

    # Load trained model
    model1 = models.no_cleavage_profile()
    model1.load_state_dict(torch.load(bestmodel1))
    model2 = models.no_cleavage_profile()
    model2.load_state_dict(torch.load(bestmodel2))
    model3 = models.no_cleavage_profile()
    model3.load_state_dict(torch.load(bestmodel3))

    # Create prediction record csv file
    predict_file = open("".join([outdir, "/", prefix,  "_prediction.csv"]),"w")
    predict_file.close()

    # Predict binding probability for each MPBS
    for region in regions:

        input_sequence = list()

        # TOBIAS footprint scores
        footprintscores = bw.values(region.chrom, region.initial-500, region.final+500)
        footprintscores_sequence = [0 if x != x else x for x in footprintscores]
        input_sequence.append(footprintscores_sequence)

        # HINT-ATAC cleavage profile (HINT-ATAC code)
        signal_bc_f_max_145, signal_bc_r_max_145 = reads_file.get_bc_signal_by_fragment_length(ref=region.chrom, start=region.initial-500, end=region.final+500,
                                                                                               bam=bam, fasta=fasta, bias_table=bias_table,
                                                                                               forward_shift=forward_shift, reverse_shift=reverse_shift,
                                                                                               min_length=None, max_length=145, strand=True)
        signal_bc_f_min_145, signal_bc_r_min_145 = reads_file.get_bc_signal_by_fragment_length(ref=region.chrom, start=region.initial-500, end=region.final+500,
                                                                                               bam=bam, fasta=fasta, bias_table=bias_table,
                                                                                               forward_shift=forward_shift, reverse_shift=reverse_shift,
                                                                                               min_length=145, max_length=None, strand=True)

        signal_bc_f_max_145 = reads_file.boyle_norm(signal_bc_f_max_145)
        perc = scoreatpercentile(signal_bc_f_max_145, 98)
        std = np.array(signal_bc_f_max_145).std()
        signal_bc_f_max_145 = reads_file.hon_norm_atac(signal_bc_f_max_145, perc, std)
        signal_bc_f_max_145_slope = reads_file.slope(signal_bc_f_max_145, reads_file.sg_coefs)

        signal_bc_r_max_145 = reads_file.boyle_norm(signal_bc_r_max_145)
        perc = scoreatpercentile(signal_bc_r_max_145, 98)
        std = np.array(signal_bc_r_max_145).std()
        signal_bc_r_max_145 = reads_file.hon_norm_atac(signal_bc_r_max_145, perc, std)
        signal_bc_r_max_145_slope = reads_file.slope(signal_bc_r_max_145, reads_file.sg_coefs)

        signal_bc_f_min_145 = reads_file.boyle_norm(signal_bc_f_min_145)
        perc = scoreatpercentile(signal_bc_f_min_145, 98)
        std = np.array(signal_bc_f_min_145).std()
        signal_bc_f_min_145 = reads_file.hon_norm_atac(signal_bc_f_min_145, perc, std)
        signal_bc_f_min_145_slope = reads_file.slope(signal_bc_f_min_145, reads_file.sg_coefs)

        signal_bc_r_min_145 = reads_file.boyle_norm(signal_bc_r_min_145)
        perc = scoreatpercentile(signal_bc_r_min_145, 98)
        std = np.array(signal_bc_r_min_145).std()
        signal_bc_r_min_145 = reads_file.hon_norm_atac(signal_bc_r_min_145, perc, std)
        signal_bc_r_min_145_slope = reads_file.slope(signal_bc_r_min_145, reads_file.sg_coefs)

        input_sequence.append(signal_bc_f_max_145)
        input_sequence.append(signal_bc_f_max_145_slope)
        input_sequence.append(signal_bc_r_max_145)
        input_sequence.append(signal_bc_r_max_145_slope)
        input_sequence.append(signal_bc_f_min_145)
        input_sequence.append(signal_bc_f_min_145_slope)
        input_sequence.append(signal_bc_r_min_145)
        input_sequence.append(signal_bc_r_min_145_slope)

        # generate inputsinal numpy array
        input_signal = np.array(input_sequence)
        input_signal = np.expand_dims(input_signal, 0)
        input_signal = torch.from_numpy(input_signal)

        # prediction
        with torch.no_grad():
            softmax = nn.Softmax(dim=1)
            prediction_score1 = np.asarray(softmax(model1(input_signal.float()))[:,1])[0]
            prediction_score2 = np.asarray(softmax(model2(input_signal.float()))[:,1])[0]
            prediction_score3 = np.asarray(softmax(model3(input_signal.float()))[:,1])[0]

        # export
        predict_file = open("".join([outdir, "/", prefix,  "_prediction.csv"]),"a")
        predict_file.write("\t".join([str(region.chrom), str(region.initial), str(region.final), str(region.name.split("_")[-1]), str(prediction_score1), str(prediction_score2), str(prediction_score3), str(region.orientation), "\n"]))
        predict_file.close()
