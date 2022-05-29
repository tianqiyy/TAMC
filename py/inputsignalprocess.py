import os
import shutil
import argparse
import numpy as np
import pyBigWig
from pysam import Samfile, Fastafile
from scipy.stats import scoreatpercentile
from rgt.HINT.signalProcessing import GenomicSignal # HINT-ATAC code
from rgt.GenomicRegionSet import GenomicRegionSet # HINT-ATAC code
from rgt.HINT.biasTable import BiasTable # HINT-ATAC code
from rgt.Util import HmmData # HINT-ATAC code

def default(refgenome, atac_bam, TOBIAS_FPS_bw, mpbs_bed, biastable_F, biastable_R, forward_shift, reverse_shift, outdir, prefix):
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
    bias_table = BiasTable().load_table(table_file_name_F=biastable_F, table_file_name_R=biastable_R)  # HINT-ATAC code

    # generate inputsinal numpy array
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
        input_label = region.chrom + ":" + str(region.initial) + "-" + str(region.final) + "_" + region.name.split("_")[-1]
        name = "".join([outdir, "/", prefix, "_", str(input_label), ".npy"])
        np.save(name, input_signal)
