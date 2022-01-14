import os
import shutil
import argparse

import inputsignalprocess # internal

# argparser
parser = argparse.ArgumentParser(description='prepare inputsignals')

parser.add_argument("--input_format", type=str, metavar="String", required=True, help="type of input format: default, no_footprint, no_strand, no_size, no_cleavage_profile")
parser.add_argument("--mpbs_bed", type=str, metavar="FILE", required=True, help="mpbs input file (in bed format) location")
parser.add_argument("--outdir", type=str, metavar="PATH", required=True, help="output directory (do not add / at the end)")
parser.add_argument("--prefix", type=str, metavar="String", required=True, help="edit the name of output file")
parser.add_argument("--TOBIAS_FDS_bw", type=str, metavar="FILE", default=None, help="TOBIAS foorprintscore bigwig file")
parser.add_argument("--atac_bam", type=str, metavar="FILE", default=None, help="ATAC bam file")
parser.add_argument("--refgenome", type=str, metavar="FILE", default=None, help="directory to reference genome fastq file")
parser.add_argument("--biastable_F", type=str, metavar="FILE", default=None, help="directory to forward biastable file (do not add / at the end)")
parser.add_argument("--biastable_R", type=str, metavar="FILE", default=None, help="directory to reverse biastable file (do not add / at the end)")
parser.add_argument("--forward_shift", type=int, metavar="INT", default=4, help="cut_site = read.pos + forward_shift")
parser.add_argument("--reverse_shift", type=int, metavar="INT", default=-4, help="cut_site = read.aend + reverse_shift - 1")

args = parser.parse_args()

# Define input format
if args.input_format == "no_cleavage_profile":
    if args.TOBIAS_FDS_bw is None:
        print("TOBIAS_FDS_bw has not been set")
    else:
        inputsignalprocess.no_cleavage_profile(args.mpbs_bed, args.TOBIAS_FDS_bw, args.outdir, args.prefix)
