import argparse
import inputsignalprocess # internal

# argparser
parser = argparse.ArgumentParser(description='prepare inputsignals')

parser.add_argument("--input_format", type=str, metavar="String", required=True, help="type of input format: default, no_footprint, no_strand, no_size, no_cleavage_profile")
parser.add_argument("--mpbs_bed", type=str, metavar="FILE", required=True, help="mpbs input file (in bed format)")
parser.add_argument("--outdir", type=str, metavar="PATH", required=True, help="output directory (do not add / at the end)")
parser.add_argument("--prefix", type=str, metavar="String", required=True, help="edit the name of output file")
parser.add_argument("--TOBIAS_FPS_bw", type=str, metavar="FILE", default=None, help="TOBIAS foorprintscore bigwig file")
parser.add_argument("--atac_bam", type=str, metavar="FILE", default=None, help="ATAC-seq bam file")
parser.add_argument("--refgenome", type=str, metavar="FILE", default=None, help="directory to reference genome fastq file")
parser.add_argument("--biastable_F", type=str, metavar="FILE", default=None, help="directory to forward biastable file (do not add / at the end)")
parser.add_argument("--biastable_R", type=str, metavar="FILE", default=None, help="directory to reverse biastable file (do not add / at the end)")
parser.add_argument("--forward_shift", type=int, metavar="INT", default=4, help="cut_site = read.pos + forward_shift")
parser.add_argument("--reverse_shift", type=int, metavar="INT", default=-4, help="cut_site = read.aend + reverse_shift - 1")

args = parser.parse_args()

# Prepare input signal
if args.input_format == "default":
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
    else:
        inputsignalprocess.default(args.refgenome, args.atac_bam, args.TOBIAS_FPS_bw, args.mpbs_bed, args.biastable_F, args.biastable_R, outdir, prefix)
