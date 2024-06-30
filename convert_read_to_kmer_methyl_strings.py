import numpy as np
import pandas as pd
import pathlib

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")
import pyfaidx
import os
import re
import argparse
from tqdm import tqdm
import sys
def main(args):
    path_to_all_fa =  "/datassd/hieunguyen/ECD/storage/resources/hg19"

    ##### HELPER FUNCTIONS
    def get_refseq(path_to_all_fa, chrom, start, end):
        import pyfaidx
        refseq = pyfaidx.Fasta(os.path.join(path_to_all_fa, "chr{}.fa".format(chrom)))
        return(str.upper(refseq.get_seq(name = "chr{}".format(chrom), start = start, end = end).seq))

    def convert_read_to_methyl_string(read, XR, XG, all_cpgs_in_cluster):
        if (XR == "CT" and XG == "CT") or (XR == "GA" and XG == "CT"):
            methyl_string = [read[i] + read[i+1] for i in all_cpgs_in_cluster]
            methyl_string_numeric = [1 if item  == "CG" else 0 for item in methyl_string]
        elif (XR == "CT" and XG == "GA") or (XR == "GA" and XG == "GA"):
            methyl_string = [read[i] + read[i+1] for i in all_cpgs_in_cluster]
            methyl_string_numeric = [1 if item  == "CG" else 0 for item in methyl_string]
        return "_".join([str(item) for item in methyl_string_numeric])

    def prepare_seq_to_texts(read, chrom, read_start, read_end, XR, XG):
        refseq_at_cluster = get_refseq(path_to_all_fa = path_to_all_fa, 
                                    chrom = chrom, 
                                    start = read_start, 
                                    end = read_end)
        all_cpg_in_read = [m.start(0) for m in re.finditer("CG", refseq_at_cluster)]
        all_strings = []
        for split_at in all_cpg_in_read:
            split_at = split_at - sum([len(item) for item in all_strings])
            left_string, right_string = read[:split_at], read[split_at:]

            all_strings.append(left_string)
            all_strings.append(right_string[0:2])

            read = right_string[2:]

        all_strings = [convert_string_to_kmer_list(item) if len(item) != 2 else [item] for item in all_strings]

        output = []
        for item in all_strings:
            output += item

        final_string = " ".join(output)
        
        return final_string

    def convert_string_to_kmer_list(s):
        return [s[i: i + 5] for i in range(len(s)) if len(s[i: i + 5]) == 5]

    # path_to_test_sample = "/datassd/hieunguyen/ECD/15052023_BERT/test_code_data/KZAY37.deduplicated.bam.filtered.bam.txt"
    # path_to_output_file = "./test.txt"
    path_to_input_file = args.input
    path_to_output_file = args.output

    output_file = open(path_to_output_file, "w")
    with open(path_to_input_file, "r") as input_file:
        for line in input_file.readlines():
            if "chrom" not in line:
                line_list = line.split(",")
                chrom = line_list[2]
                read_start = int(line_list[3])
                cigar = line_list[4]
                read = line_list[6]
                read_end = read_start + int(cigar.replace("M", "")) - 1

                refseq_at_cluster = get_refseq(path_to_all_fa = path_to_all_fa, 
                                                chrom = chrom, 
                                                start = read_start, 
                                                end = read_end)
                all_cpg_in_read = [m.start(0) for m in re.finditer("CG", refseq_at_cluster)]
                all_strings = []
                for split_at in all_cpg_in_read:
                    split_at = split_at - sum([len(item) for item in all_strings])
                    left_string, right_string = read[:split_at], read[split_at:]

                    all_strings.append(left_string)
                    all_strings.append(right_string[0:2])

                    read = right_string[2:]
                if len(right_string) != 0:
                    all_strings.append(right_string)
                    
                all_strings = [convert_string_to_kmer_list(item) if len(item) >= 5 else [item] for item in all_strings]

                output = []
                for item in all_strings:
                    output += item

                final_string = " ".join(output)

                output_file.write(final_string + "\n")
                
    output_file.close()
    input_file.close()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input', type=str,
        help='Path to input file', action='store')
    parser.add_argument('--output', type=str, action = "store",
        help='Path to output file')
    
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))