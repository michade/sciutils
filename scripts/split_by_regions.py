#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os

from chiapet.chiapet import ChiapetData, split_by_regions, as_normalized_tables, save_raw_bed_file
from chiapet.data_specific import CHROMOSOMES
from sciutils.timer import Timer


OUTPUT_FILE_PATTERN = '{reg_chr}_{reg_start}-{reg_end}.bed'


def split_and_save(data, regions, output_dir):
    anchors, contacts = as_normalized_tables(data)
    for _, (reg_chr, reg_start, reg_end), df in split_by_regions(anchors, contacts, regions, midpoints=True):
        df = df[['mid_A', 'mid_B', 'petcount']]
        outfile = os.path.join(
            output_dir,
            OUTPUT_FILE_PATTERN.format(reg_chr=reg_chr, reg_start=reg_start, reg_end=reg_end),
        )
        save_raw_bed_file(df, outfile)


def main() -> None:
    parser = argparse.ArgumentParser(description='Split ChIA-PET contacts by regions.')
    parser.add_argument('data_dir', help='Directory with organized ChIA-PET data.')
    parser.add_argument('regions_file', help='Regions .bed file.')
    parser.add_argument('output_dir', help='Output directory for split data.')
    parser.add_argument('--celline', default='GM12878')
    parser.add_argument('--protein', default='CTCF')
    parser.add_argument('--petcount', type=int, default=4)
    args = parser.parse_args()

    input_dir = args.data_dir
    regions_file = args.regions_file
    output_dir = args.output_dir

    assert os.path.exists(input_dir)
    assert os.access(input_dir, os.R_OK)
    assert os.path.exists(regions_file)
    assert os.access(regions_file, os.R_OK)
    assert os.path.exists(output_dir)
    assert os.access(output_dir, os.W_OK)

    with Timer() as t:
        chiapet_data = ChiapetData(input_dir)
        loader = chiapet_data.loader(args.celline, args.protein, 'intra', CHROMOSOMES, args.petcount)
        split_and_save(loader.load(), loader.load_regions(regions_file, CHROMOSOMES), output_dir)
    print(f'Done splitting in {t.elapsed:.2f}s')


if __name__ == '__main__':
    main()
