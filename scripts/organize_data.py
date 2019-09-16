#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os

from chiapet.chiapet import ChiapetData
from sciutils.timer import Timer


def main() -> None:
    parser = argparse.ArgumentParser(description='Organize ChIA-PET contact data.')
    parser.add_argument('raw_data_dir', help='Directory with raw data.')
    parser.add_argument('output_dir', help='Output directory for organized data.')
    args = parser.parse_args()

    input_dir = args.raw_data_dir
    output_dir = args.output_dir

    assert os.path.exists(input_dir)
    assert os.access(input_dir, os.R_OK)
    assert os.path.exists(output_dir)
    assert os.access(output_dir, os.W_OK)

    with Timer() as t:
        data = ChiapetData(args.output_dir)
        data.organize_data(args.raw_data_dir)
    print(f'Done organizing in {t.elapsed:.2f}s')


if __name__ == '__main__':
    main()
