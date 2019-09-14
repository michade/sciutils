#!/usr/bin/env python

import argparse
import os

from chiapet.chiapet import ChiapetData


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

    data = ChiapetData(args.output_dir)
    data.organize_data(args.raw_data_dir)


if __name__ == '__main__':
    main()
