#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os

from chiapet import chiapet
from chiapet.chiapet import ChiapetData
from sciutils.timer import Timer


def example(data_dir, chromosome, petcount):
    chiapet_data = ChiapetData(data_dir)
    loader = chiapet_data.loader('GM12878', 'CTCF', 'intra')

    t = Timer()

    with t:
        data = loader.load(chromosome, petcount)
        print(f'Loaded {chromosome}, PET{petcount}+ in {t.elapsed:.2f}s')
        print(data)

    with t:
        anchors, contacts = chiapet.as_normalized_tables(data)
        print(f'Normalized {chromosome}, PET{petcount}+ data prepared in {t.elapsed:.2f}s')
        print(anchors)
        print(contacts)

    with t:
        graph = chiapet.as_nx_graph(anchors, contacts)
        print(f'Graph for {chromosome}, PET{petcount}+ created in {t.elapsed:.2f}s')
        print(graph[0][4])


def main():
    parser = argparse.ArgumentParser(description='ChIA-PET handling examples')
    parser.add_argument('data_dir', help='Directory for organized data.')
    args = parser.parse_args()

    data_dir = args.data_dir

    assert os.path.exists(data_dir)
    assert os.access(data_dir, os.R_OK)

    example(data_dir, 'chr22', 4)


if __name__ == "__main__":
    main()
