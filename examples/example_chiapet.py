#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os

import networkx as nx

from chiapet import chiapet
from chiapet.chiapet import ChiapetData
from sciutils.timer import Timer


def example(data_dir, chromosomes, petcount, regions_file_path):
    chiapet_data = ChiapetData(data_dir)
    loader = chiapet_data.loader('GM12878', 'CTCF', 'intra')

    t = Timer()

    with t:
        data = loader.load(chromosomes, petcount)
    print(f'Loaded {chromosomes}, PET{petcount}+ in {t.elapsed:.2f}s')
    print(data)

    with t:
        anchors, contacts = chiapet.as_normalized_tables(data)
    print(f'Normalized {chromosomes}, PET{petcount}+ data prepared in {t.elapsed:.2f}s')
    print(anchors)
    print(contacts)

    with t:
        graph = chiapet.as_nx_graph(anchors, contacts)
    print(f'Graph for {chromosomes}, PET{petcount}+ created in {t.elapsed:.2f}s')
    print(f'|V|={graph.number_of_nodes()}, |E|={graph.number_of_edges()}, |CC|={nx.number_connected_components(graph)}')

    with t:
        regions = loader.load_regions(regions_file_path, chromosomes)
    print(f'Regions for {chromosomes} loaded in {t.elapsed:.2f}s')
    print(regions)

    with t:
        for _, reg, df in chiapet.split_by_regions(anchors, contacts, regions):
            print('*' * 40)
            print(reg)
            print(df)
    print(f'Anchors mapped to regions for {chromosomes}, PET{petcount}+ in {t.elapsed:.2f}s')


def main():
    parser = argparse.ArgumentParser(description='ChIA-PET handling examples')
    parser.add_argument('data_dir', help='Directory for organized data.')
    parser.add_argument('regions_file', help='Directory for organized data.')
    args = parser.parse_args()

    data_dir = args.data_dir

    assert os.path.exists(data_dir)
    assert os.access(data_dir, os.R_OK)

    example(data_dir, ['chr21', 'chr22'], 4, args.regions_file)


if __name__ == "__main__":
    main()
