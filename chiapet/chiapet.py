# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import os
import re
import time
from typing import Tuple, List, Optional, Generator

import pandas as pd
import networkx as nx

from sciutils.partial import ProperPartial
from .data_specific import RAW_CHIAPET_FILE_COLUMNS, RAW_BED_FILE_COLUMNS, RAW_DATA_FILES, CHROMOSOMES, ChromosomeDtype
from .points_in_regions import points_in_disjoint_regions


def load_raw_chiapet_text_file(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(
        file_path,
        sep='\t',
        header=None,
        index_col=False,
        usecols=range(7),
        names=[name for name, _ in RAW_CHIAPET_FILE_COLUMNS],
        dtype={name: dtype for name, dtype in RAW_CHIAPET_FILE_COLUMNS},
        engine='c',
        quoting=csv.QUOTE_NONE
    )
    return df


def load_raw_bed_file(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(
        file_path,
        sep='\t',
        header=None,
        index_col=False,
        usecols=range(3),
        names=[name for name, _ in RAW_BED_FILE_COLUMNS],
        dtype={name: dtype for name, dtype in RAW_BED_FILE_COLUMNS},
        engine='c',
        quoting=csv.QUOTE_NONE,
    )
    return df


def save_raw_bed_file(df: pd.DataFrame, file_path: str):
    df.to_csv(file_path, sep='\t', header=False, index=False)


def write_pandas_to_raw_chiapet_text_file(df: pd.DataFrame, file_path: str) -> None:
    df.to_csv(file_path, sep='\t', header=False, index=False, quoting=csv.QUOTE_NONE)


class ChiapetData(object):
    STORED_MIN_PETCOUNTS = (1, 2, 4)

    def __init__(
            self, data_dir,
            data_filename_pattern=r'(?P<celline>[A-Za-z0-9]+)_(?P<protein>[A-Za-z0-9]+)_(?P<chromosome>chrom(?:X|Y|M|\d\d?))_PET(?P<min_petcount>\d+)\+.txt',
            data_filename_format='{celline}_{protein}_{chromosome}_PET{min_petcount}+.txt'
    ):
        assert os.path.exists(data_dir)
        assert os.access(data_dir, os.R_OK)
        self._data_dir = data_dir
        self._data_filename_pattern = re.compile(data_filename_pattern)
        self._data_filename_format = data_filename_format

    @property
    def data_dir(self) -> str:
        return self._data_dir

    def _chiapet_filename(self, celline: str, protein: str, chromosome: str, stored_min_petcount: int) -> str:
        return self._data_filename_format.format(
            celline=celline, protein=protein, chromosome=chromosome, min_petcount=stored_min_petcount
        )

    def get_data_files(self) -> List[str]:
        files = []
        for filename in os.listdir(self.data_dir):
            m = self._data_filename_pattern.match(filename)
            if m is not None:
                files.append(m.group(0))
        return files

    def organize_data(self, input_dir: str) -> None:
        for filename in self.get_data_files():
            os.remove(filename)
        for celline, celline_data in RAW_DATA_FILES.items():
            for target_protein, data_files in celline_data.items():
                t0 = time.time()
                print(f'Starting processing {celline}-{target_protein}')
                group_df = self._read_group_data(input_dir, data_files)
                print(f'Read data files for {celline}-{target_protein} in {time.time() - t0:.2f}s.')
                self._organize_group(self.data_dir, group_df, celline, target_protein)
                print(f'Done processing {celline}-{target_protein} in {time.time() - t0:.2f}s.')

    @staticmethod
    def _read_group_data(input_dir: str, data_files: List[str]) -> pd.DataFrame:
        group_df = pd.concat([
            load_raw_chiapet_text_file(os.path.join(input_dir, filename))
            for filename in data_files
        ], ignore_index=True)
        return group_df

    def _organize_group(self, output_dir: str, group_df: pd.DataFrame, celline: str, protein: str) -> None:
        positional_cols = list(name for name, _ in RAW_CHIAPET_FILE_COLUMNS)[:6]
        group_df.sort_values(by=positional_cols, ascending=True, inplace=True)
        group_df.drop_duplicates(positional_cols, inplace=True)

        same_chrom = group_df.A_chrom == group_df.B_chrom
        intra_df = group_df[same_chrom]
        inter_df = group_df[~same_chrom]

        def _filter_and_save(df, _min_petcount, _chromosome):
            filtered_df = df[df.petcount >= min_petcount]
            filename = self._chiapet_filename(celline, protein, _chromosome, _min_petcount)
            write_pandas_to_raw_chiapet_text_file(filtered_df, os.path.join(output_dir, filename))
            print(f'Wrote {filename}')

        for min_petcount in self.STORED_MIN_PETCOUNTS:
            _filter_and_save(inter_df, min_petcount, 'inter')
            for chromosome in CHROMOSOMES:
                _filter_and_save(intra_df[intra_df.A_chrom == chromosome], min_petcount, chromosome)

    def load_data(
            self,
            celline: str, target_protein: str, kind: str, regions: List[str], min_petcount: int
    ) -> pd.DataFrame:
        assert kind in ['intra', 'inter', 'all']

        if regions == 'all':
            regions = list(CHROMOSOMES)
        else:
            if isinstance(regions, str):
                regions = [regions]
            for region in regions:
                assert region in CHROMOSOMES  # TODO: region filtering

        dfs = []
        if kind == 'intra' or kind == 'all':
            dfs.extend(self._load_intra(celline, target_protein, regions, min_petcount))
        if kind == 'inter' or kind == 'all':
            dfs.extend(self._load_inter(celline, target_protein, regions, min_petcount))
        df = pd.concat(dfs)
        return df

    def _load_intra(
            self,
            celline: str, target_protein: str, regions: List[str], min_petcount: int
    ) -> List[pd.DataFrame]:
        stored_min_petcount = self._get_stored_min_petcount(min_petcount)
        dfs = []
        for chromosome in regions:
            df = load_raw_chiapet_text_file(os.path.join(
                self._data_dir,
                self._chiapet_filename(celline, target_protein, chromosome, stored_min_petcount)
            ))
            df = df[df.petcount >= min_petcount]
            dfs.append(df)
        return dfs

    def _load_inter(
            self,
            celline: str, target_protein: str, regions: List[str], min_petcount: int
    ) -> List[pd.DataFrame]:
        stored_min_petcount = self._get_stored_min_petcount(min_petcount)
        df = load_raw_chiapet_text_file(os.path.join(
            self._data_dir,
            self._chiapet_filename(celline, target_protein, 'inter', stored_min_petcount)
        ))
        chromosomes = set(regions)
        df = df[df.A_chrom.isin(chromosomes) & df.B_chrom.isin(chromosomes) & (df.petcount >= min_petcount)]
        return [df]

    def _get_stored_min_petcount(self, petcount: int) -> int:
        for k in self.STORED_MIN_PETCOUNTS[::-1]:
            if k < petcount:
                return k
        return 1

    def loader(self, *args, **kwargs) -> ChiapetLoader:
        return ChiapetLoader(self, *args, **kwargs)

    def load_regions(self, path: str, regions: Optional[List[str]] = None):
        df = load_raw_bed_file(path)
        if regions is not None:
            chromosomes = set(regions)
            df = df[df.chrom.isin(chromosomes)]
        df.index.names = ['region_id']
        return df


class ChiapetLoader(object):
    def __init__(self, chiapet_data: ChiapetData, *args, **kwargs):
        self._chiapet_data = chiapet_data
        load_partial = ProperPartial(self.load_impl, *args, **kwargs)
        self.load = load_partial

    def load_impl(
            self,
            celline: str, target_protein: str, kind: str,
            regions: List[str], min_petcount: int
    ) -> pd.DataFrame:
        return self._chiapet_data.load_data(celline, target_protein, kind, regions, min_petcount)

    def load_regions(self, path: str, regions: Optional[List[str]] = None):
        if regions is None and 'regions' in self.load:
            regions = self.load['regions']
        return self._chiapet_data.load_regions(path, regions)


_ANCHOR_ID_PARTS = ['chrom', 'start', 'end']
_ANCHOR_ID_A = ['A_' + s for s in _ANCHOR_ID_PARTS]
_ANCHOR_ID_B = ['B_' + s for s in _ANCHOR_ID_PARTS]


def _select_contacts_by_anchors(anchors: pd.DataFrame, contacts: pd.DataFrame) -> pd.DataFrame:
    contacts = pd.merge(
        contacts, anchors.reset_index().rename(
            columns=dict(zip(_ANCHOR_ID_PARTS, _ANCHOR_ID_A))
        ),
        on=_ANCHOR_ID_A
    )
    contacts = pd.merge(
        contacts, anchors.reset_index().rename(
            columns=dict(zip(_ANCHOR_ID_PARTS, _ANCHOR_ID_B))
        ),
        on=_ANCHOR_ID_B,
        suffixes=['_A', '_B']
    )
    contacts.index.names = ['contact_id']
    return contacts


def as_normalized_tables(df, use_midpoints=True):
    an1 = df[_ANCHOR_ID_A]
    an1.columns = _ANCHOR_ID_PARTS
    an2 = df[_ANCHOR_ID_B]
    an2.columns = _ANCHOR_ID_PARTS
    anchors = pd.concat([an1, an2], axis=0, ignore_index=True, sort=False)
    anchors.columns = _ANCHOR_ID_PARTS
    anchors = anchors.drop_duplicates(subset=_ANCHOR_ID_PARTS, keep='first')
    if use_midpoints:
        anchors['mid'] = (anchors.start + anchors.end) // 2
        sort_cols = ['chrom', 'mid']
    else:
        sort_cols = _ANCHOR_ID_PARTS
    anchors.sort_values(by=sort_cols, inplace=True)
    anchors = anchors.reset_index(drop=True)
    anchors.index.names = ['anchor_id']

    contacts = _select_contacts_by_anchors(anchors, df)

    # reorder columns & get rid of coords
    contacts = contacts[['anchor_id_A', 'anchor_id_B', 'petcount']]
    return anchors, contacts  # TODO: copy to prevent set-on-view issues?


def anchor_midpoints_in_regions(anchors: pd.DataFrame, regions: pd.DataFrame, add_chrom_col: bool = False) -> pd.DataFrame:
    chromosomes = sorted(regions.chrom.unique())

    results = []
    for chrom in chromosomes:
        chrom_anchors = anchors[anchors.chrom == chrom]
        chrom_regions = regions[regions.chrom == chrom]
        id_pairs = points_in_disjoint_regions(
            chrom_anchors.anchor_id.values,
            chrom_anchors.mid.values,
            chrom_regions.region_id.values,
            chrom_regions.start.values,
            chrom_regions.end.values
        )
        chrom_df = pd.DataFrame(id_pairs, copy=False)
        chrom_df.columns = ['anchor_id', 'region_id']
        results.append(chrom_df)

    if add_chrom_col:
        df = pd.concat(results, keys=chromosomes, sort=False, names=['chrom']).reset_index('chrom')
        df['chrom'] = df.chrom.astype(ChromosomeDtype)
        df = df.reset_index(drop=True)
    else:
        df = pd.concat(results, sort=False, ignore_index=True)
    return df


def split_by_regions(anchors: pd.DataFrame, contacts: pd.DataFrame, regions: pd.DataFrame, midpoints=False) -> Generator[Tuple[int, Tuple[str, int, int], pd.DataFrame]]:
    pairings = anchor_midpoints_in_regions(anchors.reset_index(), regions.reset_index())
    for reg_id, reg_pairing in pairings.groupby('region_id'):
        reg_anchors_set = set(reg_pairing.anchor_id)
        reg_contacts = contacts[contacts.anchor_id_A.isin(reg_anchors_set) & contacts.anchor_id_B.isin(reg_anchors_set)]
        if midpoints:
            reg_anchors = anchors[anchors.index.isin(reg_anchors_set)].reset_index()
            reg_mid = reg_contacts
            reg_mid = pd.merge(reg_mid, reg_anchors[['anchor_id', 'mid']].rename(
                columns={'anchor_id': 'anchor_id_A', 'mid': 'mid_A'}
            ), on='anchor_id_A')
            reg_mid = pd.merge(reg_mid, reg_anchors[['anchor_id', 'mid']].rename(
                columns={'anchor_id': 'anchor_id_B', 'mid': 'mid_B'}
            ), on='anchor_id_B')
            res = reg_mid
        else:
            res = reg_contacts
        yield reg_id, tuple(regions.loc[reg_id]), res


def as_nx_graph(anchors, contacts, petcounts='petcount', drop_isolated_nodes=True):
    g = nx.Graph()
    if petcounts:
        g.add_weighted_edges_from(
            (
                (u, v, w) if u < v else (v, u, w)
                for _, (u, v, w) in contacts[['anchor_id_A', 'anchor_id_B', 'petcount']].iterrows()
            ),
            petcounts
        )
    else:
        g.add_edges_from(
            (
                (u, v) if u < v else (v, u)
                for _, (u, v) in contacts[['anchor_id_A', 'anchor_id_B']].iterrows()
            )
        )

    if not drop_isolated_nodes:
        g.add_nodes_from(anchors.index)

    return g