import pytest

import pandas as pd

from chiapet.chiapet import anchor_midpoints_in_regions
from chiapet.data_specific import RAW_BED_FILE_COLUMNS, ChromosomeDtype


def make_anchors_df(*anchors):
    df = pd.DataFrame.from_records(
        ((id_, ch, m - 5, m + 5, m) for (id_, ch, m) in anchors)
        , columns=['anchor_id', 'chrom', 'start', 'end', 'mid']
    )
    for col, dtype in RAW_BED_FILE_COLUMNS:
        df[col] = df[col].astype(dtype)
    df['mid'] = df.mid.astype(RAW_BED_FILE_COLUMNS[1][1])  # 'start'
    return df


def make_regions_df(*regions):
    df = pd.DataFrame.from_records(regions, columns=['region_id', 'chrom', 'start', 'end'])
    for col, dtype in RAW_BED_FILE_COLUMNS:
        df[col] = df[col].astype(dtype)
    return df


def make_anchor_region_pairs_df(*pairs):
    df = pd.DataFrame.from_records(pairs, columns=['chrom', 'anchor_id', 'region_id'])
    df['chrom'] = df.chrom.astype(ChromosomeDtype)
    return df


@pytest.mark.parametrize(
    'anchors, regions, expected', [
        (
            make_anchors_df((1, 'chr1', 10), (2, 'chr1', 20), (3, 'chr1', 30), (4, 'chr1', 40), (5, 'chr1', 50)),
            make_regions_df((100, 'chr1', 15, 25), (200, 'chr1', 30, 50), (300, 'chr1', 60, 70)),
            make_anchor_region_pairs_df(('chr1', 2, 100), ('chr1', 3, 200), ('chr1', 4, 200), ('chr1', 5, 200))
        ), (
            make_anchors_df((1, 'chr1', 10), (2, 'chr2', 20), (3, 'chr2', 30)),
            make_regions_df((100, 'chr1', 10, 30), (200, 'chr2', 10, 30)),
            make_anchor_region_pairs_df(('chr1', 1, 100), ('chr2', 2, 200), ('chr2', 3, 200))
        ),
    ], ids=[
        'same chromosome',
        'different chromosomes'
    ]
)
def test_anchor_midpoints_in_regions(anchors, regions, expected):
    res = anchor_midpoints_in_regions(anchors, regions, True)  # with chrom. col
    pd.testing.assert_frame_equal(res, expected)

    res_no_chrom = anchor_midpoints_in_regions(anchors, regions, False)  # without chrom. col
    expected_no_chrom = expected.drop(columns=['chrom'])
    pd.testing.assert_frame_equal(res_no_chrom, expected_no_chrom)
