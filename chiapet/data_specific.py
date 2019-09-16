import pandas as pd


CHROMOSOMES = ['chr%d' % i for i in range(1, 22 + 1)] + ['chrX', 'chrY', 'chrM']

RAW_DATA_FILES = {
    'GM12878': {
        'CTCF': [
            'GM12878.CTCF.clusters_PET4+.with.ConvergentLeftRightWard.motifs.txt',
            'GM12878.CTCF.clusters_PET4+.with.Divergent.txt',
            'GM12878.CTCF.clusters_PET4+.none_motifs.simple.txt',
            'GM12878.CTCF.singletons_cluster_PET_2_3.txt',
            'GM12878_CTCF_Rep1.withLinker.clusters.inter.PET2+.bed',
            'GM12878_CTCF_Rep1.withLinker.singleton.inter.bed'
        ]
    }
}

ChromosomeDtype = pd.api.types.CategoricalDtype(
    # In human genome there are chromosomes 1-22, chromosome X, Y
    # mitochondrial DNA is also included, encoded as 'chrM'
    CHROMOSOMES,
    ordered=True
)

RAW_CHIAPET_FILE_COLUMNS = [
    ('A_chrom', ChromosomeDtype),
    ('A_start', 'int32'),
    ('A_end', 'int32'),
    ('B_chrom', ChromosomeDtype),
    ('B_start', 'int32'),
    ('B_end', 'int32'),
    ('petcount', 'int32')
]

RAW_BED_FILE_COLUMNS = [
    ('chrom', ChromosomeDtype),
    ('start', 'int32'),
    ('end', 'int32')
]