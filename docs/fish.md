# Multiplex FISH Data for ChromGP Validation

External 3D structural validation: compare ChromGP posterior-mean coordinates
against orthogonal imaging measurements. We use the Wang et al. 2016
multiplex-FISH dataset, which is also the validation benchmark used by PoisMS
[Tuzhilina, Hastie & Segal 2022] and DBMS [Tuzhilina, Hastie & Segal 2024],
giving us an apples-to-apples FISH-distance comparison.

## Files

All from the same Wang 2016 publication, in 4DN FOF-CT v0.1 format
(GRCh38, microns).

| Cell line | Chromosome | Probes | Source | Local file |
|-----------|-----------|--------|--------|------------|
| IMR90 | chr20 | 30 TADs | 4DN `4DNFIES5RS7Q` | `/gladstone/engelhardt/lab/lchumpitaz/datasets/fish/wang2016/4DNFIES5RS7Q_chr20.csv` |
| IMR90 | chr21 | 34 TADs | 4DN `4DNFIW2N41FQ` | `/gladstone/engelhardt/lab/lchumpitaz/datasets/fish/wang2016/4DNFIW2N41FQ_chr21.csv` |
| IMR90 | chr22 | 27 TADs | 4DN `4DNFIXRYL1SK` | `/gladstone/engelhardt/lab/lchumpitaz/datasets/fish/wang2016/4DNFIXRYL1SK_chr22.csv` |

## Source

**Wang S, Su J-H, Beliveau BJ, Bintu B, Moffitt JR, Wu C-T, Zhuang X.**
*Spatial organization of chromatin domains and compartments in single
chromosomes.* Science 353(6299):598-602, August 2016.
DOI: [10.1126/science.aaf8084](https://doi.org/10.1126/science.aaf8084)

Archived on the 4D Nucleome data portal:

- Publication: <https://data.4dnucleome.org/publications/6162d287-5782-4f40-aacd-d5da75f0770e/>
- chr20 experiment set: `4DNESFLAEE5P`, processed file `4DNFIES5RS7Q`
- chr21 experiment set: `4DNESC1DDSAH`, processed file `4DNFIW2N41FQ`
- chr22 experiment set: `4DNESR9M2CET`, processed file `4DNFIXRYL1SK`

To re-download:

```bash
DIR=/gladstone/engelhardt/lab/lchumpitaz/datasets/fish/wang2016
S3=https://4dn-open-data-public.s3.amazonaws.com/fourfront-webprod/wfoutput
curl -L -o $DIR/4DNFIES5RS7Q_chr20.csv \
  $S3/139f22c0-efda-4559-b982-6e4443ca8b1a/4DNFIES5RS7Q.csv
curl -L -o $DIR/4DNFIW2N41FQ_chr21.csv \
  $S3/813b498d-b2ae-4db7-a4ac-eb2507d8d8b7/4DNFIW2N41FQ.csv
curl -L -o $DIR/4DNFIXRYL1SK_chr22.csv \
  $S3/d995836c-a8e5-48aa-93a1-b76f1f1e33a3/4DNFIXRYL1SK.csv
```

The 4DN portal also hosts Xi (`4DNFIQVE9EQA`, 40 TADs) and Xa
(`4DNFI9PGBB4I`, 40 TADs) under the same publication, but our Hi-C runs
don't yet include the X chromosomes.

## Format

The file is in **4DN FOF-CT v0.1** (FISH Omics Format — Chromatin Tracing).
Lines beginning with `#` are header metadata; data rows follow `##columns=`.

Header (selected):

```
##FOF-CT_version= v0.1
##Table_namespace= 4dn_FOF-CT_core
##genome_assembly=GRCh38
##XYZ_unit=micron
##columns=(Spot_ID,Trace_ID,X,Y,Z,Chrom,Chrom_Start,Chrom_End)
```

| Column | Meaning |
|--------|---------|
| `Spot_ID` | Unique spot id within the file |
| `Trace_ID` | Cell identifier (one trace = one chromosome copy in one cell) |
| `X`, `Y`, `Z` | 3D position of the FISH spot, microns |
| `Chrom` | Chromosome (`21` for this file) |
| `Chrom_Start`, `Chrom_End` | Genomic coordinates (**GRCh38**) of the imaged region |

The 34 probes target chr21 TADs spanning ~10.42 Mb – 46.46 Mb, each ~100 kb
wide. The file contains **120 traces** (cells) and **3,933 spots**
(~96% probe completeness; missing probes correspond to spots not detected in
a given cell).

## Assembly

GRCh38 — matches ChromGP Hi-C inputs (Rao 2014 mcools are lifted to hg38 in
DATA.md). No coordinate lift-over required.

## Derived reference distance matrix

For ChromGP validation we collapse the per-cell spot table into a
probe-level reference distance matrix:

1. For each cell (`Trace_ID`), compute pairwise Euclidean distances between
   all observed spots → `(34, 34)` matrix with NaNs where probes are missing.
2. Take the **per-pair median across cells** → single `(34, 34)` reference
   distance matrix in microns (matches Wang 2016 and Tuzhilina convention —
   robust to outlier cells and to the fact that 3D coordinates are defined
   only up to rigid motion across cells).

This median matrix is what `chromgp.datasets.fish.load_wang2016` returns. The
loader also reports the per-probe GRCh38 midpoint so each row/column can be
mapped to a Hi-C bin index given a resolution + region.

## Probe-level aggregation (matches PoisMS / DBMS)

ChromGP fits 3D coordinates per 25 kb bin (~1335 bins on chr21); each Wang
2016 probe spans ~100 kb (~4 bins). To compare apples-to-apples with
Tuzhilina, Hastie & Segal, we **aggregate ChromGP coordinates per probe**
before computing distances:

For each probe `(start, end)`, average ChromGP's 3D posterior-mean
coordinates over every bin whose midpoint falls inside `[start - res/2,
end + res/2]`, yielding a single 3D point per probe (`(34, 3)`). Pairwise
distances between probes are then computed from these aggregated points
and compared to the FISH median distance matrix.

Probes whose footprint falls in a region cooler trimmed out (acrocentric
short arm, centromere) get no overlapping bins and are dropped.

## Validation metrics

Implemented in `chromgp/analysis.py::fish_validation`:

- **Pairwise-distance Spearman r** between the FISH median distance matrix
  and the ChromGP probe-level distance matrix over upper-triangle probe
  pairs (`|i-j| >= 1`). This is the headline number used by PoisMS /
  DBMS — invariant to monotone distance rescaling, so comparable across
  methods regardless of unit.
- **Log-distance Pearson r** between `log(FISH dist)` and `log(predicted
  dist)`. Secondary diagnostic; sensitive to distance-amplitude
  calibration.
- **Procrustes RMSD** (auxiliary) between probe-level ChromGP coordinates
  and a 3D MDS embedding of the FISH median distance matrix, after
  isotropic unit-RMS rescaling. Stacks two error sources (MDS 3D
  embedding loss + alignment) so it is reported only as a diagnostic.

Spearman alone is the metric to put in the manuscript head-to-head table.
