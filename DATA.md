# DATA.md

ChromGP data inventory — Hi-C contact maps and ChromHMM annotations.

## Hi-C data (Rao et al. 2014)

In-situ Hi-C, MboI digestion, 25kb resolution. All from [Rao et al. 2014 (PMID 25497547)](https://doi.org/10.1016/j.cell.2014.11.021).
Multi-cooler files stored in `/gladstone/engelhardt/lab/lchumpitaz/hi-c/mcools/`.

| Cell line | Description | Roadmap E-ID | 4DN accession | Local file |
|-----------|-------------|-------------|---------------|------------|
| **GM12878** | B-lymphoblastoid (karyotypically normal) | E116 | `4DNFIXP4QG5B` | `4DNFIXP4QG5B.mcool` |
| **K562** | Chronic myeloid leukemia | E123 | TBD | — |
| **HeLa S3** | Cervical adenocarcinoma | — | TBD | — |
| **HUVEC** | Umbilical vein endothelial | E122 | TBD | — |
| **NHEK** | Epidermal keratinocytes | E057 | TBD | — |
| **KBM7** | CML, near-haploid | — | TBD | — |
| **IMR90** | Fetal lung fibroblasts | E017 | TBD | — |
| **HMEC** | Mammary epithelial | E119 | TBD | — |

Six of eight cell lines (all except HeLa S3 and KBM7) have matching Roadmap Epigenomics 15-state ChromHMM models.
HeLa has ENCODE ChromHMM data but no Roadmap ID.

## ChromHMM annotations

**Roadmap Epigenomics** 15-state core-marks models (the ones we use).
Stored in `/gladstone/engelhardt/lab/lchumpitaz/datasets/chromhmm/`.
Source: `https://egg2.wustl.edu/roadmap/data/byFileType/chromhmmSegmentations/ChmmModels/coreMarks/jointModel/final/`

Naming pattern: `E<ID>_15_coreMarks_hg38lift_mnemonics.bed.gz`

### Downloaded

| Roadmap ID | Cell line | Local file |
|------------|-----------|------------|
| E116 | GM12878 | `E116_15_coreMarks_hg38lift_mnemonics.bed.gz` (and `.bed`) |

### Needed (for other Rao 2014 cell lines)

| Roadmap ID | Cell line | URL |
|------------|-----------|-----|
| E123 | K562 | `E123_15_coreMarks_hg38lift_mnemonics.bed.gz` |
| E122 | HUVEC | `E122_15_coreMarks_hg38lift_mnemonics.bed.gz` |
| E057 | NHEK | `E057_15_coreMarks_hg38lift_mnemonics.bed.gz` |
| E017 | IMR90 | `E017_15_coreMarks_hg38lift_mnemonics.bed.gz` |
| E119 | HMEC | `E119_15_coreMarks_hg38lift_mnemonics.bed.gz` |

### Legacy (ENCODE models — retained for reference, not used going forward)

| File | Consortium | States | Notes |
|------|-----------|--------|-------|
| `ENCFF140VIG.bed` | ENCODE (Weng Lab) | 15-state | GM12878, under-calls Polycomb (~5.5× less than E116) |
| `ENCFF338RIC.bed` | ENCODE (Weng Lab) | 18-state | GM12878 |

### ChromHMM state grouping

All Roadmap 15-state models use the same numbered naming convention.
Our merge map in `chromgp/datasets/chromhmm.py` maps them to 5 coarse groups:

| Group | Roadmap states |
|-------|---------------|
| Active | 1_TssA, 2_TssAFlnk, 6_EnhG, 7_Enh |
| Transcribed | 3_TxFlnk, 4_Tx, 5_TxWk |
| Heterochromatin | 8_ZNF/Rpts, 9_Het |
| Polycomb | 10_TssBiv, 11_BivFlnk, 12_EnhBiv, 13_ReprPC, 14_ReprPCWk |
| Quiescent | 15_Quies |

The Roadmap core-marks model was trained on 5 histone marks: H3K4me3, H3K4me1, H3K36me3, **H3K27me3**, H3K9me3.
H3K27me3 (canonical Polycomb/PRC2 mark) gives it strong power to resolve repressed chromatin
into the 5 distinct Polycomb sub-states, yielding ~10% Polycomb coverage genome-wide vs
~1.8% with the ENCODE 15-state model on chr9.

## Reference genome

| File | Path |
|------|------|
| hg38 FASTA | `/gladstone/engelhardt/home/lchumpitaz/gitclones/ChromGP/notebooks/hg38.fa` |
| hg38 FASTA index | `/gladstone/engelhardt/home/lchumpitaz/gitclones/ChromGP/notebooks/hg38.fa.fai` |
| GC content (15kb) | `/gladstone/engelhardt/home/lchumpitaz/gitclones/ChromGP/notebooks/hg38_gc_cov_15kb.tsv` |

## Configs

Config YAMLs live under `configs/4DNFIXP4QG5B/`, `configs/4DNFIXP4QG5B_10k/`, `configs/4DNFIXP4QG5B_chr19/`, `configs/4DNFIXP4QG5B_chr9/`.
Each directory contains per-model configs (`general.yaml`, `svgp.yaml`, `lcgp.yaml`, `mggp_svgp.yaml`, `mggp_lcgp.yaml`).

The `chromhmm_bed` field should point to the Roadmap BedGraph for the matching cell line.
For GM12878 (4DNFIXP4QG5B): `chromhmm_bed: /gladstone/engelhardt/lab/lchumpitaz/datasets/chromhmm/E116_15_coreMarks_hg38lift_mnemonics.bed`
