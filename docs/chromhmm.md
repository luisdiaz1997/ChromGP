# ChromHMM Annotations for GM12878

Reference chromatin state segmentations from ENCODE for use with `groups_by: chromhmm_state` in ChromGP configs.

## Files

All files are for **GM12878** cell line, **GRCh38/hg38** assembly.

| File ID | States | Format | Path | ENCODE Accession |
|---------|--------|--------|------|------------------|
| ENCFF140VIG | 15-state | BED | `/gladstone/engelhardt/lab/lchumpitaz/datasets/chromhmm/ENCFF140VIG.bed` | ENCSR310DFZ |
| ENCFF338RIC | 18-state | BED | `/gladstone/engelhardt/lab/lchumpitaz/datasets/chromhmm/ENCFF338RIC.bed` | ENCSR988QYW |

---

## 15-State Model (ENCFF140VIG)

**Source:** ENCSR310DFZ - ChromHMM 15-state model of GM12878
**Lab:** Manolis Kellis, Broad Institute
**Assembly:** GRCh38

| State | Description |
|-------|-------------|
| `Tss` | Active transcription start sites |
| `TssFlnk` | Flanking regions around TSS |
| `TssFlnkD` | Downstream flanking regions |
| `TssFlnkU` | Upstream flanking regions |
| `Tx` | Transcribed gene bodies |
| `TxWk` | Weak transcription |
| `Enh1` | Active enhancers (type 1) |
| `Enh2` | Active enhancers (type 2) |
| `EnhG1` | Genic enhancers (type 1) |
| `EnhG2` | Genic enhancers (type 2) |
| `ZNF/Rpts` | ZNF genes and repeats |
| `Het` | Heterochromatin |
| `ReprPC` | Polycomb repressed |
| `Biv` | Bivalent (poised) regions |
| `Quies` | Quiescent/low signal |

---

## 18-State Model (ENCFF338RIC)

**Source:** ENCSR988QYW - ChromHMM 18-state model of GM12878
**Lab:** Manolis Kellis, Broad Institute (ENCODE4 project)
**Assembly:** GRCh38

| State | Description |
|-------|-------------|
| `TssA` | Active transcription start sites |
| `TssBiv` | Bivalent TSS (poised) |
| `TssFlnk` | Flanking regions around TSS |
| `TssFlnkD` | Downstream flanking regions |
| `TssFlnkU` | Upstream flanking regions |
| `Tx` | Transcribed gene bodies |
| `TxWk` | Weak transcription |
| `EnhA1` | Active enhancers (type 1) |
| `EnhA2` | Active enhancers (type 2) |
| `EnhG1` | Genic enhancers (type 1) |
| `EnhG2` | Genic enhancers (type 2) |
| `EnhWk` | Weak/poised enhancers |
| `EnhBiv` | Bivalent enhancers |
| `ZNF/Rpts` | ZNF genes and repeats |
| `Het` | Heterochromatin |
| `ReprPC` | Polycomb repressed |
| `ReprPCWk` | Weak Polycomb repressed |
| `Quies` | Quiescent/low signal |

---

## Usage in ChromGP Config

```yaml
preprocessing:
  region: chr14
  groups_by: chromhmm_state
  chromhmm_bed: /gladstone/engelhardt/lab/lchumpitaz/datasets/chromhmm/ENCFF140VIG.bed
  # Optional: whitelist specific states
  chromhmm_states: [Tss, Enh1, Enh2, ReprPC, Het]
```

When `groups_by: chromhmm_state`, each Hi-C bin is assigned to a chromatin state based on the ChromHMM BED annotation. MGGP models then learn state-specific 3D structure.
