#!/usr/bin/env Rscript
# Fit PoisMS (Tuzhilina, Hastie & Segal 2022) on a Hi-C contact-count matrix.
#
# Usage:
#   Rscript poisms_fit.R <counts.csv> <out.csv> [<df>] [<maxepoch>]
#
# - <counts.csv>: square N x N integer Hi-C contact matrix, CSV with no header.
# - <out.csv>:    output path for the fitted (N, 3) 3D coordinates X.
# - <df>:         B-spline degrees of freedom for the principal-curve basis
#                 (default 5; PoisMS paper convention).
# - <maxepoch>:   maximum PoisMS outer-loop epochs (default 100).
#
# Writes the fitted X matrix as CSV (no header, no row names) plus a sidecar
# <out.csv>.json with {"beta": ..., "loss": ..., "epoch": ..., "df": ...}.

user_lib <- Sys.getenv("CHROMGP_R_USER_LIB",
                       "/gladstone/engelhardt/home/lchumpitaz/R_libs/r-poisms")
if (dir.exists(user_lib)) {
  .libPaths(c(user_lib, .libPaths()))
}
suppressPackageStartupMessages({
  library(PoisMS)
  library(splines)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript poisms_fit.R <counts.csv> <out.csv> [<df>] [<maxepoch>]")
}
counts_path <- args[[1]]
out_path    <- args[[2]]
df_val      <- if (length(args) >= 3) as.integer(args[[3]]) else 5L
maxepoch    <- if (length(args) >= 4) as.integer(args[[4]]) else 100L

C <- as.matrix(read.csv(counts_path, header = FALSE))
storage.mode(C) <- "double"
stopifnot(nrow(C) == ncol(C))

# PoisMS expects a symmetric square matrix. Force symmetry just in case.
C <- (C + t(C)) / 2

# Orthogonal B-spline basis along the genomic coordinate. The PoisMS paper
# uses df=5 as the default; the smoothness of the recovered curve is roughly
# inversely proportional to df.
H <- splines::bs(seq_len(ncol(C)), df = df_val, intercept = FALSE)
H <- qr.Q(qr(H))

t0 <- Sys.time()
fit <- PoisMS(
  C, H,
  maxepoch     = maxepoch,
  verbose_poisms = FALSE,
  verbose_wpcms  = FALSE
)
elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

write.table(fit$X, file = out_path, sep = ",",
            row.names = FALSE, col.names = FALSE)

side <- list(
  beta       = fit$beta,
  loss       = fit$loss,
  epoch      = fit$epoch,
  iter_total = fit$iter_total,
  df         = df_val,
  maxepoch   = maxepoch,
  n_bins     = nrow(C),
  elapsed_s  = elapsed
)
writeLines(jsonlite::toJSON(side, auto_unbox = TRUE, pretty = TRUE),
           con = paste0(out_path, ".json"))

cat(sprintf("PoisMS done: N=%d, df=%d, epoch=%d, beta=%.3f, loss=%.3e, elapsed=%.1fs\n",
            nrow(C), df_val, fit$epoch, fit$beta, fit$loss, elapsed))
