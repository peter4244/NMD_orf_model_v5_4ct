#!/usr/bin/env Rscript
# make_architecture_figure.R — v5 model architecture diagram (redesigned)
#
# Color vocabulary:
#   Branch identity — teal (ATG), coral (Stop), gold (Structural)
#   Model processing — warm grey (fusion, attention, classification)
#   Data prep — light grey

library(ggplot2)

# ---------------------------------------------------------------------------
# Color palette — semantic groups
# ---------------------------------------------------------------------------
col_atg      <- "#4ECDC4"   # teal   — ATG branch
col_stop     <- "#FF6B6B"   # coral  — Stop branch
col_struct   <- "#FFE66D"   # gold   — Structural branch
col_process  <- "#C8C8C8"   # warm grey — fusion, attention, classification
col_process_light <- "#DCDCDC"  # lighter grey — ORF pills, embeddings
col_prep     <- "#E8E8E8"   # light grey
col_arrow    <- "grey30"

# SHAP method annotation colors
col_jds  <- "#2166AC"   # Joint DeepSHAP — blue
col_ks   <- "#B2182B"   # KernelSHAP — red
col_att  <- "#7B3294"   # Attention weights — purple

# Sequence channel colors
ch_cols <- c("#109648", "#255C99", "#F7B32B", "#D62839",  # A, C, G, T
             "#888888", "#BBBBBB",                          # junction, GC
             "#9E7BB5", "#B99AD4", "#D4BDE8")               # frame 0, 1, 2

arr <- function(len = 0.3) arrow(length = unit(len, "cm"), type = "closed")

# ---------------------------------------------------------------------------
# Build the figure
# ---------------------------------------------------------------------------
p <- ggplot() +
  coord_fixed(ratio = 1, xlim = c(-3, 18), ylim = c(5.5, 30)) +
  theme_void() +
  theme(plot.margin = margin(5, 5, 5, 5),
        plot.background = element_rect(fill = "white", color = NA),
        panel.background = element_rect(fill = "white", color = NA))

# =========================================================================
# TITLE
# =========================================================================
p <- p +
  annotate("text", x = 7, y = 29.2, label = "v5 ORF-Centric Model Architecture",
           size = 9, fontface = "bold") +
  annotate("text", x = 7, y = 28.5, label = "From transcript sequence to NMD prediction",
           size = 6, color = "grey40") +
  annotate("text", x = 7, y = 27.7,
           label = "K = 5 ORFs per transcript",
           size = 5.5, fontface = "italic", color = "grey40")

# =========================================================================
# DATA PREPARATION PIPELINE
# =========================================================================
prep_y1 <- 25.5; prep_y2 <- 26.8; prep_ym <- (prep_y1 + prep_y2) / 2
prep_title_y <- prep_ym + 0.25; prep_detail_y <- prep_ym - 0.25

p <- p +
  annotate("rect", xmin = -0.5, xmax = 2.5, ymin = prep_y1, ymax = prep_y2,
           fill = col_prep, color = "grey50", linewidth = 0.8) +
  annotate("text", x = 1.0, y = prep_title_y, label = "Spliced mRNA",
           size = 6, fontface = "bold") +
  annotate("text", x = 1.0, y = prep_detail_y, label = "SQANTI FASTA",
           size = 5, color = "grey40") +

  annotate("segment", x = 2.5, xend = 3.1, y = prep_ym, yend = prep_ym,
           arrow = arr(), color = col_arrow, linewidth = 0.8) +

  annotate("rect", xmin = 3.1, xmax = 6.1, ymin = prep_y1, ymax = prep_y2,
           fill = col_prep, color = "grey50", linewidth = 0.8) +
  annotate("text", x = 4.6, y = prep_title_y, label = "ORFik Scan",
           size = 6, fontface = "bold") +
  annotate("text", x = 4.6, y = prep_detail_y, label = "All ORFs identified",
           size = 5, color = "grey40") +

  annotate("segment", x = 6.1, xend = 6.7, y = prep_ym, yend = prep_ym,
           arrow = arr(), color = col_arrow, linewidth = 0.8) +

  annotate("rect", xmin = 6.7, xmax = 10.3, ymin = prep_y1, ymax = prep_y2,
           fill = col_prep, color = "grey50", linewidth = 0.8) +
  annotate("text", x = 8.5, y = prep_title_y, label = "Priority Selection",
           size = 6, fontface = "bold") +
  annotate("text", x = 8.5, y = prep_detail_y, label = "Ref CDS > TD2 CDS > Kozak",
           size = 3.8, color = "grey40") +

  annotate("segment", x = 10.3, xend = 10.9, y = prep_ym, yend = prep_ym,
           arrow = arr(), color = col_arrow, linewidth = 0.8) +

  annotate("rect", xmin = 10.9, xmax = 14.5, ymin = prep_y1, ymax = prep_y2,
           fill = col_prep, color = "grey50", linewidth = 0.8) +
  annotate("text", x = 12.7, y = prep_title_y, label = "Per-ORF Encoding",
           size = 5.5, fontface = "bold") +
  annotate("text", x = 12.7, y = prep_detail_y, label = "9ch \u00D7 500 + 5 features",
           size = 4, color = "grey40")

# =========================================================================
# FAN-OUT: 1 transcript -> K=5 ORFs
# =========================================================================
spread_y <- 24.3
orf_y1 <- 23.0; orf_y2 <- 23.8

p <- p +
  annotate("segment", x = 7.0, xend = 7.0, y = prep_y1, yend = spread_y,
           arrow = arr(), color = col_arrow, linewidth = 0.8)

orf_xs <- c(1.0, 4.0, 7.0, 10.0, 13.0)
orf_w <- 1.2

p <- p +
  annotate("segment", x = orf_xs[1], xend = orf_xs[5], y = spread_y, yend = spread_y,
           color = col_arrow, linewidth = 0.6)

for (i in seq_along(orf_xs)) {
  ox <- orf_xs[i]
  p <- p +
    annotate("segment", x = ox, xend = ox, y = spread_y, yend = orf_y2,
             color = col_arrow, linewidth = 0.6) +
    annotate("rect", xmin = ox - orf_w, xmax = ox + orf_w,
             ymin = orf_y1, ymax = orf_y2,
             fill = col_process_light, alpha = 0.9, color = "grey40", linewidth = 0.5) +
    annotate("text", x = ox, y = (orf_y1 + orf_y2) / 2,
             label = paste0("ORF ", i), size = 5)
}

# =========================================================================
# SHARED-WEIGHT PER-ORF ENCODER (enclosing dashed box)
# All 5 ORFs pass through this encoder; detail shown for one ORF.
# =========================================================================
enc_box_top <- 22.3
enc_box_bot <- 13.6

# Enclosing dashed box
p <- p +
  annotate("rect", xmin = -0.3, xmax = 14.3, ymin = enc_box_bot, ymax = enc_box_top,
           fill = NA, color = "grey50", linewidth = 1.0, linetype = "dashed") +
  annotate("text", x = -2.5, y = (enc_box_top + enc_box_bot) / 2,
           label = "Per-ORF\nencoder\n(shared\nweights)",
           size = 6, fontface = "italic", color = "grey50",
           hjust = 0.5, vjust = 0.5, lineheight = 0.85)

# Arrows from ALL 5 ORF pills into the top of the enclosing box
# Spread arrival points across the top edge
orf_arrival_xs <- c(1.0, 4.0, 7.0, 10.0, 13.0)
for (i in seq_along(orf_xs)) {
  p <- p +
    annotate("segment", x = orf_xs[i], xend = orf_arrival_xs[i],
             y = orf_y1, yend = enc_box_top,
             arrow = arr(0.2), color = col_arrow, linewidth = 0.6)
}

# --- Visual sequence strip for ATG branch (x centered at 2.5) ---
strip_x1 <- 1.0; strip_x2 <- 4.0
strip_base <- 19.5
ch_h <- 0.2
ch_names <- c("A", "C", "G", "T", "Jnc", "GC", "F0", "F1", "F2")
strip_top <- strip_base + 9 * ch_h

for (i in 1:9) {
  sy1 <- strip_base + (i - 1) * ch_h
  sy2 <- sy1 + ch_h * 0.85
  p <- p +
    annotate("rect", xmin = strip_x1, xmax = strip_x2,
             ymin = sy1, ymax = sy2,
             fill = ch_cols[i], alpha = 0.7, color = NA)
}
p <- p +
  annotate("rect", xmin = strip_x1, xmax = strip_x2,
           ymin = strip_base, ymax = strip_top,
           fill = NA, color = "grey40", linewidth = 0.4) +
  annotate("text", x = 2.5, y = strip_top + 0.35,
           label = "ATG window", size = 5.5, fontface = "bold", color = "grey40") +
  annotate("text", x = 2.5, y = strip_top + 0.7,
           label = "9ch \u00D7 500", size = 3.8, color = "grey50")

for (i in 1:9) {
  sy <- strip_base + (i - 0.5) * ch_h
  p <- p +
    annotate("text", x = strip_x1 - 0.2, y = sy, label = ch_names[i],
             size = 3.8, hjust = 1, color = "grey30")
}

# --- Visual sequence strip for Stop branch (x centered at 7.0) ---
strip2_x1 <- 5.5; strip2_x2 <- 8.5

for (i in 1:9) {
  sy1 <- strip_base + (i - 1) * ch_h
  sy2 <- sy1 + ch_h * 0.85
  p <- p +
    annotate("rect", xmin = strip2_x1, xmax = strip2_x2,
             ymin = sy1, ymax = sy2,
             fill = ch_cols[i], alpha = 0.7, color = NA)
}
p <- p +
  annotate("rect", xmin = strip2_x1, xmax = strip2_x2,
           ymin = strip_base, ymax = strip_top,
           fill = NA, color = "grey40", linewidth = 0.4) +
  annotate("text", x = 7.0, y = strip_top + 0.35,
           label = "STOP window", size = 5.5, fontface = "bold", color = "grey40") +
  annotate("text", x = 7.0, y = strip_top + 0.7,
           label = "9ch \u00D7 500", size = 3.8, color = "grey50")

# --- Visual feature table for Structural branch (x centered at 11.5) ---
feat_x1 <- 10.0; feat_x2 <- 13.0
feat_base <- 19.5
feat_h <- 0.3
feat_names <- c("frac_start", "frac_stop", "is_ref_cds", "is_sqanti_cds", "n_down_ejc")
feat_alphas <- c(0.4, 0.5, 0.7, 0.6, 0.9)
feat_top <- feat_base + 5 * feat_h

for (i in 1:5) {
  fy1 <- feat_base + (i - 1) * feat_h
  fy2 <- fy1 + feat_h * 0.85
  p <- p +
    annotate("rect", xmin = feat_x1, xmax = feat_x2,
             ymin = fy1, ymax = fy2,
             fill = col_struct, alpha = feat_alphas[i],
             color = "grey50", linewidth = 0.3) +
    annotate("text", x = feat_x1 + 0.15, y = (fy1 + fy2) / 2,
             label = feat_names[i], size = 3.5, hjust = 0, color = "grey20")
}
p <- p +
  annotate("text", x = 11.5, y = feat_top + 0.25,
           label = "Structural", size = 5.5, fontface = "bold", color = "grey40") +
  annotate("text", x = 11.5, y = feat_top + 0.6,
           label = "5 features", size = 4, color = "grey40")

# --- Arrows from visual inputs down to encoder boxes ---
enc_y2 <- 18.5; enc_y1 <- 16.8
p <- p +
  annotate("segment", x = 2.5, xend = 2.5, y = strip_base, yend = enc_y2,
           arrow = arr(0.25), color = col_arrow, linewidth = 0.7) +
  annotate("segment", x = 7.0, xend = 7.0, y = strip_base, yend = enc_y2,
           arrow = arr(0.25), color = col_arrow, linewidth = 0.7) +
  annotate("segment", x = 11.5, xend = 11.5, y = feat_base, yend = enc_y2,
           arrow = arr(0.25), color = col_arrow, linewidth = 0.7)

# --- Encoder boxes (branch identity colors) ---
p <- p +
  annotate("rect", xmin = 0.8, xmax = 4.2, ymin = enc_y1, ymax = enc_y2,
           fill = col_atg, alpha = 0.7, color = "grey30", linewidth = 0.8) +
  annotate("text", x = 2.5, y = 18.0, label = "ATG CNN",
           size = 7, fontface = "bold") +
  annotate("text", x = 2.5, y = 17.5, label = "Conv1D \u00D72 + Pool",
           size = 5, color = "grey30") +
  annotate("text", x = 2.5, y = 17.1, label = "4,500 \u2192 32-dim",
           size = 5, color = "grey30") +

  annotate("rect", xmin = 5.3, xmax = 8.7, ymin = enc_y1, ymax = enc_y2,
           fill = col_stop, alpha = 0.7, color = "grey30", linewidth = 0.8) +
  annotate("text", x = 7.0, y = 18.0, label = "STOP CNN",
           size = 7, fontface = "bold") +
  annotate("text", x = 7.0, y = 17.5, label = "Conv1D \u00D72 + Pool",
           size = 5, color = "grey30") +
  annotate("text", x = 7.0, y = 17.1, label = "4,500 \u2192 32-dim",
           size = 5, color = "grey30") +

  annotate("rect", xmin = 9.8, xmax = 13.2, ymin = enc_y1, ymax = enc_y2,
           fill = col_struct, alpha = 0.7, color = "grey30", linewidth = 0.8) +
  annotate("text", x = 11.5, y = 18.0, label = "Structural FC",
           size = 7, fontface = "bold") +
  annotate("text", x = 11.5, y = 17.5, label = "Linear + ReLU",
           size = 5, color = "grey30") +
  annotate("text", x = 11.5, y = 17.1, label = "5 \u2192 32-dim",
           size = 5, color = "grey30")

# =========================================================================
# FUSION (inside the enclosing box)
# =========================================================================
fus_y1 <- 14.0; fus_y2 <- 15.5

p <- p +
  annotate("segment", x = 2.5, xend = 5.5, y = enc_y1, yend = fus_y2,
           arrow = arr(0.25), color = col_arrow, linewidth = 0.7) +
  annotate("segment", x = 7.0, xend = 7.0, y = enc_y1, yend = fus_y2,
           arrow = arr(0.25), color = col_arrow, linewidth = 0.7) +
  annotate("segment", x = 11.5, xend = 8.5, y = enc_y1, yend = fus_y2,
           arrow = arr(0.25), color = col_arrow, linewidth = 0.7) +

  annotate("rect", xmin = 3.5, xmax = 10.5, ymin = fus_y1, ymax = fus_y2,
           fill = col_process, alpha = 0.7, color = "grey30", linewidth = 0.8) +
  annotate("text", x = 7.0, y = 15.0, label = "Concatenate + Fusion",
           size = 7, fontface = "bold") +
  annotate("text", x = 7.0, y = 14.4,
           label = "96-dim \u2192 64-dim ORF embedding",
           size = 5, color = "grey30")

# =========================================================================
# ARROWS FROM ENCLOSING BOX TO 5 EMBEDDINGS
# =========================================================================
emb_xs <- c(3, 5, 7, 9, 11)
emb_w <- 0.7
emb_y1 <- 12.0; emb_y2 <- 12.6

# Arrows from enclosing box bottom to each embedding — spread departure points
orf_depart_xs <- c(1.0, 4.0, 7.0, 10.0, 13.0)
for (i in seq_along(emb_xs)) {
  p <- p +
    annotate("segment", x = orf_depart_xs[i], xend = emb_xs[i],
             y = enc_box_bot, yend = emb_y2,
             arrow = arr(0.2), color = col_arrow, linewidth = 0.6) +
    annotate("rect", xmin = emb_xs[i] - emb_w, xmax = emb_xs[i] + emb_w,
             ymin = emb_y1, ymax = emb_y2,
             fill = col_process_light, alpha = 0.5, color = "grey40", linewidth = 0.5) +
    annotate("text", x = emb_xs[i], y = (emb_y1 + emb_y2) / 2,
             label = paste0("e", i), size = 5, fontface = "bold")
}

# =========================================================================
# ATTENTION AGGREGATOR
# =========================================================================
att_y1 <- 9.0; att_y2 <- 10.8

# Arrows from embeddings to attention — symmetric spread
for (i in seq_along(emb_xs)) {
  p <- p +
    annotate("segment", x = emb_xs[i], xend = 5.0 + (i - 1) * 1.0,
             y = emb_y1, yend = att_y2,
             arrow = arr(0.2), color = col_arrow, linewidth = 0.5)
}

p <- p +
  annotate("rect", xmin = 4.0, xmax = 10.0, ymin = att_y1, ymax = att_y2,
           fill = col_process, alpha = 0.7, color = "grey30", linewidth = 0.8) +
  annotate("text", x = 7.0, y = 10.3, label = "Attention Aggregator",
           size = 7, fontface = "bold") +
  annotate("text", x = 7.0, y = 9.75,
           label = "Softmax weights across K ORFs",
           size = 5.5, color = "grey30") +
  annotate("text", x = 7.0, y = 9.3,
           label = "\u2192 64-dim transcript embedding",
           size = 5.5, color = "grey30")

# =========================================================================
# CLASSIFICATION HEAD
# =========================================================================
cls_y1 <- 6.5; cls_y2 <- 8.0

p <- p +
  annotate("segment", x = 7.0, xend = 7.0, y = att_y1, yend = cls_y2,
           arrow = arr(), color = col_arrow, linewidth = 0.8) +
  annotate("rect", xmin = 4.5, xmax = 9.5, ymin = cls_y1, ymax = cls_y2,
           fill = col_process, alpha = 0.7, color = "grey30", linewidth = 0.8) +
  annotate("text", x = 7.0, y = 7.5, label = "Classification Head",
           size = 7, fontface = "bold") +
  annotate("text", x = 7.0, y = 6.9,
           label = "64 \u2192 32 \u2192 1 (NMD logit)",
           size = 5.5, color = "grey30")


# =========================================================================
# SHAP METHOD ANNOTATIONS (right side brackets)
# =========================================================================
bx <- 15.0; tick_l <- 0.5

jds_y1 <- enc_y1; jds_y2 <- strip_top + 0.5
p <- p +
  annotate("segment", x = bx, xend = bx, y = jds_y1, yend = jds_y2,
           color = col_jds, linewidth = 1.5) +
  annotate("segment", x = bx - tick_l, xend = bx, y = jds_y1, yend = jds_y1,
           color = col_jds, linewidth = 1.5) +
  annotate("segment", x = bx - tick_l, xend = bx, y = jds_y2, yend = jds_y2,
           color = col_jds, linewidth = 1.5) +
  annotate("text", x = bx + 0.3, y = (jds_y1 + jds_y2) / 2,
           label = "Joint\nDeepSHAP", size = 5, fontface = "bold",
           color = col_jds, hjust = 0, lineheight = 0.85)

p <- p +
  annotate("segment", x = bx, xend = bx, y = fus_y1, yend = fus_y2,
           color = col_ks, linewidth = 1.5) +
  annotate("segment", x = bx - tick_l, xend = bx, y = fus_y1, yend = fus_y1,
           color = col_ks, linewidth = 1.5) +
  annotate("segment", x = bx - tick_l, xend = bx, y = fus_y2, yend = fus_y2,
           color = col_ks, linewidth = 1.5) +
  annotate("text", x = bx + 0.3, y = (fus_y1 + fus_y2) / 2,
           label = "KernelSHAP\n(branches)", size = 5, fontface = "bold",
           color = col_ks, hjust = 0, lineheight = 0.85)

p <- p +
  annotate("segment", x = bx, xend = bx, y = att_y1, yend = att_y2,
           color = col_att, linewidth = 1.5) +
  annotate("segment", x = bx - tick_l, xend = bx, y = att_y1, yend = att_y1,
           color = col_att, linewidth = 1.5) +
  annotate("segment", x = bx - tick_l, xend = bx, y = att_y2, yend = att_y2,
           color = col_att, linewidth = 1.5) +
  annotate("text", x = bx + 0.3, y = (att_y1 + att_y2) / 2,
           label = "Attention\nweights", size = 5, fontface = "bold",
           color = col_att, hjust = 0, lineheight = 0.85)

cat("Architecture figure built successfully\n")
