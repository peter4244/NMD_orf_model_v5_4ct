#!/usr/bin/env Rscript
# make_shap_interpretation_figure.R — Joint DeepSHAP interpretation diagram
#
# Companion figure to make_architecture_figure.R showing the three
# SHAP interpretation methods applied at different model layers.

library(ggplot2)

# ---------------------------------------------------------------------------
# Color palette (matches architecture figure)
# ---------------------------------------------------------------------------
col_atg    <- "#4ECDC4"   # teal   — ATG branch
col_stop   <- "#FF6B6B"   # coral  — STOP branch
col_struct <- "#FFE66D"   # gold   — Structural branch
col_jds    <- "#2166AC"   # Joint DeepSHAP — blue

arr <- function(len = 0.3) arrow(length = unit(len, "cm"), type = "closed")

# ---------------------------------------------------------------------------
# Build the figure
# ---------------------------------------------------------------------------
p_shap <- ggplot() +
  coord_fixed(ratio = 1, xlim = c(-1, 15), ylim = c(-0.5, 7)) +
  theme_void() +
  theme(plot.margin = margin(10, 10, 10, 10),
        plot.background = element_rect(fill = "white", color = NA),
        panel.background = element_rect(fill = "white", color = NA))

# Outer dashed box
p_shap <- p_shap +
  annotate("rect", xmin = 0.0, xmax = 14.0, ymin = -0.2, ymax = 6.5,
           fill = "#F0F4F8", alpha = 0.5, color = col_jds, linewidth = 1.2,
           linetype = "dashed")

# Title and description
p_shap <- p_shap +
  annotate("text", x = 7.0, y = 6.0,
           label = "Joint DeepSHAP Interpretation",
           size = 8, fontface = "bold", color = col_jds) +
  annotate("text", x = 7.0, y = 5.3,
           label = "All rank-0 ORF inputs varied simultaneously (9,005 dimensions)",
           size = 6, color = col_jds) +
  annotate("text", x = 7.0, y = 4.7,
           label = "500 background samples \u00B7 5 replicates \u00B7 All 15,574 test transcripts",
           size = 5, color = "grey40")

# Three SHAP component boxes (branch identity colors)
p_shap <- p_shap +
  # ATG SHAP
  annotate("rect", xmin = 0.5, xmax = 4.5, ymin = 1.5, ymax = 3.8,
           fill = col_atg, alpha = 0.5, color = "grey40", linewidth = 0.5) +
  annotate("text", x = 2.5, y = 3.2, label = "ATG SHAP",
           size = 7, fontface = "bold") +
  annotate("text", x = 2.5, y = 2.3,
           label = "4,500 values\n(9 \u00D7 500 positions)",
           size = 5, color = "grey30", lineheight = 0.85) +

  # STOP SHAP
  annotate("rect", xmin = 5.0, xmax = 9.0, ymin = 1.5, ymax = 3.8,
           fill = col_stop, alpha = 0.5, color = "grey40", linewidth = 0.5) +
  annotate("text", x = 7.0, y = 3.2, label = "STOP SHAP",
           size = 7, fontface = "bold") +
  annotate("text", x = 7.0, y = 2.3,
           label = "4,500 values\n(9 \u00D7 500 positions)",
           size = 5, color = "grey30", lineheight = 0.85) +

  # Structural SHAP
  annotate("rect", xmin = 9.5, xmax = 13.5, ymin = 1.5, ymax = 3.8,
           fill = col_struct, alpha = 0.5, color = "grey40", linewidth = 0.5) +
  annotate("text", x = 11.5, y = 3.2, label = "Structural SHAP",
           size = 7, fontface = "bold") +
  annotate("text", x = 11.5, y = 2.3, label = "5 values",
           size = 5, color = "grey30")

# Additivity formula
p_shap <- p_shap +
  annotate("text", x = 7.0, y = 0.5,
           label = "Additive:  SHAP(ATG) + SHAP(STOP) + SHAP(Struct) = f(x) \u2212 E[f(x)]",
           size = 6, color = col_jds, fontface = "italic")

cat("SHAP interpretation figure built successfully\n")
