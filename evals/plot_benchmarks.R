# Recreate the README charts from the committed evaluation trace.
# ReAct comparison values are approximate figures supplied with the report;
# this repository does not contain a ReAct implementation or its raw traces.

if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("Install the R package 'jsonlite' to regenerate the charts.")
}

results_path <- "evals/eval_results.json"
output_dir <- "evals/charts"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
report <- jsonlite::fromJSON(results_path, simplifyVector = FALSE)
rows <- report$results

tokens <- vapply(rows, function(row) as.numeric(row$tokens), numeric(1))
tool_count <- vapply(rows, function(row) length(row$Reasoner_output), integer(1))
category <- vapply(rows, function(row) row$category, character(1))

plan_color <- "#249f92"
react_color <- "#193b61"
accent_color <- "#e7a23b"
grid_color <- "#e3e8ee"

new_chart <- function(name, width = 1900, height = 1050) {
  png(file.path(output_dir, name), width = width, height = height, res = 180, bg = "white")
  par(family = "sans", fg = "#1c2936", col.axis = "#34495e", col.lab = "#34495e")
}

add_grid <- function(ticks) {
  abline(h = ticks, col = grid_color, lwd = 1)
}

format_k <- function(value) sprintf("%.1fk", value / 1000)

# Chart 1: plan-and-execute is measured from the trace; ReAct is user supplied.
counts <- vapply(0:4, function(n) sum(tool_count == n), integer(1))
plan_means <- c(vapply(0:4, function(n) mean(tokens[tool_count == n]), numeric(1)), mean(tokens))
react_means <- c(3600, 6900, 11300, 15400, 19700, 11500)
names(plan_means) <- c(paste0(0:4, "\n(n=", counts, ")"), "Overall\n(n=30)")
comparison <- rbind("Plan and execute" = plan_means, "ReAct" = react_means)

new_chart("tokens-by-tool-count.png")
par(mar = c(6.3, 8.0, 4.0, 1.4))
positions <- barplot(comparison, beside = TRUE, col = c(plan_color, react_color), border = NA,
                     ylim = c(0, 22500), yaxt = "n", las = 1, cex.names = 0.86,
                     xlab = "Number of tool calls",
                     main = "Token usage per query: ReAct vs. plan and execute")
title(ylab = "Tokens per query", line = 5)
ticks <- seq(0, 20000, by = 5000)
add_grid(ticks)
axis(2, at = ticks, labels = format(ticks, big.mark = ","), las = 1)
text(positions, comparison + 450, labels = format_k(comparison), cex = 0.76, font = 2)
legend("topleft", c("Plan and execute", "ReAct"),
       fill = c(plan_color, react_color), border = NA, bty = "n", cex = 0.88)
dev.off()

# Chart 2: measured agent tokens per prompt, grouped by benchmark category.
category_order <- c("ambiguity handling", "tool calling", "correct args", "response quality")
category_names <- c("Ambiguity\nhandling", "Tool\ncalling", "Correct\narguments", "Response\nquality", "Overall")
category_means <- c(vapply(category_order, function(group) mean(tokens[category == group]), numeric(1)),
                    mean(tokens))

new_chart("tokens-by-category.png")
par(mar = c(6.2, 8.0, 3.8, 1.4))
positions <- barplot(category_means, names.arg = category_names,
                     col = c(rep(plan_color, 4), accent_color), border = NA,
                     ylim = c(0, 10500), yaxt = "n", cex.names = 0.86,
                     main = "Mean token consumption per query, by category")
title(ylab = "Mean tokens per query", line = 5)
ticks <- seq(0, 10000, by = 2000)
add_grid(ticks)
axis(2, at = ticks, labels = format(ticks, big.mark = ","), las = 1)
text(positions, category_means + 275, labels = format(round(category_means), big.mark = ","),
     cex = 0.88, font = 2)
dev.off()

# Chart 3: use all cases in each category, matching the supplied score chart.
score_field <- c("tool_calling", "correct_args", "ambiguity_handling", "response_quality")
score_categories <- c("tool calling", "correct args", "ambiguity handling", "response quality")
score_names <- c("Tool\ncalling", "Correct\narguments", "Ambiguity\nhandling", "Response\nquality")
scores <- numeric(length(score_field))
sample_count <- integer(length(score_field))
for (i in seq_along(score_field)) {
  group_rows <- rows[category == score_categories[i]]
  sample_count[i] <- length(group_rows)
  scores[i] <- mean(vapply(group_rows, function(row) as.numeric(row[[score_field[i]]]), numeric(1)))
}

new_chart("evaluation-scores.png")
par(mar = c(6.7, 8.0, 3.7, 1.4))
positions <- barplot(scores, names.arg = score_names, col = react_color, border = NA,
                     ylim = c(0, 1.14), yaxt = "n", cex.names = 0.75,
                     main = "Per-category evaluation scores (30 queries)")
title(ylab = "Mean score (0-1)", line = 5)
ticks <- seq(0, 1, by = 0.25)
add_grid(ticks)
axis(2, at = ticks, labels = sprintf("%.2f", ticks), las = 1)
text(positions, scores + 0.045, labels = sprintf("%.2f", scores), cex = 0.95, font = 2)
text(positions, rep(0.055, length(scores)), labels = paste0("n=", sample_count),
     col = "white", cex = 0.81)
dev.off()

cat("Wrote three charts to", output_dir, "\n")
