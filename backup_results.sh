#!/bin/bash

echo "=== Research Backup Mode ==="

# create clean folder
mkdir -p clean_results

# copy ONLY summaries + best models
find experiments -name "comparison_results.yaml" -exec cp {} clean_results/ \;
find experiments -name "summary.yaml" -exec cp {} clean_results/ \;
find experiments -name "best_fold_*.pt" -exec cp {} clean_results/ \;

# compress them
tar -czf clean_results_backup.tar.gz clean_results

echo "Backup created:"
ls -lh clean_results_backup.tar.gz
