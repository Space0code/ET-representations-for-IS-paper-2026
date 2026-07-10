# AGENTS.md

## Project

This repository contains the experiment, evaluation, descriptive-analysis, and
visualization code for a controlled comparison of four Tobii eye-tracking
representations for the IS SCAI 2026 paper.



## Coding Rules

- Use Python type hints and concise docstrings for new functions.
- Prefer explicit, readable data-science code over clever abstractions.
- Keep all paths configurable through YAML or CLI arguments.
- Do not commit raw data, processed data, model checkpoints, or large results.
- Default terminal examples assume the single conda environment `trust-me-et` is active.
- Do not add separate MOMENT or GazeMAE environment files; keep one root `requirements.txt`.
- For confusion matrices, normalize rows and use a fixed color scale `[0, 1]`.

## MEMORY.md
Always read MEMORY.md in root of the project and update it regularly. 
Remember new decisions and other important info.