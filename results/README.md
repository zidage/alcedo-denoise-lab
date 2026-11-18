# Results

Text results of the runs: JSON files and tables. One folder for each step of
[the plan](../docs/plan.md), for example `results/reference_measurement/`.

- Each result file states the command, the commit of this repository, the model file hash, the
  dataset split, and the machine.
- An entry in [docs/findings/](../docs/findings/README.md) names the result file that supports it.
- Images, arrays, datasets, and weights are not committed (see `.gitignore`). They stay under
  `build/tmp/`.
