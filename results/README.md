# Results

The results of each step of [the plan](../docs/plan.md), one folder for each step, for example
`results/reference_measurement/`. The results of a step that is not complete are also kept here.

| Content | Committed |
| --- | --- |
| Text results: JSON files, tables, Markdown | Yes |
| `results/<step>/images/`: comparison sheets and other images | No |
| Binary arrays (`*.npz`, `*.npy`) | No |

- Each result file states the command, the commit of this repository, the model path and file
  hash, the dataset split, and the machine.
- An entry in [docs/findings/](../docs/findings/README.md) names the result file that supports it.
- The images show photos that must not be shared, so they stay on the local machine. The README of
  the step under `scripts/` states the command that makes them again.
