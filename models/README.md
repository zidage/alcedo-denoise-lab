# Models

Model weights, training checkpoints, and exported graphs. The content of this folder is not
committed (see `.gitignore`); only this file is.

| Path | Content |
| --- | --- |
| `models/public/` | The public RawNIND UtNet2 weights, unchanged |
| `models/<run>/` | The checkpoints of one training run, named after the run in [docs/findings/training.md](../docs/findings/training.md) |
| `models/export/<run>/` | The ONNX and safetensors exports of a run |

- Each folder has a `SOURCE.md` that states the URL or the run, the license, and the file hashes.
- A result file under `results/` names the model by its path here and by its file hash.
