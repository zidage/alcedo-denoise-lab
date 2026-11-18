# Alcedo Denoise Lab

Research and Python prototypes for the RAW noise reduction of
[Alcedo Studio](https://github.com/zidage/AlcedoStudio), a RAW photo editor.

The lab makes a denoising model for the Develop stage of Alcedo Studio. It starts from the public
[RawNIND](https://arxiv.org/abs/2501.08924) UtNet2 weights and fine-tunes them on the RawNIND
dataset. The integration (the CUDA, OpenCL, and Metal forward code, the stored result, the editor
control) is done in the Alcedo Studio repository.

## Where to read

| You want | Read |
| --- | --- |
| What the lab knows now | The section "Current state" at the top of each file in [docs/findings/](docs/findings/README.md) |
| What the lab does next | [docs/plan.md](docs/plan.md) |
| The numbers behind a statement | The dated entries in the findings files, and `results/` |
| The rules for agents and contributors | [AGENTS.md](AGENTS.md), [CONTRIBUTING.md](CONTRIBUTING.md) |

## How the lab records knowledge

The lab has no decision list. It keeps one findings file for each category of experiment data
(datasets, methods, quality, training, inference cost). Each file has a "Current state" text that
is changed in place, and dated entries below it. A new result changes the present text; it does not
add a new numbered decision beside the old ones. The rules are in
[docs/findings/README.md](docs/findings/README.md).

## Repository layout

| Path | Content |
| --- | --- |
| `docs/plan.md` | The plan, with the constraints from the user and the parameters |
| `docs/findings/` | Present knowledge and dated entries, one file for each category |
| `scripts/<step>/` | The scripts of each step, each with a README that states its commands |
| `results/<step>/` | The text results of each run (JSON, tables) |
| `build/tmp/` | Not committed: datasets, weights, logs, virtual environments, images |

Photos, images made from photos, datasets, and model weights are not committed.

## License

The code and the documentation of this repository have the license GPL-3.0 (see
[LICENSE](LICENSE)). The licenses of the dataset and of the public weights are in
[docs/findings/datasets.md](docs/findings/datasets.md) and
[docs/findings/methods.md](docs/findings/methods.md).
