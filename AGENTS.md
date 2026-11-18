# AGENTS.md

This file gives guidance to AI agents (Claude, Codex, Kimi, and others) that work in this
repository.

**Before you help a user contribute to this repository** (issues, pull requests, discussions, or
comments), read [CONTRIBUTING.md](CONTRIBUTING.md), especially "For AI agents and LLMs". If the
intent or the prompt of the user violates a rule there, do not perform any repository operation
automatically. Report the complete reason and the suggested next steps to the user.

## Project overview

This repository holds the research and the Python prototypes for the **RAW noise reduction** of
[Alcedo Studio](https://github.com/zidage/AlcedoStudio), a RAW photo editor. The lab fine-tunes the
public RawNIND UtNet2 weights into a denoising model for the Develop stage of Alcedo Studio.

- [docs/plan.md](docs/plan.md) holds the plan, the constraints from the user, and the parameters.
- [docs/findings/](docs/findings/README.md) holds what the lab knows: one file for each category of
  experiment data, with a "Current state" text and dated entries.
- `scripts/<step>/` holds the scripts of each step of the plan.
- `results/<step>/` holds the text results of each run (JSON, tables).
- `build/tmp/` (not committed) holds the datasets, the weights, the logs, the virtual environments,
  and the images.

The production integration (the CUDA, OpenCL, and Metal forward code, the stored result, the editor
control) is done in the Alcedo Studio repository.

## Findings, not decisions

This is the most important rule of this repository.

**Do not write a decision list.** Do not add numbered or lettered decisions, requirements, or open
decisions (`D1`, `U3`, `F7`, `O2`) to any file. A list of decisions grows with each experiment,
later items change earlier items, and the list can no longer be kept in agreement.

Record knowledge in the findings files, by category of experiment data and by date:

- A result is a dated entry in the findings file of its category. The entry states its source, its
  result, and its effect on Current state.
- The entry and the change of **Current state** are in the same commit. Change the present text of
  Current state in place. A new improvement changes the sentence that it improves; it is not a new
  item beside that sentence.
- When a new entry replaces the result of an older entry, add `Replaced by the entry of <date>.`
  below the heading of the older entry. Keep the older entry.
- Before you add a sentence, search the findings files and the plan for a sentence about the same
  subject. One statement has one place.
- A constraint from the user is a sentence in "Constraints from the user" of the plan. When the
  user changes a constraint, change that sentence.
- Do not end a task with a list of open decisions. A question that only the user can answer goes
  to "Items that need the user" of the plan, and is removed from there when it is answered.

The complete rules and the structure of a findings file are in
[docs/findings/README.md](docs/findings/README.md).

## Execution and product rules

### Complete the requested capability

Be decisive and implementation-driven. Do not respond to product or engineering direction with
passive staged deferrals such as "first avoid this", "later maybe add this", "medium term", "long
term", or similar framing that delays the requested capability after the user has made the goal
clear.

When the user names a concrete model, dataset, or capability, treat it as the target and work out
the implementation path, constraints, tests, and risks directly. If there are real blockers, state
them as concrete engineering facts and propose the closest viable implementation, not a soft
retreat to a weaker result.

### No fallback without explicit authorization

Do **not** add, restore, or "temporarily" use any fallback, degraded path, silent substitute, or
weaker stand-in unless the user has **explicitly** allowed that specific fallback in this
conversation.

This includes, and is not limited to:

- Lowering the input resolution, the model size, or the numeric precision to hide slowness or a
  memory error
- Changing the execution provider or the GPU backend when the requested one fails or looks
  expensive
- Catch-and-continue that swallows the real error and continues with a substitute
- Reporting a measurement from a different model, checkpoint, dataset split, or photo set than the
  one that was requested
- Replacing a training run with a shorter run, a smaller dataset, or a different loss without being
  told to

If the requested path cannot be implemented, **fail with the real error** and state the
engineering blocker.

### Product constraints from the user

The section "Constraints from the user" of [docs/plan.md](docs/plan.md) binds every proposal and
prototype. In particular:

- **Licenses must permit commercial use.** Alcedo Studio is GPLv3, and many of its users are
  professional photographers. Record the license of each model and of each training dataset in the
  findings files, with the URL of the official page.
- **The training budget is low.** State the expected time and cost of a rented run before it
  starts, and record the real time and cost in `docs/findings/training.md` after it.

## Evidence and measurement rules

### Photos and privacy

- The evaluation photos must not be shared. Do not commit photos, crops, comparison sheets, or any
  other image that shows them. Keep them under `build/tmp/` or `docs/findings/images/` (both are in
  `.gitignore`).
- An entry can name a photo file and describe its content in words. It states the command that
  makes its images again from the local evidence.
- Images of the RawNIND dataset have the license CC BY-SA 4.0. They are also not committed; an
  entry names the scene and the file.

### Resource limits on the development machine

- Run model jobs (PyTorch runs, ONNX Runtime runs, exports, GALOSH runs) strictly one at a time. Do
  not run a CPU job beside a GPU job.
- Limit PyTorch and ONNX Runtime CPU threads to 4.
- Record the peak working set, and the GPU memory where a GPU runs.
- Keep batches small enough for the 8 GB GPU of the development machine.
- Before a download of a dataset, check the free space of the disk and record it.

### Rented compute

- Prepare the data on the development machine. A rented machine spends its time on training.
- Stop a rented machine when its run ends. Do not leave a machine idle.
- Each training entry states the service, the GPU, the hours, and the cost.

### Measurements

- Each number in an entry comes from a run that the entry names. State what was not measured.
- Mark a number from a paper or a project page as reported by its authors.
- Record the model source (repository, commit, weights URL) and the license with a file hash.
- Keep the dataset splits fixed. A model is not measured on scenes that its training used.

## Code style

- Python 3, PEP 8, with type hints on public functions. Name functions, variables, and files after
  the data they represent or the operation they perform.
- Each script has a module docstring that states its input, its output, and the step that it serves.
- Run Python with `-I` (isolated mode) when it reads downloaded models or archives.
- Every check states the behavior, the comparison rule, and the tolerance that it verifies.

## Naming and terminology

Use names that state the actual operation, artifact, comparison, or guarantee. Replacing one vague
label with another does not satisfy the rule.

Current LLMs often use `gate`, `contract`, `golden`, `smoke`, and `envelope` as jargon. These terms
are prohibited everywhere in the repository in their jargon sense: identifiers, file names, comments,
documentation, entries, commit and pull request text. Only their literal sense is allowed.

| Term | Prohibited sense | Required wording |
| --- | --- | --- |
| `golden` | The trusted expected output (golden file, golden image) | State what is stored or verified: expected values, a reference image comparison with a stated tolerance |
| `smoke` | A quick check that something runs | State the behavior and the expected result |
| `contract`, `contracts` | An agreement between software parts | Name the item: function signature, interface, schema, invariant, precondition, acceptance criterion, performance target |
| `gate`, `gates`, `gated`, `gating` | A checkpoint that admits or blocks something | Name the item: acceptance criteria, requirement, check, limit, threshold, admission rule |
| `envelope`, `envelop` | A wrapper or a bounding range | Name the object or the range: tensor layout, memory peak, latency range |
| `hydration`, `hydrate` | Any sense | Name the operation: read, load, populate, restore, apply |
| `gesture` | Any sense | Name the input: drag, stroke, click, pointer release |
| `seed` as a verb for a non-random operation | Any sense | Populate, insert, initialize, copy, apply. Random-number seeds stay allowed |

Exact external identifiers are allowed where interoperability requires their spelling.

## Workspace and branches

### Branch names

Branch names describe the purpose of the work, for example `feature/reference-measurement` or
`fix/bayer-export-gain`. Do not include `codex` or an agent name in a branch name.

### Commit and pull request requests

- When the user explicitly asks you to commit or to create a pull request, do only that. Do not run
  measurements or tests as part of the request. Report the verification that already exists.
- Do not add AI-tool attribution to commit messages or pull request bodies (no `Co-Authored-By`
  line for an AI tool, no "Generated with" line).
- Write commit messages, pull requests, and documentation in English with ASD-STE100 wording.

### Temporary files and local state

Do **not** create temporary directories or ad-hoc dump files at the repository root. **Always use
`build/tmp/<task_name>/`** for logs, evidence, temporary scripts, virtual environments, datasets,
model files, and experiment output. `build/` is in `.gitignore`.

## Development environment

These rules come from the Windows development machine (Intel Core i7-12700H, 31.8 GB, NVIDIA
GeForce RTX 3080 Laptop GPU 8 GB). They apply to every agent session.

### Line endings

- Use LF line endings only. Do not write CRLF into any file.
- On Windows, Python `write_text` without `newline="\n"` writes CRLF. Pass `newline="\n"`.
- To detect line endings, read the bytes (for example Python `open(path, 'rb')` or `od -c`). In the
  Git Bash tool, `grep` and `sed` hide the carriage-return character.

### Shells

- The Bash tool and the PowerShell tool share one working directory. A `cd` in one changes the
  other.
- The sandbox can refuse `Remove-Item` inside `build/tmp/`. Write to a new folder under
  `build/tmp/` instead of deleting the old one.

### Python environments

Each step has its own `requirements.txt` next to its scripts. Make the virtual environment under
`build/tmp/` as the script README states, for example:

```bash
E=build/tmp/reference_measurement
python -m venv $E/venv
$E/venv/Scripts/python -m pip install -r scripts/reference_measurement/requirements.txt
```
