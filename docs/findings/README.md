# Findings

This folder is the record of what the lab knows. It has one file for each category of experiment
data. Each file holds the present knowledge of its category and the dated entries that support it.

## Why the record has this form

A list of numbered decisions grows with every experiment. A later decision changes an earlier one,
both stay in the list, and the reader must find out which one applies. The
[film lab](https://github.com/zidage/alcedo-film-lab) has more than 90 decisions, and a person can
no longer keep them in agreement.

This lab keeps one statement of the present knowledge for each category, and changes that statement
in place when a new result arrives.

## Categories

| File | Experiment data |
| --- | --- |
| [datasets.md](datasets.md) | Training and evaluation data: content, cameras, size, license |
| [methods.md](methods.md) | Published methods and public weights: design, reported results, license |
| [quality.md](quality.md) | Image quality that the lab measured: metrics and checks by eye |
| [training.md](training.md) | Training runs: data, loss, settings, time, cost |
| [inference_cost.md](inference_cost.md) | Run time, memory, and numeric precision of a model on each backend |

Add a category only when a result fits none of these files.

## Structure of a findings file

Each file has two parts.

1. **Current state.** A short text that states what holds now for the category. It is the only
   place that a reader must read. Each statement names the date of the entry that supports it.
2. **Entries.** One section for each result, with the date first in the heading
   (`### 2026-11-17 Title`). The newest entry is at the top.

An entry states:

- **Source**: the run, the script and command, the result file under `results/`, or the published
  page. A statement from a published page gives the URL.
- **Result**: the numbers or the observation. State what was not measured.
- **Effect**: the sentences of Current state that this entry adds or changes.

## Rules

- **Change Current state in place.** A new result edits the text of Current state in the same
  commit as its entry. Current state has no history; Git and the entries hold the history.
- **Keep entries; mark the replaced ones.** When a new entry replaces the result of an older entry,
  add the line `Replaced by the entry of <date>.` below the heading of the older entry. Do not
  delete the older entry and do not edit its result.
- **Refer to an entry by its file and date**, for example `datasets.md, 2026-11-17`. Do not give
  entries, findings, or choices an identifier (`D1`, `U3`, `F7`).
- **Do not write a decision list** in any file of this repository. A choice that follows from a
  result is a sentence in Current state, with the entry that supports it. A constraint from the user
  is a sentence in the section "Constraints from the user" of [the plan](../plan.md), and is also
  changed in place.
- **One statement, one place.** Before you add a sentence to Current state, search the findings
  files for a sentence about the same subject and change that sentence.
- **Separate what was measured from what was read.** Mark a number from a paper or a project page
  as reported by its authors until a run of this lab reproduces it.
