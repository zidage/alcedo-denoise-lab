# Scripts

The scripts of the lab, committed. One folder for each step of [the plan](../docs/plan.md), for
example `scripts/reference_measurement/`.

- Each step folder has a `README.md` that states the commands in the order of their use, and a
  `requirements.txt`.
- A script reads from `data/` and `models/`, and writes its text results to `results/<step>/`.
- A script that is used one time to examine a problem stays under `build/tmp/` and is not
  committed. A script that makes a number of a findings entry is committed here.
