# Inference cost

Run time, memory, and numeric precision of a model on each backend. The rules of this file are in
[README.md](README.md).

## Current state

- The lab has no cost measurement. Step 5 of [the plan](../plan.md) makes the first one.
- Estimate, not a measurement: the Bayer model costs 55 GMAC per megapixel as reported
  ([methods.md](methods.md), 2026-11-17, RawNIND models), which is about 2.5 TMAC for a
  45 megapixel photo. At the 1.6 to 2.0 TFLOP/s that the FP32 convolution kernels of Alcedo Studio
  reach on an RTX 3080 Laptop GPU, this is about 3 seconds for one photo. Alcedo Studio therefore
  stores the denoised result and computes it one time for each photo.

## Entries

No entries.
