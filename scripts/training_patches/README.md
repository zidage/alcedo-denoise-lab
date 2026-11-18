# Training patches (step 2)

Makes the training data of the Bayer model from RawNIND: the clean targets demosaiced by
`demosaic_target` (the original Bayer DemosaicNet), the packed Bayer inputs, and the alignment and
loss mask of each (clean, other) pair of the authors. The results go to
`results/training_patches/`. The findings are in
[docs/findings/datasets.md](../../docs/findings/datasets.md).

## Layout of the training data

The training data is stored as aligned images, one file for each image, and the training reads
random crops of `patch_size` (128 x 128 packed, 256 x 256 RGB) from them, as the authors' training
does. A fixed grid of patches for each pair repeats each clean image for each of its pairs: in the
check of 8 scenes it is 4.1 times larger (docs/findings/datasets.md).

`data/patches/rawnind_bayer_demosaicnet/` (not committed):

| Path | Content |
| --- | --- |
| `clean_rec2020/<scene>/<clean file>.npy` | Clean target, (H, W, 3) float16, linear Rec.2020: DemosaicNet, then the authors' camera RGB to Rec.2020 |
| `packed_bayer/<scene>/<file>.npy` | Network input, (H/2, W/2, 4) float16, the authors' packed RGGB; for each clean and each noisy file |
| `pair_masks/<scene>/<clean sha1>_<other sha1>.npy` | Loss mask of an admitted pair, `np.packbits(mask, axis=1)`, in the aligned frame |
| `clean_targets.jsonl`, `packed_inputs.jsonl`, `pairs.jsonl` | One line for each file and each pair: camera, ISO, alignment, gains, color matrix, admission |
| `manifest.json`, `sha256sums.txt` | Command, commits, counts, bytes, time; SHA-256 of each `.npy` file |

Each pair has the camera of its scene, so a training run removes its held-out camera when it reads
the data (`training_crops.TrainingImages(exclude_cameras=...)`), and one set of files serves all
runs. The test split, the 4 research-only scenes, and the 5 Canon EOS M100 scenes outside the test
split are not in the training data (`list_bayer_scenes.py`).

## Environment

```bash
E=build/tmp/training_patches
py -3.14 -m venv $E/venv
$E/venv/Scripts/python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
$E/venv/Scripts/python -m pip install -r scripts/training_patches/requirements.txt
```

The authors' code is the clone of step 1
(`build/tmp/reference_measurement/rawnind_jddc`, commit e80fb7b). exiftool must be on PATH.

## Commands, in order

```bash
E=build/tmp/training_patches
P="$E/venv/Scripts/python -I"
A=build/tmp/reference_measurement/rawnind_jddc
OUT=data/patches/rawnind_bayer_demosaicnet

# 1. Bayer scenes: camera (EXIF of the first 256 KiB of one file of each scene), role, files.
$P scripts/training_patches/list_bayer_scenes.py --test-reserve $A/src/rawnind/config/test_reserve.yaml

# 2a. Development machine: the first training scene (by name) of each camera, two of the
#     Sony ILCE-7C (8 scenes, 2.5 GB).
$P scripts/training_patches/download_rawnind_scenes.py --scenes 2pilesofplates \
  European_hornbeam_bark MuseeL-C500D-mammal Bark 7D-1 D60-2 Iain01 bananapi
# 2b. Rented machine: all training scenes, 16 files at the same time.
$P scripts/training_patches/download_rawnind_scenes.py --all-train --parallel 16

# 3. Clean targets (GPU), then packed inputs and pairs (CPU, 4 processes). A stopped run
#    continues. Use the same --scenes or --all-train as in 2.
$P scripts/training_patches/make_training_images.py --authors-repo $A --tmp-dir $E/work/make \
  --out $OUT --scenes 2pilesofplates European_hornbeam_bark MuseeL-C500D-mammal Bark 7D-1 \
  D60-2 Iain01 bananapi

# 4. Checks, crop positions, read rate, estimate of the full set, and one sheet for each camera.
$P scripts/training_patches/check_training_images.py --images $OUT --authors-repo $A \
  --tmp-dir $E/work/check --out results/training_patches/check_development_machine.json --sheets
```

## Rented run (RunPod)

`run_rented_machine.sh` runs commands 2b, 3, and 4 on all 175 training scenes on a RunPod pod. It
installs uv with its standalone installer (the system Python of the Ubuntu 24.04 image refuses
`pip install`), makes a Python 3.14 environment, clones the authors' code at commit e80fb7b, and
writes its logs and the phase times (`results/training_patches/rented_run.json`). The count of CPU
workers comes from the CPU quota and the memory limit of the container: `nproc` and `free` show
the host (96 CPUs and 503 GB on a pod with a quota of 7.65 CPUs and 50 GB).

Pod settings:

| Setting | Value | Reason |
| --- | --- | --- |
| Network volume | 200 GB, mounted at `/workspace`, in a data center that also has the GPUs of steps 3 to 5 | The training data (about 161 GB) stays on it for steps 3 to 5 |
| Container disk | 120 GB | The RawNIND download (91.7 GB) and the environment; deleted with the pod |
| vCPU and memory | As many vCPU as the budget permits, at least 3.5 GiB of memory for each vCPU | One CPU worker for each CPU of the quota; each worker has a peak of 3.1 GB; the CPU work is 32.6 CPU-core hours of the development machine |
| GPU | Any NVIDIA GPU with a driver for CUDA 12.8 (driver 570 or later) | DemosaicNet uses 2.2 GB of GPU memory, 34 GPU minutes in total |
| Image | A RunPod PyTorch template (Ubuntu) | The script installs its own environment |

Commands on the pod:

```bash
cd /workspace
git clone https://github.com/zidage/alcedo-denoise-lab.git
cd alcedo-denoise-lab
nohup bash scripts/training_patches/run_rented_machine.sh > /workspace/step2_run.log 2>&1 &
tail -f /workspace/step2_run.log
```

A pod with a network volume cannot be stopped, only terminated; the network volume stays. The
runpodctl of the pod has no API key, so by default the script does not delete the pod, and the pod
is terminated by hand when the run ends. To delete it at the end, also after an error, give
runpodctl an API key (`runpodctl config --apiKey ...`) and start the script with
`DELETE_POD_AT_END=1` (it then checks `runpodctl get pod` before the download), or create the file
`build/tmp/training_patches/rented_run/DELETE_POD_AT_END` during the run.

## Notes on the procedure

- The alignment and the mask of a pair are computed on the OpenCV demosaic of the two files with
  the authors' function, as for the public weights. Only the clean target uses DemosaicNet.
- Each sheet under `results/training_patches/images/` shows a 512 x 512 crop of one noisy pair of
  a camera: the input after the authors' training-time demosaic and color conversion, the clean
  target, and the mask, after the curve of the developed metric. The sheets show RawNIND images
  (CC BY-SA 4.0) and are not committed.
- `training_crops.py` is the reader of the training runs. Its crop rule is the authors' rule,
  including their effective mask threshold of 1/6 valid pixels (see its docstring).
