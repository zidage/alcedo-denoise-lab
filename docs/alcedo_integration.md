# Specification of the denoising model for Alcedo Studio

This file states what the forward code of Alcedo Studio must do to give the output of the lab
model. The integration plan is in the Alcedo Studio repository:
`docs/roadmap/alcedo_studio/raw-processing/ai_denoise/ai_denoise_integration_plan.md` (tracking
issue zidage/AlcedoStudio#1030).

Each statement about the authors' code gives the file and the lines that prove it. The paths are in
[rawnind_jddc](https://github.com/trougnouf/rawnind_jddc) at commit `e80fb7b1` (the commit of
`models/dark_area_loss/run.json`), below `src/`. A statement about a measurement names the result
file under `results/export/` and the entry of
[findings/inference_cost.md](findings/inference_cost.md).

This file covers the weights file, the tensors, the input preparation, the tiles, and the output.
The highlight handling and the stored result format are added to this file when Alcedo Studio
selects them from the measurements of step 7 of [the plan](plan.md)
([findings/quality.md](findings/quality.md), 2026-11-21).

## Weights file

| Item | Value |
| --- | --- |
| File | `utnet2_bayer_dark_loss_10k.safetensors`, 31,047,448 bytes |
| SHA-256 | `0bfb9ddb07e6e6c20e6764dfc30336d9314d5760f875992f2144f13dd0df13a5` |
| Source | `models/dark_area_loss/checkpoint_10000.pt` (step 5), SHA-256 `b2244dec55f9f0ee82eaded6b9b053b7460a60638814ed1f2faa5ea1bb7a5818` |
| Download | Release asset of this repository, see [scripts/export/README.md](../scripts/export/README.md) |
| Parameters | 7,760,748 in 46 tensors, float32 |
| License | GPL-3.0 and CC BY 4.0, the licenses of the public RawNIND weights that the model starts from ([findings/methods.md](findings/methods.md)) |

Metadata of the file (all values are strings):

| Key | Value | Use in Alcedo Studio |
| --- | --- | --- |
| `architecture` | `utnet2_bayer_funit32` | The loader stops when the value is different |
| `leaky_relu_slope` | `0.2` | The negative slope of each LeakyReLU. The loader stops when the key is absent |
| `input_preparation_version` | `1` | The version of the section "Input preparation" of this file. The stored result records it |
| `checkpoint_sha256`, `checkpoint_step` | The source checkpoint | Record only |
| `license`, `license_note` | `GPL-3.0 AND CC-BY-4.0` and the attribution | Shown with the model |
| `authors_commit` | `e80fb7b14dc440bba38c01f312fd3cc02deb566c` | Record only |
| `layout` | The tensor layout below | Record only |

The slope 0.2 is the value of `get_activation_class_params("LeakyReLU")`
(`rawnind/models/raw_denoiser.py:41-42`).

## Tensors and topology

The tensor names are the names of the PyTorch state dict. The layout is the PyTorch layout:
Conv2d weights are OIHW (output channels, input channels, height, width); ConvTranspose2d weights
are IOHW (input channels, output channels, height, width). Each bias has one value for each output
channel. The tensors are in the order of the forward pass, and the loader finds each tensor by
its name. Alcedo Studio changes the layout when it loads the file. The table of all 46 names and
shapes is `results/export/utnet2_bayer_dark_loss_10k_tensors.json`.

The forward pass (`rawnind/models/raw_denoiser.py:66-131` for the layers, `:142-155` for the order):

| # | Tensor prefix | Operation | Channels | Input of the layer |
| ---: | --- | --- | --- | --- |
| 1 | `convs1.0` | conv 3 x 3, pad 1, LeakyReLU | 4 to 32 | the packed input |
| 2 | `convs1.2` | conv 3 x 3, pad 1, LeakyReLU | 32 to 32 | 1 (output `l1`) |
| 3 | `convs2.0` | conv 3 x 3, pad 1, LeakyReLU | 32 to 64 | max pool 2 x 2 of 2 |
| 4 | `convs2.2` | conv 3 x 3, pad 1, LeakyReLU | 64 to 64 | 3 (output `l2`) |
| 5 | `convs3.0` | conv 3 x 3, pad 1, LeakyReLU | 64 to 128 | max pool of 4 |
| 6 | `convs3.2` | conv 3 x 3, pad 1, LeakyReLU | 128 to 128 | 5 (output `l3`) |
| 7 | `convs4.0` | conv 3 x 3, pad 1, LeakyReLU | 128 to 256 | max pool of 6 |
| 8 | `convs4.2` | conv 3 x 3, pad 1, LeakyReLU | 256 to 256 | 7 (output `l4`) |
| 9 | `bottom.0` | conv 3 x 3, pad 1, LeakyReLU | 256 to 512 | max pool of 8 |
| 10 | `bottom.2` | conv 3 x 3, pad 1, LeakyReLU | 512 to 512 | 9 |
| 11 | `up1` | transposed conv 2 x 2, stride 2 | 512 to 256 | 10 |
| 12 | `tconvs1.0` | conv 3 x 3, pad 1, LeakyReLU | 512 to 256 | concat(11, 8) |
| 13 | `tconvs1.2` | conv 3 x 3, pad 1, LeakyReLU | 256 to 256 | 12 |
| 14 | `up2` | transposed conv 2 x 2, stride 2 | 256 to 128 | 13 |
| 15 | `tconvs2.0` | conv 3 x 3, pad 1, LeakyReLU | 256 to 128 | concat(14, 6) |
| 16 | `tconvs2.2` | conv 3 x 3, pad 1, LeakyReLU | 128 to 128 | 15 |
| 17 | `up3` | transposed conv 2 x 2, stride 2 | 128 to 64 | 16 |
| 18 | `tconvs3.0` | conv 3 x 3, pad 1, LeakyReLU | 128 to 64 | concat(17, 4) |
| 19 | `tconvs3.2` | conv 3 x 3, pad 1, LeakyReLU | 64 to 64 | 18 |
| 20 | `up4` | transposed conv 2 x 2, stride 2 | 64 to 32 | 19 |
| 21 | `tconvs4.0` | conv 3 x 3, pad 1, LeakyReLU | 64 to 32 | concat(20, 2) |
| 22 | `tconvs4.2` | conv 3 x 3, pad 1, LeakyReLU | 32 to 32 | 21 |
| 23 | `output_module.0` | conv 1 x 1, then PixelShuffle(2) | 32 to 12, then 3 at 2 times the size | 22 |

- The padding of the 3 x 3 convolutions is zero padding (`nn.Conv2d(..., 3, padding=1)`, default
  `padding_mode="zeros"`).
- The transposed convolutions have no padding and no output padding, so each input pixel gives
  one 2 x 2 output block.
- A concatenation puts the upsampled tensor first: channels 0 to A-1 are the output of the
  transposed convolution, channels A to A+B-1 are the encoder output (`torch.cat([up(x), skip],
  dim=1)`, lines 149-152).
- PixelShuffle(2): output channel `c` (R, G, B for `c` = 0, 1, 2) at the position `(dy, dx)` of
  each 2 x 2 cell is channel `4c + 2dy + dx` of the 1 x 1 convolution.
- The 4 max pooling layers divide the size by 16, so the packed height and width of one forward
  pass are multiples of 16.
- The activations are large: the largest value of the output layer is about 7.7e4 on the random
  input and 1.7e4 on the RawNIND crop, before the gain match
  (`results/export/expected_outputs_manifest.json`). FP32 is required
  ([findings/inference_cost.md](findings/inference_cost.md), 2026-11-18, 16-bit float).

## Input preparation (version 1)

The input is the packed RGGB image: 4 channels (R, G1, G2, B) at half the mosaic size. Channel 0 is
the mosaic at (even row, even column), 1 at (even, odd), 2 at (odd, even), 3 at (odd, odd)
(`rawnind/libs/raw.py:332-353`).

### Black level and white level

The authors' preparation (`raw.py:369-401`, called at `raw.py:307-308`):

```text
value = (raw - black) / (white_level - black)
```

- `white_level` is LibRaw `color.maximum` (rawpy `white_level`, `raw.py:245`), one value for all
  colors. The authors do not use the per-channel `linear_max` (`compat=True`, `raw.py:384-385`).
- `black` is `black_level_per_channel[0]` for all four CFA positions: the loop of `raw.py:380`
  runs over `img.shape[-3]`, which is 1 for the mosaic.
- No value is clipped. Values below the black level stay negative and values above the white
  level stay above 1 (the RawNIND crop of the expected outputs has input values from -0.0117 to
  0.198). No function between the raw file and the network input clips
  (`raw.py:295-309`, `common/libs/np_imgops.py:70-71`, `common/libs/pt_helpers.py:59-63`).

Alcedo Studio:

- Uses LibRaw `color.maximum` as the white level, as the authors do. On the Sony ILCE-7C files of
  RawNIND, `maximum` is 16,383 and `linear_max` is 15,360, so a saturated pixel has the value
  (15,360 - 512) / (16,383 - 512) = 0.9355, not 1. The highlight handling of the stored result
  must use this value domain (the measurement of step 7 decides how).
- Subtracts the black level of each CFA position (LibRaw `black` + `cblack`). The model is
  trained with the first value for all positions. The two agree when the four values are equal.
  On one local RawNIND file of each of the 23 local scene folders, the four values differ by at
  most 2 levels, and only on Canon files (EOS 7D, 6D, 500D, M100). That is at most 1.8e-4 of the
  value range (2 / (13,584 - 2,068), EOS 7D).
- Does not clip, as the authors do not.

### White balance

The authors do not apply a white balance to the network input. The input path
`pt_helpers.fpath_to_tensor` -> `np_imgops.img_fpath_to_np_flt` (`np_imgops.py:70-71`) ->
`raw.raw_fpath_to_rggb_img_and_metadata` (`raw.py:452-456`) -> `raw_fpath_to_mono_img_and_metadata`
(`raw.py:67-309`) has no call of `apply_whitebalance`. The only call in `raw.py` is in its
`__main__` demonstration (`raw.py:945-953`). The training images of this lab use the same function
(`scripts/training_patches/make_training_images.py:163-164`).

Alcedo Studio gives the mosaic without white balance to the model.

### CFA phase

The model expects the RGGB phase. The authors crop the mosaic to move another phase to RGGB
(`raw.py:87-130`): for GBRG they remove the first and the last row, for BGGR the first and the
last row and column, for GRBG the first and the last column. Before that, they remove the top and
left margins of LibRaw and crop to the smallest of `height`, `iheight`, and `raw_height`
(`raw.py:161-191`), and after it they crop the bottom and the right to a multiple of 4 mosaic
pixels (`raw.py:207-229`).

Alcedo Studio does not remove rows or columns of the area that it develops. It extends the mosaic
by reflection about the edge pixel, without a repeat of the edge pixel (row -1 is row 1, column -1
is column 1). This reflection keeps the color of each position, because rows `-k` and `k` have the
same parity:

1. If the pattern of the area is GBRG or BGGR, add one row at the top.
2. If the pattern of the area is GRBG or BGGR, add one column at the left.
3. If the height or the width is then odd, add one row at the bottom or one column at the right
   by the same reflection.

The mosaic then starts with R at (0, 0). The output rows and columns of the added mosaic rows and
columns are not part of the result. So each pixel of the area gets one output pixel, also the
rows and columns that the authors remove.

### Size alignment

The authors crop the packed image at the bottom and the right to a multiple of 16
(`rawnind/tools/denoise_image.py:103-106`, `crop_to_multiple=16`; `common/libs/pt_ops.py:126-131`).

Alcedo Studio does not crop. A tile is filled by reflection of the packed image (each of the 4
channels separately) to a multiple of 16, and the output of the filled area is not part of the
result. The tile size and the overlap between tiles are in the section "Tiles".

### Exposure

The authors apply no gain to the network input at inference (`denoise_image.py:181-190`). In
training, the noisy input of a pair is multiplied by the gain that matches its exposure to the
clean image (`scripts/loss_check/train_bayer.py:285`). Alcedo Studio applies no gain to the input.

## Tiles

The model runs in square tiles of the packed image, so that the GPU holds only the weights, the
packed image, and the workspace of one tile. `scripts/export/measurement_common.py`
(`plan_axis`) is the reference planner:

- The tile size `T` and the overlap `O` are multiples of 16 packed px, and `T > 2 O`.
- On an axis of `N` packed px with `N <= T`: one tile, whose input is the whole axis filled by
  reflection to a multiple of 16.
- Otherwise: the first tile has its input at `[0, T)` and keeps `[0, T - O)`. Each next tile
  keeps `T - 2 O` px and has its input from `O` px before to `O` px after its kept range. The last
  tile has its input at `[N - T, N)` and keeps the rest of the axis. The kept ranges cover the
  axis exactly one time.
- So every tile input starts at a multiple of 16 except the last tile on an axis whose length is
  not a multiple of 16. Each kept range has `O` px or more of input on each side that is not an
  image border.

**Overlap: 96 packed px.** With this overlap, the tiled output in IEEE FP32 is the one-pass output:
the difference is 0 for tiles of 512, 768, and 1024 packed px on three crops, and at most 7.8e-7
after the +2 EV development for tiles of 256 (another cuDNN algorithm for that size). At 80 packed
px, the border difference is 5.5e-4 to 4.8e-3 after the development, 3 to 30 times the difference
between TF32 and IEEE FP32. A smaller overlap is not correct; a larger overlap does not change the
output ([findings/inference_cost.md](findings/inference_cost.md), 2026-11-21, tile overlap).

**Tile size.** Use the largest tile whose workspace fits in the free GPU memory. With an overlap of
96, a tile of 1024 keeps 66% of its area and a tile of 2048 keeps 82%, so a larger tile needs less
time for one photo. The GPU memory and the time of each tile size are in
[findings/inference_cost.md](findings/inference_cost.md) (2026-11-21, PyTorch time of the model
of step 5).

## Output and gain match

The output has 3 channels at the mosaic size: linear camera RGB without white balance, in the
color space of the camera that LibRaw `rgb_xyz_matrix` describes. Channel `c` is the color of the
CFA positions of color `c` in the input. The color matrix is not applied: the authors convert the
output to linear Rec.2020 after the network with `inv(rgb_xyz_matrix[:3])` and a fixed XYZ to
Rec.2020 matrix (`rawnind/libs/rawproc.py:526-551`). Alcedo Studio applies its own color pipeline
after the stored result, as after its demosaic.

The output scale is not defined by the training. The training loss matches the gain of the output
to the clean image before the comparison (`process_net_output`,
`rawnind/libs/abstract_trainer.py:1618-1647`, with `match_gain: output` of
`rawnind/config/train_denoise.yaml:22`, a configuration file of every denoiser training,
`abstract_trainer.py:2097-2100`; the public Bayer weights have `mgout` in their name). The lab
training does the same at
`scripts/loss_check/train_bayer.py:484-486`). On the RawNIND crop of the expected outputs, the
output is about 1e5 times the input, and the gain match factor is 9.69e-6. So a gain match is
required.

### Definition of the gain match factor

The authors' definition (`denoise_image.py:120-146`, `rawproc.py:135-151`): the reference is the
OpenCV edge-aware demosaic of the input (`rawproc.py:554-566`, `raw.py:459-496`), converted to
linear Rec.2020. The factor is

```text
g = mean(Rec2020(reference)) / mean(Rec2020(output))
```

where the mean is over all pixels and all three channels of the whole image.

Alcedo Studio uses the same quotient in a form that it can add up tile by tile. With `M` the
camera RGB to linear Rec.2020 matrix and `w_c` the sum of column `c` of `M`:

```text
g = sum_c w_c * mean_in(c) / sum_c w_c * mean_out(c)
```

- `mean_in(c)`: the mean of the input mosaic values of color `c` (for G, both G positions
  together), after the preparation above.
- `mean_out(c)`: the mean of output channel `c`.
- The pixels: the pixels of the result, that is the area that Alcedo Studio develops, without
  the rows and columns added by reflection and without the alignment fill. Negative values and
  values above the white level are included, as in the authors' mean.
- One factor for the whole photo. The stored result is `g` times the output.

The mean of a Rec.2020 image over its three channels is `(1/3) sum_c w_c mean(c)`, so the two
forms differ only by the difference between the mean of the demosaic and the mean of the mosaic
samples of each color. On the RawNIND crop the two factors differ by 3.4e-4 of the factor
(`results/export/expected_outputs_manifest.json`, `gain_match_rawnind_crop_512`). After the gain
match, the channel means of the output are 0.997, 1.000, and 1.015 times the means of the R, G,
and B mosaic values, which agrees with an output without white balance.

After the gain match, the output is in the value domain of the input: 0 is the black level, 1 is
LibRaw `maximum` minus the black level.

## Expected outputs and tolerance

`scripts/export/make_expected_outputs.py` makes three files for the forward tests of Alcedo Studio
(release asset, [scripts/export/README.md](../scripts/export/README.md)):

| File | Input | Content |
| --- | --- | --- |
| `random_64.safetensors` | 4 x 64 x 64, uniform 0 to 1, seed 20261121 | `input`, `output` (3 x 128 x 128), and `layer.<prefix>` for each of the 23 layers |
| `random_256.safetensors` | 4 x 256 x 256, the next tensor of the same generator | `input`, `output` (3 x 512 x 512) |
| `rawnind_crop_512.safetensors` | 4 x 512 x 512 from the centre of the packed image of `Bayer_TEST_TitusToys_ISO6400` (Sony ILCE-7C, RawNIND test split, CC BY-SA 4.0), prepared as the authors prepare it | `input`, `output` (3 x 1024 x 1024), `rgb_xyz_matrix` (3 x 3) |

- The tensors are float32 in the NCHW layout without the batch axis. The outputs are before the
  gain match.
- A layer output is the output after LeakyReLU for a 3 x 3 convolution, the output of a
  transposed convolution, and the output after PixelShuffle for layer 23.
- The expected outputs come from PyTorch 2.11.0 on the GPU in IEEE FP32 (TF32 off, deterministic
  cuDNN algorithms).

Comparison rule of the Alcedo Studio tests: a tensor agrees with the expected tensor when

```text
max |alcedo - expected| <= 1e-5 * max |expected|
```

The manifest `results/export/expected_outputs_manifest.json` gives the product as `tolerance_abs`
for each tensor. The value 1e-5 is 4 times the largest difference between the CPU and the GPU
output of PyTorch over all 26 compared tensors (2.4e-6 of the largest value, layer `tconvs1.0`), rounded up
to one significant digit ([findings/inference_cost.md](findings/inference_cost.md), 2026-11-21).
