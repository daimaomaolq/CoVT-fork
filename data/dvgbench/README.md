# DVGBench Train Metadata

This directory contains DVGBench training metadata used for adapter experiments.

- `dvg_train.jsonl`: 1,990 DVGBench train rows.

The image files are not stored in this repository. Place the extracted images at a
runtime dataset path such as:

```text
/root/autodl-tmp/datasets/DVGBench/images/images/{era,visdrone}
```

The local source copy had corrupted/garbled Chinese text fields in some rows. Use
the English query fields (`question`, `question_e`) plus `bbox`, `image_id`,
`dataset`, `class`, and `split` for training/evaluation conversion.
