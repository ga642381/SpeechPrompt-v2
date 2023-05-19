# SpeechPrompt-v2

![](https://ga642381.github.io/SpeechPrompt/SpeechPrompt-v2-assets/SpeechPrompt-v2-12tasks-framework.png)

- GitHub: https://github.com/ga642381/SpeechPrompt-dev
- Website: https://ga642381.github.io/SpeechPrompt/
- Pipeline Charts: [place holder]
- Datasets: [place holder]

## :elephant: Pre-trained models and files

There are 4 files you will be having:

1. **HuBERT model**: encoding speech
2. **K-means model**: quantizing the speech representations into discrete units
3. **dictionary file**: defining the unit space for the unit language model.
4. **unit Language Model (uLM)**: performing generative language modeling on the disrete units

These models can be automatically downloaded when running preprocessing pipeline.

## :wrench: Preprocessing

### Concept

![](https://i.imgur.com/keY07YP.png)

- There are 4 steps in the data preprocess (Speech2unit) pipline. The main task here is to perform speech-to-units and collating the task labels

  1. **generate manifest**
  2. **quantize**
  3. **reduce_quantized**
  4. **create_lm_dataset**

- We save intermediate data in each step so that we can do further analysis on the data that we are interested in. Also, you can better understand how it works by checking each intermediate data.

### Steps

1. Download the dataset
2. Modify the **dataset config** ([downstream]/config.yaml)
3. Modify the **global config** (preprocess/config.yaml)
4. Run **Preporcess/runner.py**
   - option 1
   ```shell
   # You can run --action all to run through all the 4 stages:
   python runner.py --model GSLM --downstream SCR_google_speech_commands --action
   ```
   - option 2
   ```shell
   # Or you can run through these 4 stages sequentially by the following command:
   python runner.py --model GSLM --downstream SCR_google_speech_commands --action generate_manifest
   python runner.py --model GSLM --downstream SCR_google_speech_commands --action quantize
   python runner.py --model GSLM --downstream SCR_google_speech_commands --action reduce_quantized
   python runner.py --model GSLM --downstream SCR_google_speech_commands --action create_lm_dataset
   ```

## :arrows_counterclockwise: Verbalizer

### Concept

![](https://i.imgur.com/FUs9uTE.png =x150)

- There are 2 steps in Verbalizer, which maps the task labels into language model's vocabulary.

### Steps

```shell
python verbalizer.py --downstream SCR_google_speech_commands --action all --method freq
```

## :fish: Fairseq Preprocess

### Concept

![](https://i.imgur.com/WRH8evd.png =x100)

This step converts the verbalized data to binary files that will be used for fairseq training.

### Steps

```shell
python fairseq_preprocess.py --downstream SCR_google_speech_commands --vb_method freq
```

## :fire: Training

![](https://i.imgur.com/r1H08Kl.png =x150)

- During training, 2 kinds of checkpoints will be saved
  - base_model
  - prompt
  -

## :black_nib: Sampling

### Concept

![](https://i.imgur.com/yP0ECAS.png =x150)

### Steps

```shell
python sample.py \
    --exp_name SCR_lt_speech_commands_plen.5 \
    --downstream SCR_lt_speech_commands \
    --vb_method freq
```

- The output is a json file containing the file_name, source units, ground truth (label), and model prediction:
  ![](https://hackmd.io/_uploads/S1sAWiVBn.png)
