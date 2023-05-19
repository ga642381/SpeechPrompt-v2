# SpeechPrompt-v2

![](https://ga642381.github.io/SpeechPrompt/SpeechPrompt-v2-assets/SpeechPrompt-v2-12tasks-framework.png)

- Website: https://ga642381.github.io/SpeechPrompt/
- Paper Link: https://arxiv.org/abs/2303.00733
- Pipeline Charts: https://github.com/ga642381/SpeechPrompt-v2/blob/main/docs/pipeline.png
- Datasets Doc: https://github.com/ga642381/SpeechPrompt-v2/blob/main/docs/dataset.md

## :elephant: Pre-trained models and files

There are 4 files you will be having:

1. **HuBERT model**: encoding speech
2. **K-means model**: quantizing the speech representations into discrete units
3. **dictionary file**: defining the unit space for the unit language model.
4. **unit Language Model (uLM)**: performing generative language modeling on the disrete units

These models can be automatically downloaded when running preprocessing pipeline.

## :wrench: Preprocessing

### Concept
<img src="https://i.imgur.com/keY07YP.png">

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
<img src="https://i.imgur.com/FUs9uTE.png" height=250>

- There are 2 steps in Verbalizer, which maps the task labels into language model's vocabulary.

### Steps
* run verbalizer.py
* example:
    ```shell
    python verbalizer.py --downstream SCR_google_speech_commands --action all --method freq
    ```

## :fish: Fairseq Preprocess

### Concept
<img src="https://i.imgur.com/WRH8evd.png" height="150">

This step converts the verbalized data to binary files that will be used for fairseq training.

### Steps
* run fairseq_preprocess.py
* example:
    ```shell
    python fairseq_preprocess.py --downstream SCR_google_speech_commands --vb_method freq
    ```

## :fire: Training
### Concept
<img src="https://i.imgur.com/r1H08Kl.png" height="200">

- During training, 2 kinds of checkpoints will be saved
  - base_model
  - prompt

### steps
* run train.py
* example:
    ```shell
    python train.py \
        --downstream SCR_google_speech_commands \
        --vb_method freq \
        --exp_name SCR_google_speech_commands_plen.5 \
        --prompt_length 5 \
        --deep_prompt
    ```
## :black_nib: Sampling

### Concept
<img src="https://i.imgur.com/yP0ECAS.png" height="200">

* Load base_model and prompts to perform sampling

### Steps
* run sample.py
* example:
    ```shell

    python sample.py \
        --exp_name SCR_google_speech_commands_plen.5 \
        --downstream SCR_google_speech_commands \
        --vb_method freq
    ```

- The output is a json file containing the file_name, source units, ground truth (label), and model prediction:
  ![](https://hackmd.io/_uploads/S1sAWiVBn.png)
