# SpeechPrompt-v2 Dataset Document

### :books: Speech Command Recognition (SCR)

#### :books: Google Speech Commands v1

(Follow the download instructions from [s3prl](https://github.com/s3prl/s3prl))

- Download data
  - http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz
  - http://download.tensorflow.org/data/speech_commands_test_set_v0.01.tar.gz
- Download and unpack Speech Commands
  ```
  mkdir -p /CORPORA_DIR/speech_commands_v0.01
  tar zxf speech_commands_v0.01.tar.gz -C /CORPORA_DIR/speech_commands_v0.01
  ```
- Download and unpack Speech Commands test set
  ```
  mkdir -p /CORPORA_DIR/speech_commands_test_set_v0.01
  tar zxf speech_commands_test_set_v0.01.tar.gz -C /CORPORA_DIR/speech_commands_test_set_v0.01
  ```

#### :books: Grabo

- Download data (Grabo dataset)
  - https://www.esat.kuleuven.be/psi/spraak/downloads/
  ```
  wget ftp://ftp.esat.kuleuven.be/psi/speech/vrenkens/grabo.tar.gz
  ```

#### :books: Lithuanian Speech Commands (LT-SCR)

#### :books: Dysarthric Mandarin Speech Commands (DM-SCR)

#### :books: Arabic Speech Commands (AR-SCR)

- Download data from the following links
  - [Lithuanian Speech Commands Dataset](https://github.com/kolesov93/lt_speech_commands)
  - [Dysarthric Mandarin Speech Commands Dataset](https://reurl.cc/a5vAG4)
  - [Arabic Speech Commands Dataset](https://github.com/ltkbenamer/AR_Speech_Database.git)
- Follow the preprocessing method from [Adversarial Reprogramming on Speech Command Recognition](https://github.com/dodohow1011/SpeechAdvReprogram)

### :books: Intent Classification (IC)

(Follow the download instructions from [s3prl](https://github.com/s3prl/s3prl))

- Download and unzip data: Fluent Speech Commands
  - Official data link: http://fluent.ai:2052/jf8398hf30f0381738rucj3828chfdnchs.tar.gz
  - Official website: https://fluent.ai/fluent-speech-commands-a-dataset-for-spoken-language-understanding-research/
  - Since the official link might break occasionally, we provide a backup link. If this is not allowed please let us know and we will remove it immediately.
  - Please use wget http://140.112.21.28:9000/fluent.tar.gz

### :books: Language Identification (LID)

(Follow the download instructions from [Tensorflow](https://www.tensorflow.org/datasets/catalog/voxforge))

- The complete list of archives can be found in https://storage.googleapis.com/tfds-data/downloads/voxforge/voxforge_urls.txt
- Download data with the following command:
  `wget -i voxforge_urls.txt -x`

### :books: Fake Speech Detection (FSD)

- Download the LA portion of ASVspoof 2019 from [ASVspoof Website](https://datashare.ed.ac.uk/handle/10283/3336)
  - https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip?sequence=3&isAllowed=y

### :books: Emotion Recognition (ER)

(Follow the download instructions from [s3prl](https://github.com/s3prl/s3prl))

- Download dataset and unzip. You will need to fill a form in IEMOCAP official website to get the dataset.
  - https://sail.usc.edu/iemocap/

### :books: Accent Classification (AcC)

- Download accentdb_extended from [AccentDB Website](https://accentdb.org/)
  - https://drive.google.com/file/d/1NO1NKQSpyq3DMLEwiqA-BHIqXli8vtIL/view

### :books: Speaker Identitfication (SID)

(Follow the download instructions from [s3prl](https://github.com/s3prl/s3prl))

- Download dataset from [Voxceleb1](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html) and unzip them.

  ```
  voxceleb1_root="/CORPORA_DIR/VoxCeleb1/"
  mkdir -p $voxceleb1_root/dev
  mkdir -p $voxceleb1_root/test

  # prepare dev
  cd $voxceleb1_root/dev/
  wget https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partaa
  wget https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partab
  wget https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partac
  wget https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partad
  cat vox1_dev* > vox1_dev_wav.zip
  unzip vox1_dev_wav.zip

  # prepare test
  cd $voxceleb1_root/test/
  wget https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_test_wav.zip
  unzip vox1_test_wav.zip
  ```

### :books: Gender Identification (GID)

- Download [Voxceleb1](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html) by following the instructions from Speaker Identification
- Download meta data of VoxCeleb1 enrichment for gender identification from [VoxCeleb enrichment for Age and Gender recognition](https://github.com/hechmik/voxceleb_enrichment_age_gender)
  - https://github.com/hechmik/voxceleb_enrichment_age_gender/blob/main/dataset/vox1_meta.csv

### :books: Audio Classification (AuC)

- Download data from [ESC-50](https://github.com/karolpiczak/ESC-50)
  - https://github.com/karoldvl/ESC-50/archive/master.zip

### :books: Sarcasm Detection (SD)

#### :books: MUStARD Dataset

- Download raw videos from https://drive.google.com/file/d/1i9ixalVcXskA5_BkNnbR60sqJqvGyi6E/view
- Download metadata JSON file from https://github.com/soujanyaporia/MUStARD/blob/master/data/sarcasm_data.json
- Convert videos in `utterances_final` to audio through `/GSLM/preprocess/mustard/convert2audio.py` using the following command:
  ```bash=
  pip install ffmpeg-python nlp2
  python <SpeechPrompt_root>/GSLM/preprocess/mustard/convert2audio.py -s <mustard_root>/utterances_final
  ```

#### :books: MUStARD++ Dataset

- Download raw videos from https://drive.google.com/drive/folders/1kUdT2yU7ERJ5KdauObTj5oQsBlSrvTlW
- Download `mustard++_text.csv` from https://raw.githubusercontent.com/cfiltnlp/MUStARD_Plus_Plus/main/mustard%2B%2B_text.csv and put it in the same directory as the raw videos
- Convert videos in `final_utterances_videos` to audio with `/GSLM/preprocess/mustard/convert2audio.py`

### :books: Voice Activity Detection (VAD)
Follow the download instructions from [NVIDIA Nemo](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/speech_classification/datasets.html#speech-command-freesound-for-vad)
* Download [NeMo's Github repo](https://github.com/NVIDIA/NeMo)
* Go to **<NeMo_git_root>/scripts/freesound_download_resample/** and follow the below steps to download [Freesound](https://freesound.org/) (These steps are originated from [freesound_download.py](https://github.com/NVIDIA/NeMo/blob/main/scripts/freesound_download_resample/freesound_download.py))
    * Install required packages
		```
		pip install -r freesound_requirements.txt
		```
    * Create an API key for freesound.org at https://freesound.org/help/developers/
    * Create a python file called **freesound_private_apikey.py** and add lined 
	```
	api_key = <your Freesound api key> 
	client_id = <your Freesound client id>
	```
    * Authorize by run ```python freesound_download.py --authorize ``` and visit the website and paste response code
    * Feel free to change any arguments in download_resample_freesound.sh such as max_samples and max_filesize
    * Run bash download_resample_freesound.sh <numbers of files you want> <download data directory> <resampled data directory>. 
        ```
        bash download_resample_freesound.sh 4000 ./freesound ./freesound_resampled_background
        ```
* Download Google Speech Commands Dataset v2 
    * http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
    * http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz
    * Download and unpack Speech Commands
        ```
        mkdir -p ./speech_commands_v0.02
        tar zxf speech_commands_v0.02.tar.gz -C ./speech_commands_v0.02
        ```
    * Download and unpack Speech Commands test set
        ```
        mkdir -p ./speech_commands_test_set_v0.02
        tar zxf speech_commands_test_set_v0.02.tar.gz -C ./speech_commands_test_set_v0.02
        ```
* Process Google SC v2 & Freesound dataset
    * Modify line 484 in <NeMo_git_root>/scripts/dataset_processing/process_vad_data.py to fixed_test, fixed_val, fixed_train = 60000, 20000, 160000 and run the following command:
        ```
        python <NeMo_git_root>/scripts/dataset_processing/process_vad_data.py --out_dir='./manifest/' --speech_data_root='./speech_commands_v0.02'--background_data_root='./freesound_resampled_background' --log --rebalance_method='fixed'
        ```

