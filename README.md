# Error-driven Fixed-Budget ASR Personalization for Accented Speakers (ICASSP 2021) 

This repository provides an implementation of experiments in our [ICASSP-2021 paper](https://arxiv.org/abs/2103.03142)
```
@inproceedings{awasthi2021error,
  title={Error-Driven Fixed-Budget ASR Personalization for Accented Speakers},
  author={Awasthi, Abhijeet and Kansal, Aman and Sarawagi, Sunita and Jyothi, Preethi},
  booktitle={ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={7033--7037},
  year={2021},
  organization={IEEE}
}
```  

# Requirements
This code was developed with python 3.8.9. <br/>
Create a new virtual environment and install the dependencies by running `bash install_requirements.sh`

# Dataset
IndicTTS dataset used in our experiments can be obtained by contacting [IndicTTS](https://www.iitm.ac.in/donlab/tts/contact.php) team at `smtiitm@gmail.com`. Please mention that you require data for all the accents in Indian English.

The dataset consists of utterances from various accented speakers. The speaker utterances are in the format \<accent>_\<gender>_english.zip. The data can be unzipped in appropriate format using the `data/indicTTS_audio/unzip_indic` scripts.
```
cd data/indicTTS_audio
python3 unzip_indic.py --input_path <path_to_speaker_zip_file>
```

# Usage
  * Generate transcripts for the seed+dev set using the pre-trainded ASR (Transcripts are used while training error models)
    ```
    cd models/quartznet_asr
    bash scripts/infer_transcriptions_on_seed_set.sh
    ```
  * Train error model by aligning the references and generated transcripts for seed+dev set
    ```
    cd models/error_model
    bash train_error_model.sh
    ```
  * Infer the trained error model on the set of sentences from which we wish to do the selection
    ```
    cd models/error_model
    bash infer_error_model.sh
    ```
  * Select the sentences using the error model as proposed in our paper (Equation-2 and Algorithm-1)
    ```
    cd models/error_model
    bash error_model_sampling.sh
    ```
  * Finetune and Test the ASR on the sentences selected via error model
    ```
    cd models/quartznet_asr
    bash scripts/finetune_on_error_model_seleced_samples.sh
    bash scripts/test_ASR_finetuned_on_error_model_sents.sh
    ```
  * Finetune and Test the ASR on the randomly selected sentences
    ```
    cd models/quartznet_asr
    bash scripts/finetune_on_randomly_seleced_samples.sh
    bash scripts/test_ASR_finetuned_on_random_sents.sh
    ```

# Contents
  * `data/$accent/manifests`:
    - `all.json`: All the samples for a given accented speaker
    - `seed.json`: Randomly selected seed set for learning error model. Also used with selected samples while training ASR.
    - `dev.json`: Dev set used while training ASR. Also used with seed set while training error models
    - `seed_plus_dev.json`: Used for training error models (Concatenation of `seed.json` and `dev.json`)
    - `selection.json`: Sentences are selected from this file (either randomly or through error model, depending on selection startegy)
    - `test.json`: Used for evaluating the error model.
    - `train/error_model/$size/seed_"$seed"/train.json`: Contains `size` number of sentences selected via error model from `selection.json`, appended with `seed.json` for training the ASR model. `seed` represents an independent run.
    - `train/random/$size/seed_"$seed"/train.json`: Contains `size` number of sentences selected randomly from `selection.json`, appended with `seed.json` for training the ASR model. `seed` represents an independent run.
  * `data/indicTTS_audio`:
    - Audio files for each speaker are placed here.
    - After obtaining `.zip` files from [IndicTTS team](https://www.iitm.ac.in/donlab/tts/contact.php), place them in this folder.
    - Run `unzip_indic.sh` to unzip the speaker data of your choice.
  * `models/pretrained_checkpoints/quartznet`: Pretrained Quartznet models:
    - `librispeech/quartznet.pt`: Quartznet ckpt trained on 960hrs librispeech
    - `librispeech_mcv_others/quartznet.pt`: Quartznet ckpt trained on 960hrs librispeech + Mozilla Common Voice + Few other corpus
    - The above checkpoints were originally provided by NVIDIA to work with NeMO toolkit. We modified these slightly to make them work with Jasper's pytorch code.
    - Link to original checkpoints: [Librispeech](https://ngc.nvidia.com/catalog/models/nvidia:quartznet_15x5_ls_sp), [Librispeech+MCV+Others](https://ngc.nvidia.com/catalog/models/nvidia:tlt-jarvis:speechtotext_english_quartznet)
  * `models/pretrained_checkpoints/error_models`
    - `$accent/seed_"$seed"/best/ErrorClassifierPhoneBiLSTM_V2.pt` : Pre-trained error model on `seed_plus_dev.json` outputs.
    - `librispeech/seed_"$seed"/best/ErrorClassifierPhoneBiLSTM_V2.pt`: Pre-trained error model on `dev+test` sets of librispeech.
  * `models/error_model` : Code related to error model
    - `train_error_model.sh`: Trains the error model using the ASR's transcripts on the seed+dev set and gold references
    - `infer_error_model.sh`: Dumps aggregated error probabilities from the error model for sentence scoring
    - `error_model_sampling.sh`: Samples the sentences using error model outputs as per Equation-2 and Algorithm-1 in our paper.
  * `models/quartznet_asr`: Code related to training and inference of [Quartznet model](https://arxiv.org/pdf/1910.10261.pdf), adapted from Nvidia's [implementation of Jasper in PyTorch](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechRecognition/Jasper)
    - `scripts/finetune_on_error_model_seleced_samples.sh`: Finetunes the ASR on sentences utterances selected by error model
    - `scripts/finetune_on_randomly_seleced_samples.sh`: Finetunes the ASR on randomly selected sentences
    - `scripts/test_ASR_finetuned_on_error_model_sents.sh`: Infers the ASR finetuned on error model selected sentences
    - `scripts/test_ASR_finetuned_on_random_sents.sh`: Infers the ASR on randomly selected sentences
    - `scripts/infer_transcriptions_on_seed_set.sh`: Infers the ASR on seed+dev set. Inferred outputs used for training the error model.

# Acknowledgement
  * Dataset for a wide variety of Indian accents was provided by [IndicTTS](https://www.iitm.ac.in/donlab/tts/contact.php) team at [IIT Madras](https://www.iitm.ac.in/). Dataset for other accents was obtained from [L2-Arctic Corpus](https://psi.engr.tamu.edu/l2-arctic-corpus/)
  * Code for Quartznet ASR model is adapted from Nvidia's [implementation of Jasper in PyTorch](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechRecognition/Jasper). Pretrained quartznet checkpoints were downloaded from [here](https://ngc.nvidia.com/catalog/models/nvidia:quartznet_15x5_ls_sp) and modified slightly to make them compatible with Jasper's PyTorch implementation.
  * [Power-ASR](https://github.com/NickRuiz/power-asr) is used for aligning ASR-generated transcripts and references for obtaining error labels.
  * Park et al's [Grapheme-to-Phoneme](https://github.com/Kyubyong/g2p) model is used for converting graphemes to phonemes as an input to the error model . 
