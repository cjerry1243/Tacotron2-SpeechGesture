# Tacotron2-SpeechGesture
This is the official repository for our [paper](https://openreview.net/forum?id=gMTaia--AB2) **"The IVI Lab entry to the GENEA Challenge 2022 – A Tacotron2 Based Method for Co-Speech Gesture Generation With Locality-Constraint Attention Mechanism."**

This repository provides the code for co-speech gesture prediction using the data from [GENEA Challenge 2022](https://youngwoo-yoon.github.io/GENEAchallenge2022/).

**This work is also the offical baseline for [GENEA Challenge 2023](https://genea-workshop.github.io/2023/challenge/).**

## Demonstration of Our Results (make sure the audio is on while playing!!) 
https://user-images.githubusercontent.com/26675834/226152214-a4fc1b84-328b-4a22-828e-e8d2a69a2f89.mp4

(from _tst_2022_v1_036.bvh_)

## Environment
- Ubuntu 20.04
- Python 3.9
- Cuda 11.4

To install the required libraries, activate your python environment and use the following command:
```
pip install -r requirements.txt
```

Note:
This repository is only tested under the above environment and package settings. It may still work under different configurations.

## Data Processing
### Download GENEA22 Data
- Get the dataset (v1) from GENEA Challenge 2022 and unzip all files. Put the "dataset_v1" under this repository. Create a "tst" folder under the "dataset_v1" directory and put all test data inside.

Your file hierarchy should look like the following: 
```
dataset_v1/
    trn/
        bvh/*.bvh
        tsv/*.tsv
        wav/*.wav
        trn_metadata.csv
    val/
        bvh/*.bvh
        tsv/*.tsv
        wav/*.wav
        val_metadata.csv
    tst/
        tsv/*.tsv
        wav/*.wav
        tst_metadata.csv
```

### FastText Word Embeddings
Download and unzip the [word embedding](https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip) from FastText. Put the file "crawl-300d-2M.vec" under the project directory.

### Data Preprocessing
Preprocess the data and save them as h5 files. (This might take a while) 
```
python process_data.py -d path_to_your_dataset_v1 
```

By default, the three h5 files (_trn_v1.h5_, _val_v1.h5_, and _tst_v1.h5_) should be generated under the project directory.

Alternatively, you can simply [download](https://drive.google.com/drive/folders/1KW_cN_1pL9KgQxQyyYUATAlgA0CcX9LZ?usp=sharing) our processed data.

Calculate audio statistics based on _trn_v1.h5_:
```
python calculate_audio_statistics.py
```

### Create Motion Processing Pipelines
We will use the following command to generate pipelines (*.sav) for converting our motion representations to euler rotations.
```
python create_pipeline.py
```


## Test
We predict all gestures for both tracks (full and upper body) given the processed test data (tst_v1.h5).

- Download and unzip the [checkpoints](https://drive.google.com/drive/folders/1RNpXTMVmx36PmCESQlVFo-ez9C7oDN5R?usp=sharing). Put the "fullbody" and "upperbody" folders under Tacotron2/ 

- Navigate to Tacotron2/ folder.
```
cd Tacotron2
```

- Predicting fullbody gestures: 
```
python generate_all_gestures.py -ch fullbody/ckpt/checkpoint_21000.pt -t full
```

- Predicting upperbody gestures: 
```
python generate_all_gestures.py -ch upperbody/ckpt/checkpoint_22000.pt -t upper
```

The bvh files should be generated under "Tacotron2/outputs/" folder. By defaut, the cuda device "0" is used. If you prefer to use a different cuda device for inference, please edit line 23 in the Tacotron2/common/hparams.py

Note: 
You can import the bvh files to Blender to visualize the gesture motions. 
If you would like to render the motions, please reach out to the [repository](https://github.com/TeoNikolov/genea_visualizer) provided by GENEA Challenge 2022. 

## Train  
Edit the configurations in Tacotron2/common/hparams.py. 

1. Enter the "output_directory" and "device" of your own choice. 

2. Add the "checkpoint_path" if you would like to resume training. 

3. Make sure to edit "n_acoustic_feat_dims", 78 for full body and 57 for upper body. 

4. Then train the model using the following command:
```
cd Tacotron2
python train_genea22.py
```

The weights and logs can be found under the "output_directory". It takes roughly 20k~30k iterations to produce decent results.


## Dyadic Gestures  
We adapt the system to co-speech gesture generation in a dyadic setting (a main speaker and an interlocutor). 

Please refer to the [official baseline](https://github.com/genea-workshop/2023_ivi_baseline/) provided by GENEA23.

## Citation 
Please cite our paper if you find our code useful.
```
@inproceedings{chang2022ivi,
  title={The IVI Lab entry to the GENEA Challenge 2022--A Tacotron2 based method for co-speech gesture generation with locality-constraint attention mechanism},
  author={Chang, Che-Jui and Zhang, Sen and Kapadia, Mubbasir},
  booktitle={Proceedings of the 2022 International Conference on Multimodal Interaction},
  pages={784--789},
  year={2022}
}
```


