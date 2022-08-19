# Tacotron2-SpeechGesture
This is the official repository for our publication **"The IVI Lab entry to the GENEA Challenge 2022 – A Tacotron2 Based Method for Co-Speech Gesture Generation With Locality-Constraint Attention Mechanism."**

This repository provides the code for model training and prediction using the data from GENEA Challenge 2022.

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
Get the dataset (v1) from GENEA Challenge 2022 and unzip all files. Put the "dataset_v1" under this repository. Create a "tst" folder under the "dataset_v1" directory and put all test data inside.

Your file hierarchy should look like this: 
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

- Remove line number 6 in val_metadata.csv (val_2022_v1_005) because the audio is completely silent.

### FastText Word Embeddings
Download and unzip the [word embedding](https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip) from FastText. Put the file "crawl-300d-2M.vec" under the project directory.

### Data Preprocessing
Preprocess the data and save them as h5 files. (This might take a while) 
```
python process_data.py -d path_to_your_dataset_v1 
```

By default, the three h5 files (trn_v1.h5, val_v1.h5, and tst_v1.h5) should be generated under the project directory.

Alternatively, you can simply [download](https://drive.google.com/drive/folders/1KW_cN_1pL9KgQxQyyYUATAlgA0CcX9LZ?usp=sharing) our processed data.

### Create Motion Processing Pipelines
We will use the following command to generate pipelines (*.sav) for converting our motion representations to euler rotations.
```
python create_pipeline.py
```


## Test
We predict all gestures for both tracks (full and upper body) given the processed test data (tst_v1.h5).

Download and unzip the [checkpoints](https://drive.google.com/drive/folders/1RNpXTMVmx36PmCESQlVFo-ez9C7oDN5R?usp=sharing). Put the "fullbody" and "upperbody" folders under Tacotron2/ 

Navigate to Tacotron2/ folder.
```
cd Tacotron2
```

Predicting fullbody gestures: 
```
python generate_all_gestures.py -ch fullbody/ckpt/checkpoint_21000.pt -t full
```

Predicting upperbody gestures: 
```
python generate_all_gestures.py -ch upperbody/ckpt/checkpoint_22000.pt -t upper
```

The bvh files should be generated under "Tacotron2/outputs/" folder. By defaut, the cuda device "0" is used. If you prefer to use a different cuda device for inference, please edit line 23 in the Tacotron2/common/hparams.py

Note: 
To visualize the bvh files, please reach out to the [repository](https://github.com/TeoNikolov/genea_visualizer) provided by GENEA Challenge 2022. It provides the visualization of the motions in Blender.

## Train 
TBD

## Citation 
Please cite our paper if you use our code.
```

```


