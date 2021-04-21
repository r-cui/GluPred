# Personalised Short-Term Glucose Prediction via Recurrent Self-Attention Network
This is the official repository of the paper "Personalised Short-Term Glucose Prediction via Recurrent Self-Attention Network".
## Dependencies
This repository has been tested on the following configuration of dependencies.
* Python 3.8.8
* torch 1.8.0
* numpy 1.19, pandas 1.2

## Proprocessing
First, acquire the raw OhioT1DM data from http://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html.

The raw data should have the following structure
```
ohiot1dm
|-- OhioT1DM-training
|-- OhioT1DM-testing
|-- OhioT1DM-2-training
|-- OhioT1DM-2-testing
```
Then run the following command to preprocess the data using our scripts.
```
python3 ./preprocess/linker.py --data_folder_path path/to/ohiot1dm --extract_folder_path ./data
```

## Demo
For a fast demo of our results, run
```
python3 eval.py --ckpts_dir ./pretrained/set1_30
```

## Train
An example of replicating the setting 1 on subject 540 for 30 minutes prediciton horizon. 
```
python3 train.py --patient 540 --missing_len 6 --single_pred --transfer_learning
```
An example of replicating the setting 2 on subject 540 for 30 minutes prediciton horizon. 
```
python3 train.py --patient 540 --missing_len 6 --transfer_learning
```
An example of replicating the ablation study 1 on subject 540 for 30 minutes prediciton horizon. 
```
python3 train.py --patient 540 --missing_len 6 --single_pred --unimodal --transfer_learning
```
An example of replicating the ablation study 2 on subject 540 for 30 minutes prediciton horizon. 
```
python3 train.py --patient 540 --missing_len 6 --single_pred
python3 train.py --patient 540 --missing_len 6
```

## Citing
```
@inproceedings{cui2021personalised,
      author       = {Cui, Ran and Hettiarachchi, Chirath and Nolan, Christopher J and Daskalaki, Elena and Suominen, Hanna},
      title        = {Personalised Short-Term Glucose Prediction via Recurrent Self-Attention Network},
      booktitle    = {2021 IEEE 34th International Symposium on Computer-Based Medical Systems (CBMS)},
      year         = {2021},
      organization = {IEEE}
}
```

## License
MIT
