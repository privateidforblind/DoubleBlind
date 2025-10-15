# AlphaFree
This is the official implementation of **AlphaFree** (Recommendation Free from Users, IDs, and GNNs).

## Prerequisites
You can install the required packages with a conda environment by typing the following command in your terminal:
```bash
conda create -n AlphaFree python=3.9
conda activate AlphaFree
#install with appropriate pytorch-cuda version depending on your GPU/driver
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```
Before using the general recommendation, run the following command to install the evaluator:
```bash
pushd models/General/base
python setup.py build_ext --inplace
popd

# If any error occurs, run the following command:
conda install -c conda-forge libstdcxx-ng=13.1.0
```


## Datasets
The statistics of datasets used in AlphaFree are summarized as follows. 
| Dataset | Movie | Book | Video | Baby | Steam | Beauty | Health |
|:--|--:|--:|--:|--:|--:|--:|--:|
| **#Users** | 26,073 | 71,306 | 94,762 | 150,777 | 334,730 | 729,576 | 796,054 |
| **#Items** | 12,464 | 40,523 | 25,612 | 36,013 | 15,068 | 207,649 | 184,346 |
| **#Inter.** | 875,906 | 2,206,865 | 814,586 | 1,241,083 | 4,216,781 | 6,624,441 | 7,176,552 |

<!--<img src="./assets/data_statistics.png" width="500px" height="200px" title="data statistics"/>-->

## Usage
### Train our model from scratch
You can train AlphaFree with the best hyperparameters for each dataset using the following commands:

### Dataset Downloads
You can download the dataset using the bash script at `./data/General/download.sh`
```bash
cd ./data/General/
chmod +x download.sh  
./download.sh
```

### Run demo
```bash
# Example of arguments:
# ./demo.sh K_c lambda_align tau_r tau_a dataset llm_model

chmod +x ./demo.sh 
# Train AlphaFree in the `Movie` dataset
./demo.sh 5 0.2 0.2 0.15 amazon_movie v3

# Train AlphaFree in the `Book` dataset
./demo.sh 5 0.2 0.1 0.15 amazon_book_2014 v3

# Train AlphaFree in the `Video` dataset
./demo.sh 10 0.05 0.01 0.2 amazon_video llama

# Train AlphaFree in the `Baby` dataset
./demo.sh 10 0.01 0.2 0.2 amazon_baby llama

# Train AlphaFree in the `Steam` dataset
./demo.sh 3 0.01 0.2 0.2 steam llama

# Train AlphaFree in the `Beauty` dataset
./demo.sh 10 0.01 0.1 0.2 amazon_beauty_personal llama

# Train AlphaFree in the `Health` dataset
./demo.sh 5 0.01 0.2 0.15 amazon_health llama
```

### Inference Demo
Inference demo using the pre-trained AlphaFree model on the movie dataset.  
**Note:** Before running the inference demo, download the Amazon Movie dataset first.
```bash
chmod +x ./demo_inference.sh 
./demo_inference.sh
```

## Result of AlphaFree

### Trainlog
You can find the training logs of our model in the ./log folder.

### Performance Table

| **Model**       | **Movie**  |**Book**   | **Video**  | **Baby**   | **Steam**  | **Beauty** | **Health** |
|-------------|--------|--------|--------|--------|--------|--------|--------|
| MF-BPR      | 0.0580 | 0.0436 | 0.0177 | 0.0150 | 0.1610 | 0.0063 | 0.0091 |
| FISM-BPR    | 0.0861 | 0.0623 | 0.0392 | 0.0150 | 0.1801 | 0.0079 | 0.0104 |
| LightGCN    | 0.0860 | 0.0712 | 0.0732 | 0.0359 | 0.2013 | 0.0201 | 0.0250 |
| XSimGCL     | 0.0967 | 0.0818 | 0.0897 | 0.0390 | 0.2245 | 0.0253 | 0.0299 |
| RLMRec      | 0.1046 | 0.0905 | o.o.t. | o.o.t. | o.o.t. | o.o.t. | o.o.t. |
| AlphaRec    | 0.1219 | 0.0991 | 0.1088 | 0.0391 | 0.2360 | o.o.m. | o.o.m. |
| **AlphaFree** | **0.1267** | **0.1014** | **0.1111** | **0.0412** | **0.2402** | **0.0361** | **0.0325** |

<small>o.o.t.: Out of Time</small>  
<small>o.o.m.: Out of Memory</small>
