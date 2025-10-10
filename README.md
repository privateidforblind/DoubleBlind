# AlphaFree
This is the official implementation of **AlphaFree** (Recommendation Free from Users, IDs, and GNNs).

## Prerequisites
You can install the required packages with a conda environment by typing the following command in your terminal:
```bash
conda create -n AlphaFree python=3.9
conda activate AlphaFree
pip install -r requirements.txt
```
Before using the general recommendation, run the following command to install the evaluator:
```bash
pushd models/General/base
python setup.py build_ext --inplace
popd
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

To ensure blind review, LLM-based item representations can be obtained from the official GitHub repository associated with the AlphaRec (ICLR 2025) paper.
Place them into the corresponding dataset folders under:
`./data/General/{dataset_name}/item_info/`.

#### Train AlphaFree in the `Movie` dataset
```bash
./demo.sh 5 0.2 0.2 0.15 amazon-movie v3
```

#### Train AlphaFree in the `Amazon-book` dataset
```bash
./demo.sh 5 0.2 0.2 0.15 amazon-book-2014 v3
```


## Trainlog
You can find the training logs of our model in the ./log folder.
