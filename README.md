# alert-grouping-interns

## data sets

AIT Alert Dataset - <https://zenodo.org/records/8263181>

Github - <https://github.com/ait-aecid/alert-data-set>

AIT Log Dataset V2.0  - <https://zenodo.org/records/5789064>

AIT Netflow Dataset - <https://zenodo.org/records/6610489>

### papers

Dealing with Security Alert Flooding: Using Machine Learning for Domain-independent Alert Aggregation - <https://dl.acm.org/doi/pdf/10.1145/3510581>

Maintainable Log Datasets for Evaluation of Intrusion Detection Systems - <https://ieeexplore.ieee.org/abstract/document/9866880>

Introducing a New Alert Data Set for Multi-Step Attack Analysis - <https://arxiv.org/abs/2308.12627>

## Description (tbd)

## Installation (tbd)

### set up python environment

### Download and prepare data

If you have access to the zipped data package `alerts_json/aitads_unified.zip` then you just need to unzip it within the `alerts_json` directory and you are good to go.  
Alternatively, you can also build the dataset from source by the following recipe.

To prepare the AIT-ADS (including its labels) for training and evaluation from source follow these steps:

1. Download and unzip the three datasets into their respective dicrectories listed below.  
After this step the `alerts_json` directory should contain the files `scenario_aminer.json` and `scenario_wazuh.json` for each of the eight scenarios of AIT-ADS.
    + AIT Alert Dataset ➡️ `alerts_json`
    + AIT Log Dataset V2.0 ➡️ `aitldsv2`
    + AIT Netflow Dataset ➡️ `aitnds`

2. Run `preprocess.py`.  
This script will read the information of the three datasets, use it to assign the labels to the alerts in AIT-ADS, and save the labels to the files `alerts_csv/scenario_alerts.csv` for each scenario.

At this point, for each scenario we have the following situation:  
The files `alerts_json/scenario_wazuh.json` and `alerts_json/scenario_aminer.json` contain the alert data sorted by timestamp, but separately for the two IDSs.  
And the files `alerts_csv/scenario_alerts.csv` contain the labels for the alerts, but there the alerts are ordered so that they correspond to a concatenation of `alerts_json/scenario_wazuh.json` and `alerts_json/scenario_aminer.json`.  
Thus, the last step:

3. Run `unite_alerts_labels.py`.  
This script will simplify the situation described above by combining all the alerts and their labels, sorted by timestamps, into the files `alerts_json/scenario.json`for each scenario.  
Additionally, the script will create the files `alerts_json/scenario_light.json`, which have the same contents except for the raw alert data, and can be used to speed up loading the data if the raw alerts are not required.

Done.

### AITADS-Augmented

For information regarding the setup of AITADS-A please refer to the README file in the `aitads_augmented` directory.

## Usage (tbd)

+ [put here short descriptions of modules]
+ modules `/deep_learning/module.py` have to be run via `python -m deep_learning.module`.
+ all the important info can be found in the docstrings of modules.
