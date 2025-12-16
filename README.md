# AlertBERT

This is the code repository for the paper 'AlertBERT: A noise-robust alert grouping framework for simultaneous attacks' (publication pending).

## Description

AlertBERT is a state-of-the-art self-supervised alert grouping method, that is based on masked-language-models to obtain alert embeddings and agglomerative clustering to work under high levels of noise and simultaneous cyber attacks.
For further details on the AlertBERT framework, we refer to our paper (publication pending).

To recreate the experiments reported in the paper, first, run the module `alertbert.train_mlm` to train a masked-language-model.
There the only parameters to adjust in `MaskedLangModelParams` passed to the `main` function are the number of attention heads and the used configuration of AIT-ADS-A.
After completing the training of the masked-language-model, run the module `alertbert.eval_grouping`to compute the corresponding ROC-curves by providing the `model_id` to the `model_config_generator` function.
To view the results, please refer to the plots of ROC-curves provided in the notebooks in the `roc_results` directory.

These results should then look something like the following ROC-plot obtained on the `simul-attacks` configuration of AIT-ADS-A.

![ROC-plot of AlertBERT on the `simul-attacks` configuration of AIT-ADS-A](./roc_results/ROC_mlm_1l_2h_16d_simul-attacks_1_60k_simul-attacks_output_emb_2_dim_excl_noise.png)

The individual modules in `alertbert` have the following purposes:

+ `alertbert.aitads` provides the dataset,

+ `alertbert.eval_grouping` implements the evaluation of alert groupings,

+ `alertbert.eval_mlm` implements the evaluation of the training of masked-language-models,

+ `alertbert.model_eval_utils` provides evaluation utilities,

+ `alertbert.models` implements the alert grouping models,

+ `alertbert.preprocessing` implements vocabularies and tokenisation of the data,

+ `alertbert.train_mlm` performs the training of masked-language-models, and

+ `alertbert.utils` provides general utilities.

## Installation

To run the experiments please follow these steps:

### Python Environment

To set up the python environment, run `pip install -r requirements.txt`.
Furthermore, it is necessary to install the [graph-tools](https://graph-tool.skewed.de) library, which is not available through pip.

### Download and Prepare Data

For our experiments we use the AIT-ADS-A dataset, which is an augmented version of the [AIT Alert Dataset](https://doi.org/10.5281/zenodo.8263181).
The files to construct this dataset are part of this repository and should be present in the `aitads_augmented/data` directory.

If this is the case, the setup is complete and the remaining steps are not necessary!

To build the dataset from source, follow these steps:

1. Download and unzip the three datasets into their respective directories listed below.  
After this step the `alerts_json` directory should contain the files `scenario_aminer.json` and `scenario_wazuh.json` for each of the eight scenarios of AIT-ADS.
    + [AIT Alert Dataset](https://doi.org/10.5281/zenodo.8263180) ➡️ `alerts_json`
    + [AIT Log Dataset V2.0](https://doi.org/10.5281/zenodo.5789063) ➡️ `aitldsv2`
    + [AIT Netflow Dataset](https://doi.org/10.5281/zenodo.6610488) ➡️ `aitnds`

2. Run `preprocess.py`.  
This script will read the information of the three datasets, use it to assign the labels to the alerts in AIT-ADS, and save the labels to the files `alerts_csv/scenario_alerts.csv` for each scenario.

At this point, for each scenario we have the following situation:  
The files `alerts_json/scenario_wazuh.json` and `alerts_json/scenario_aminer.json` contain the alert data sorted by timestamp, but separately for the two IDSs.  
And the files `alerts_csv/scenario_alerts.csv` contain the labels for the alerts, but there the alerts are ordered so that they correspond to a concatenation of `alerts_json/scenario_wazuh.json` and `alerts_json/scenario_aminer.json`.  
Thus, the next step:

3. Run `unite_alerts_labels.py`.  
This script will simplify the situation described above by combining all the alerts and their labels, sorted by timestamps, into the files `alerts_json/scenario.json` for each scenario.  
Additionally, the script will create the files `alerts_json/scenario_light.json`, which have the same contents except for the raw alert data, and can be used to speed up loading the data if the raw alerts are not required.

4. Finally, to create the files necessary for building AIT-ADS-A run `build_augment_files.py`.
For further information regarding the setup of AIT-ADS-A please refer to the README file in the `aitads_augmented` directory.

## Usage

+ Modules `alertbert/module.py` have to be run via `python -m alertbert.module`.
+ The documentation of the code can be found in the respective docstrings of functions, classes, and modules.
