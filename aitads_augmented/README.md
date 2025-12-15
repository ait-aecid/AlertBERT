# AIT Alert Dataset - Augmented

A flexible and easily configured augmentation method for the [AIT Alert Dataset](https://doi.org/10.5281/zenodo.8263181).  

So far AIT-ADS-A does not contain the raw alert data of AIT-ADS because there we would have to make sure to adapt all timestamps and IP addresses correctly, and this is not yet implemented.

## Method

The reason why AIT-ADS-A was created is that the original AIT-ADS only features one attack at a time and a constant noise level.
In this regime the simple time-delta method (by [Landauer et al.](https://doi.org/10.1145/3510581)) is quite effective at the alert grouping problem, hence data with more noise and/or simultaneous attacks were required to create and evaluate more advanced alert grouping methods.  
In order to save the effort of creating a new version of AIT-ADS, the idea was born to instead augment it by mixing alerts of different scenarios or days and thereby create a dataset with a desired amount of noise and attacks which are more difficult to group.

This augmentation happens in the following way:
In order to be able to recombine the existing alerts in the most flexible way while retaining all meaningful relations among them, each scenario of AIT-ADS is split into the subsequences corresponding to the different event labels (except for the case of the "dnsteal" event label, the procedure for which is detailed below), that is the full alert sequence is separated in sequences corresponding to the false alerts on the one hand and the alerts which are triggered by each step of the attack chain on the other hand.
The different alert sequences for false positives and attacks can then be freely recombined, through specification of a configuration file, to create new alert sequences which have the same underlying syntactic and semantic structure as the original AIT-ADS.
The elementary unit of an AIT-ADS-A configuration is a day of alerts which can consist of the false alerts of several days in AIT-ADS and multiple attacks defined to take place at a certain time during this day.
All the specified alerts of this day will then be assigned timestamps which make them appear to have happened throughout this day and the different sequences are merge-sorted by these timestamps into a single alert sequence representing this day in AIT-ADS-A.
Similarly to AIT-ADS it is also possible in AIT-ADS-A to combine multiple such days into a continuous scenario, which is sensible e.g. if multiple days use noise or attacks from the same scenarios in AIT-ADS.
For precise information on how to define configurations of AIT-ADS-A, please refer to the following sections.

When designing AIT-ADS-A several design decision had to be made to ensure that the resulting data structure is both performant and flexible while only allowing for sensible configurations of the data to be made.
Thus, here is a short summary of the most important of these decisions and why they were made like this:
+ Any configuration of AIT-ADS-A is defined beforehand and its structure is hardcoded there.
This is done for two reasons:
    1) Dynamic creation of new augmentations (e.g. at every epoch) is too slow, so the augmented dataset is assembled once when it is loaded and then it is only sampled from afterwards.
    2) Using different randomised augmentations of the data during training would simulate an infinite supply of data.
    This would not be a problem if our alert grouping method would be supervised because then it is allowed to use the true labels of the data for training data augmentation, but as our method is unsupervised we must only use the labels to create a new finite supply of data on which we then train in an unsupervised way.

+ The noise level of a configuration is a positive integer and it determines how many noise sequences are overlaid on each day.
In order to simulate alert data from a single fixed-size computer network, this number should be the same for every day within a configuration of AIT-ADS-A.
Continuous noise levels are currently impossible to implement as it is impossible to subsample a sequence of false alerts while ensuring that its structure remains consistent with a real sequence of false alerts.
+ To further enforce consistency with real alert sequences, different sequences of false alerts are always merged in a way so that the alerts match in the time-of-day at which they occurred. This is done because the frequency of false alerts varies considerably over the course of a day and mixing of alerts from different daytimes would create structure not observed in real data.
This principle is the core reason why days were chosen to be the elementary unit of an AIT-ADS-A configuration.
+ Unlike other event labels, the alerts assigned to the "dnsteal" attack receive a special preprocessing where they are split up in the three sequences "dnsteal_start", "dnsteal_active", and "dnsteal_end".
This is done because, in contrast to the other labels, the dnsteal sequence is not closely localised in time but in most cases extends over several days and is in turn composed of common repeating patterns.
These patterns themselves, however, all have the properties of independent parts of an attack chain (closely localised in time and forming a distinctive pattern), and thus are encapsulated in the three new sub-labels "dnsteal_start", "dnsteal_active", and "dnsteal_end" which can then be independently placed in configurations of AIT-ADS-A.

## Hierarchical Event Labels

As AIT-ADS-A, in contrast to AIT-ADS where every attack only occurs once in a scenario, allows for the placement of multiple instances of the same attack alerts within a scenario or even within a day where they should represent different instances of the same (type of) attack, it is also necessary to be able to distinguish these different instances in the evaluation of alert grouping models.  
For this reason, AIT-ADS-A also facilitates an augmentation of the event labels of AIT-ADS by extending them in a hierarchical manner.

The hierarchical event labels consist of the following 3 levels:
1. The original event labels from AIT-ADS.
1. The attack stage.
1. A unique attack identifier.

The meaning of the first level is clear, it just says what general type of attack the alert belongs to.
The second level was introduced (and is currently only used by) the dnsteal alerts which, as outlined above, can be separated in three different stages which generally do not occur close in time to each other.
The third level is a unique identifier which is assigned to every attack during the construction of the data and serves to differentiate between different instances of the same attack alerts.

With these labels the goal of the alert grouping problem is to group together all alerts which have the same hierarchical event label on all three levels.
To measure the performance of an alert grouping model on a high-level-label of the label hierarchy one calculates the respective macro score by taking the average of all the scores over the lower-level-labels contained in it.

## Data preparation
__This step is only necessary if the files `<scenario>_<attack|noise_number>.json` are not present in the `data` directory!__  

To prepare the AIT-ADS for creating augmented versions of it run `../build_augment_files.py`.  
This script will separate the data into several files containing the noise of each AIT-ADS scenario split into days, and the alerts of individual attacks for each scenario.  
Lists describing the contents of each noise/attack file can be found at the end of this document.

## Using AIT-ADS-A

To load AIT-ADS-A into a PyTorch dataset use the `AITAlertDataset` class defined in `../alertbert/aitads.py` instantiated with the keyword arguments `flavour="augmented"` and `config` the name to a config file.  
The config name `"original"` recreates the original AIT-ADS with the augmented dataset class (except for the small difference that dnsteal alerts of the first days are not discarded but moved to the next days).  

To create a new configuration of AIT-ADS-A it is sufficient to define a config file as described below and, to evaluate models on this configuration, build a ground truth vocabulary of it by running `python -m alertbert.model_eval_utils <config-name>`.  
If you create a new config file, please add a short description of it to the list below.

### Config files

Config files are json files with the following contents:
```json
{
    "name":       str,                # name of the configuration
    "start_time": str,                # isoformat datetime string indicating the start of each scenario, e.g. "2025-02-10T00:00:00+01:00"
    "train":      array[array[day]],  # the recipes for the splits of the configuration
    "val":        array[array[day]],
    "test":       array[array[day]]
}
```

Each split of the dataset consists of multiple sequences of days (like the scenarios of the original AIT-ADS).  
Like in the original AIT-ADS the days within one scenario will be concatenated to form a longer alert sequence.  

The configuration of a day has the following format:
```json
day: {
    "noise":   array[str],             # list of names of the noise files to combine in this day
    "attacks": array[array[str,str]],  # array of attacks to use in this day with elements being arrays of filename and starting time of the attack
}
```

As an example one can consider `configs/original.json` which defines the original AIT-ADS data in terms of AIT-ADS-A.  

__Note 1:__ Day 0 noise is usually omitted because it contains many new anomaly alerts and thus is has a different distribution of alerts than noise from the remaining days.  
__Note 2:__ It is fine to mix attacks and noise of different scenarios as long as one does not use the IP address of the alerts as feature for the model!
Using the "host" feature is fine though as it is unified across scenarios.

### Implemented Configurations

#### original
Recreates the original AIT-ADS with the augmented dataset class (except for the small difference that dnsteal alerts of the first days are not discarded but moved to the next days).

#### simul-attacks
The noise, number of scenarios and days is the same as in "original", but the attacks are rearranged so that there are collisions in time of scan/scan and scan/exploit pairs.

#### more-noise-1/2/6/11
The purpose of this family of configurations is to enable the study of alert grouping under increased densities of false alerts in the data.
In order to do this systematically, the following design decisions have been made:

+ In each configuration more-noise-x every day contains x+1 days of noise of the original configuration, that is the configuration has noise level x+1.

+ In order to keep the noise balanced and realistic despite the augmentation, it was tried whenever possible to 1) keep the noise alerts in their original order within scenarios, 2) let every day within a scenario have a similar distribution of noise, 3) let every noise file occur equally often in the configuration, and 4) not have the same noise occur on different days. With increasing noise levels, however, it is not possible to completely satisfy these constraints anymore.
Easing this burden was part of the reason for:
+ The total amount of noise alerts in each configuration is capped at around 2.1 million.
While the original configuration contains about 700k noise alerts, and more-noise-1 accordingly 1.4m, in the remaining configurations it was decided to cap the total number of noise alerts because otherwise, on the one hand, the overall signal/noise ratio in the data would become very low and, on the other hand, the noise would become repetitive and thus unrealistic.
To implement this limit the number of days in the scenarios of more-noise-6 and more-noise-11 was reduced.
An overview of the situation is provided in the table below.
+ In order to maintain comparability between the different configurations, only the number of days per scenario was adapted, and the number of scenarios and the attacks belonging to each scenario were left the same.
There were, however, two modifications made to the attacks:
+ As the purpose of these configurations is to examine the attacks under high levels of noise and the attacks in the russellmitchell scenario occur in the early morning hours where only little noise is present, they were postponed by precisely 6 hours to take place during working hours.
+ Because, due to the cap on the total amount of noise, more-noise-11 contains only 1 day per scenario anymore, there it was necessary to move all attacks of each scenario inside this day. This only affected dnsteal alerts and their timestamps were sometimes slightly adjusted to avoid collisions and maintain their temporal order.
In order to keep the different configurations comparable, the same changes were also applied to the other configurations, that is in all more-noise-x configurations, all attacks take place on the same day.

| configuration | noise level | number of noise alerts | days per scenario
|:-|-:|-:|-:|
| original/simul-attacks | 1 | 712.304 | [5,4,4,5,4,3,4,3]
| more-noise-1 | 2 | 1.424.608 | [5,4,4,5,4,3,4,3]
| more-noise-2 | 3 | 2.136.912 | [5,4,4,5,4,3,4,3]
| more-noise-6 | 7 | 2.205.741 | [2,2,2,2,2,1,2,1]
| more-noise-11 | 12 | 2.083.900 | [1,1,1,1,1,1,1,1]


## Noise files

| scenario | day | number of alerts |
|:---------|----:|-----------------:|
| fox | 0 | 10809 |
| fox | 1 | 11709 |
| fox | 2 | 9741 |
| fox | 3 | 9698 |
| fox | 4 | 11229 |
| harrison | 0 | 39634 |
| harrison | 1 | 34904 |
| harrison | 2 | 34331 |
| harrison | 3 | 36119 |
| harrison | 4 | 24081 |
| russellmitchell | 0 | 8582 |
| russellmitchell | 1 | 7451 |
| russellmitchell | 2 | 9691 |
| russellmitchell | 3 | 8858 |
| santos | 0 | 39915 |
| santos | 1 | 40661 |
| santos | 2 | 22918 |
| santos | 3 | 16118 |
| shaw | 0 | 8663 |
| shaw | 1 | 11269 |
| shaw | 2 | 9554 |
| shaw | 3 | 9454 |
| shaw | 4 | 9854 |
| shaw | 5 | 15927 |
| wardbeck | 0 | 19102 |
| wardbeck | 1 | 14566 |
| wardbeck | 2 | 21928 |
| wardbeck | 3 | 15328 |
| wardbeck | 4 | 14234 |
| wheeler | 0 | 35000 |
| wheeler | 1 | 37957 |
| wheeler | 2 | 37262 |
| wheeler | 3 | 38382 |
| wheeler | 4 | 37007 |
| wilson | 0 | 37452 |
| wilson | 1 | 36133 |
| wilson | 2 | 34968 |
| wilson | 3 | 34617 |
| wilson | 4 | 26091 |
| wilson | 5 | 30264 |

## Attack files

| scenario | event_label | number of alerts | duration | original day(s) | original start time(s) |
|:---------|:------------|-----------------:|---------:|-------------:|--------------------:|
| fox | dirb | 410336 | 0:19:19 | 3 | 12:18:30 |
| fox | wpscan | 9515 | 0:00:26 | 3 | 12:17:50 |
| fox | service_scan | 38 | 0:00:17 | 3 | 12:17:26 |
| fox | escalated_sudo_command | 7 | 0:00:08 | 3 | 13:14:41 |
| fox | attacker_change_user | 10 | 0:00:01 | 3 | 13:14:31 |
| fox | webshell_cmd | 3 | 0:00:46 | 3 | 12:38:25 |
| fox | dnsteal_start | 2 | 0:00:00 | 0 | 00:00:03 |
| fox | dnsteal_active | 1 | 0:00:00 | 0,0 | 08:54:01,09:49:07 |
| fox | dnsteal_end | 2 | 0:00:00 | 2 | 09:04:47 |
| fox | crack_passwords | 1 | 0:00:00 | 3 | 12:59:54 |
| fox | online_cracking | 2 | 0:00:00 | 3 | 12:39:06 |
| harrison | dirb | 415108 | 0:26:09 | 4 | 07:29:41 |
| harrison | wpscan | 9676 | 0:00:52 | 4 | 07:28:41 |
| harrison | service_scan | 26 | 0:00:03 | 4 | 07:16:31 |
| harrison | escalated_sudo_command | 41 | 0:00:14 | 4 | 08:36:54 |
| harrison | attacker_change_user | 17 | 0:00:01 | 4 | 08:36:38 |
| harrison | webshell_cmd | 3 | 0:00:35 | 4 | 07:56:31 |
| harrison | dnsteal_start | 2 | 0:00:00 | 0 | 00:00:07 |
| harrison | dnsteal_active | 1 | 0:00:00 | 0,0,1 | 08:00:05,18:00:03,11:59:55 |
| harrison | dnsteal_end | 2 | 0:00:00 | 4 | 09:15:00 |
| harrison | crack_passwords | 1 | 0:00:00 | 4 | 07:58:16 |
| russellmitchell | dirb | 4522 | 0:00:13 | 3 | 03:57:26 |
| russellmitchell | wpscan | 6355 | 0:00:20 | 3 | 03:57:52 |
| russellmitchell | service_scan | 50 | 0:00:20 | 3 | 03:56:58 |
| russellmitchell | escalated_sudo_command | 15 | 0:00:08 | 3 | 04:37:58 |
| russellmitchell | attacker_change_user | 9 | 0:00:00 | 3 | 04:37:40 |
| russellmitchell | webshell_cmd | 3 | 0:00:34 | 3 | 03:59:14 |
| russellmitchell | dnsteal_start | 2 | 0:00:00 | 0 | 00:00:09 |
| russellmitchell | dnsteal_active | 1 | 0:00:00 | 0,1 | 06:56:57,07:00:05 |
| russellmitchell | dnsteal_end | 2 | 0:00:00 | 3 | 13:50:39 |
| russellmitchell | crack_passwords | 2 | 0:13:30 | 3 | 04:01:07 |
| santos | dirb | 4522 | 0:00:11 | 3 | 11:22:02 |
| santos | wpscan | 6557 | 0:00:23 | 3 | 11:22:23 |
| santos | service_scan | 29 | 0:00:03 | 3 | 11:21:43 |
| santos | escalated_sudo_command | 28 | 0:00:27 | 3 | 11:58:27 |
| santos | attacker_change_user | 17 | 0:00:01 | 3 | 11:58:17 |
| santos | webshell_cmd | 4 | 0:00:31 | 3 | 11:24:14 |
| santos | dnsteal_start | 2 | 0:00:00 | 0 | 00:00:09 |
| santos | dnsteal_end | 2 | 0:00:00 | 2 | 07:16:18 |
| santos | crack_passwords | 2 | 0:18:45 | 3 | 11:25:43 |
| santos | online_cracking | 4 | 0:00:00 | 3 | 11:24:39 |
| shaw | dirb | 4522 | 0:00:12 | 4 | 14:39:14 |
| shaw | wpscan | 1478 | 0:00:12 | 4 | 14:38:52 |
| shaw | escalated_sudo_command | 28 | 0:00:07 | 4 | 15:21:02 |
| shaw | attacker_change_user | 17 | 0:00:00 | 4 | 15:20:51 |
| shaw | webshell_cmd | 3 | 0:00:32 | 4 | 14:39:56 |
| shaw | dnsteal_end | 2 | 0:00:00 | 3 | 21:08:01 |
| shaw | crack_passwords | 2 | 0:33:00 | 4 | 14:41:53 |
| shaw | dns_scan | 9 | 0:00:00 | 4 | 14:37:14 |
| wardbeck | dirb | 4522 | 0:00:13 | 4 | 12:11:49 |
| wardbeck | wpscan | 1506 | 0:00:12 | 4 | 12:11:29 |
| wardbeck | service_scan | 23 | 0:00:03 | 4 | 12:11:12 |
| wardbeck | escalated_sudo_command | 28 | 0:00:05 | 4 | 12:55:14 |
| wardbeck | attacker_change_user | 10 | 0:00:00 | 4 | 12:55:01 |
| wardbeck | webshell_cmd | 4 | 0:01:38 | 4 | 12:12:36 |
| wardbeck | dnsteal_start | 2 | 0:00:00 | 0 | 00:00:05 |
| wardbeck | dnsteal_end | 2 | 0:00:00 | 1 | 22:12:28 |
| wardbeck | crack_passwords | 2 | 0:29:15 | 4 | 12:15:04 |
| wheeler | dirb | 417174 | 0:16:21 | 4 | 07:39:29 |
| wheeler | wpscan | 13291 | 0:00:30 | 4 | 07:56:07 |
| wheeler | service_scan | 42 | 0:00:04 | 4 | 07:39:15 |
| wheeler | escalated_sudo_command | 28 | 0:00:06 | 4 | 17:52:07 |
| wheeler | attacker_change_user | 9 | 0:00:00 | 4 | 17:51:52 |
| wheeler | webshell_cmd | 2 | 0:00:05 | 4 | 07:56:52 |
| wheeler | dnsteal_start | 2 | 0:00:00 | 0 | 00:00:07 |
| wheeler | dnsteal_active | 1 | 0:00:00 | 0,1,1 | 05:53:54,05:42:02,06:00:01 |
| wheeler | dnsteal_end | 2 | 0:00:00 | 3 | 05:27:25 |
| wilson | dirb | 428239 | 0:20:01 | 4 | 10:59:44 |
| wilson | wpscan | 6401 | 0:00:23 | 4 | 11:19:56 |
| wilson | service_scan | 50 | 0:00:08 | 4 | 10:59:26 |
| wilson | escalated_sudo_command | 8 | 0:00:19 | 4 | 11:48:32 |
| wilson | attacker_change_user | 11 | 0:00:01 | 4 | 11:48:18 |
| wilson | webshell_cmd | 3 | 0:00:26 | 4 | 11:20:37 |
| wilson | dnsteal_start | 2 | 0:00:00 | 0 | 00:00:04 |
| wilson | dnsteal_active | 1 | 0:00:00 | 0,1,1,1 | 02:59:50,05:46:37,05:59:56,12:00:00 |
| wilson | dnsteal_end | 2 | 0:00:00 | 3 | 10:47:16 |
| wilson | crack_passwords | 1 | 0:00:00 | 4 | 11:29:49 |
