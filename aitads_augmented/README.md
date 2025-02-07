# AIT Alert Dataset - Augmented

Day 0 noise is usually omitted bc of new anomaly alerts.

DNS Steal is short but spread over many days :|
This is problematic bc eg they overshoot in russellmitchell.
Maybe split them up in the short bursts in which they occur?

Still to do: finish data set creation script and create data loading class.

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

| scenario | event_label | number of alerts | duration | original day | original start time |
|:---------|:------------|-----------------:|---------:|-------------:|--------------------:|
| fox | dirb | 410336 | 0:19:19 | 3 | 12:18:30 |
| fox | wpscan | 9515 | 0:00:26 | 3 | 12:17:50 |
| fox | service_scan | 38 | 0:00:17 | 3 | 12:17:26 |
| fox | escalated_sudo_command | 7 | 0:00:08 | 3 | 13:14:41 |
| fox | attacker_change_user | 10 | 0:00:01 | 3 | 13:14:31 |
| fox | webshell_cmd | 3 | 0:00:46 | 3 | 12:38:25 |
| fox | dnsteal | 6 | 2 days, 9:04:44 | 0 | 00:00:03 |
| fox | crack_passwords | 1 | 0:00:00 | 3 | 12:59:54 |
| fox | online_cracking | 2 | 0:00:00 | 3 | 12:39:06 |
| harrison | dirb | 415108 | 0:26:09 | 4 | 07:29:41 |
| harrison | wpscan | 9676 | 0:00:52 | 4 | 07:28:41 |
| harrison | service_scan | 26 | 0:00:03 | 4 | 07:16:31 |
| harrison | escalated_sudo_command | 41 | 0:00:14 | 4 | 08:36:54 |
| harrison | attacker_change_user | 17 | 0:00:01 | 4 | 08:36:38 |
| harrison | webshell_cmd | 3 | 0:00:35 | 4 | 07:56:31 |
| harrison | dnsteal | 7 | 4 days, 9:14:53 | 0 | 00:00:07 |
| harrison | crack_passwords | 1 | 0:00:00 | 4 | 07:58:16 |
| russellmitchell | dirb | 4522 | 0:00:13 | 3 | 03:57:26 |
| russellmitchell | wpscan | 6355 | 0:00:20 | 3 | 03:57:52 |
| russellmitchell | service_scan | 50 | 0:00:20 | 3 | 03:56:58 |
| russellmitchell | escalated_sudo_command | 15 | 0:00:08 | 3 | 04:37:58 |
| russellmitchell | attacker_change_user | 9 | 0:00:00 | 3 | 04:37:40 |
| russellmitchell | webshell_cmd | 3 | 0:00:34 | 3 | 03:59:14 |
| russellmitchell | dnsteal | 6 | 3 days, 13:50:30 | 0 | 00:00:09 |
| russellmitchell | crack_passwords | 2 | 0:13:30 | 3 | 04:01:07 |
| santos | dirb | 4522 | 0:00:11 | 3 | 11:22:02 |
| santos | wpscan | 6557 | 0:00:23 | 3 | 11:22:23 |
| santos | service_scan | 29 | 0:00:03 | 3 | 11:21:43 |
| santos | escalated_sudo_command | 28 | 0:00:27 | 3 | 11:58:27 |
| santos | attacker_change_user | 17 | 0:00:01 | 3 | 11:58:17 |
| santos | webshell_cmd | 4 | 0:00:31 | 3 | 11:24:14 |
| santos | dnsteal | 4 | 2 days, 7:16:09 | 0 | 00:00:09 |
| santos | crack_passwords | 2 | 0:18:45 | 3 | 11:25:43 |
| santos | online_cracking | 4 | 0:00:00 | 3 | 11:24:39 |
| shaw | dirb | 4522 | 0:00:12 | 4 | 14:39:14 |
| shaw | wpscan | 1478 | 0:00:12 | 4 | 14:38:52 |
| shaw | escalated_sudo_command | 28 | 0:00:07 | 4 | 15:21:02 |
| shaw | attacker_change_user | 17 | 0:00:00 | 4 | 15:20:51 |
| shaw | webshell_cmd | 3 | 0:00:32 | 4 | 14:39:56 |
| shaw | dnsteal | 2 | 0:00:00 | 3 | 21:08:01 |
| shaw | crack_passwords | 2 | 0:33:00 | 4 | 14:41:53 |
| shaw | dns_scan | 9 | 0:00:00 | 4 | 14:37:14 |
| wardbeck | dirb | 4522 | 0:00:13 | 4 | 12:11:49 |
| wardbeck | wpscan | 1506 | 0:00:12 | 4 | 12:11:29 |
| wardbeck | service_scan | 23 | 0:00:03 | 4 | 12:11:12 |
| wardbeck | escalated_sudo_command | 28 | 0:00:05 | 4 | 12:55:14 |
| wardbeck | attacker_change_user | 10 | 0:00:00 | 4 | 12:55:01 |
| wardbeck | webshell_cmd | 4 | 0:01:38 | 4 | 12:12:36 |
| wardbeck | dnsteal | 4 | 1 day, 22:12:23 | 0 | 00:00:05 |
| wardbeck | crack_passwords | 2 | 0:29:15 | 4 | 12:15:04 |
| wheeler | dirb | 417174 | 0:16:21 | 4 | 07:39:29 |
| wheeler | wpscan | 13291 | 0:00:30 | 4 | 07:56:07 |
| wheeler | service_scan | 42 | 0:00:04 | 4 | 07:39:15 |
| wheeler | escalated_sudo_command | 28 | 0:00:06 | 4 | 17:52:07 |
| wheeler | attacker_change_user | 9 | 0:00:00 | 4 | 17:51:52 |
| wheeler | webshell_cmd | 2 | 0:00:05 | 4 | 07:56:52 |
| wheeler | dnsteal | 7 | 3 days, 5:27:18 | 0 | 00:00:07 |
| wilson | dirb | 428239 | 0:20:01 | 4 | 10:59:44 |
| wilson | wpscan | 6401 | 0:00:23 | 4 | 11:19:56 |
| wilson | service_scan | 50 | 0:00:08 | 4 | 10:59:26 |
| wilson | escalated_sudo_command | 8 | 0:00:19 | 4 | 11:48:32 |
| wilson | attacker_change_user | 11 | 0:00:01 | 4 | 11:48:18 |
| wilson | webshell_cmd | 3 | 0:00:26 | 4 | 11:20:37 |
| wilson | dnsteal | 8 | 3 days, 10:47:12 | 0 | 00:00:04 |
| wilson | crack_passwords | 1 | 0:00:00 | 4 | 11:29:49 |
