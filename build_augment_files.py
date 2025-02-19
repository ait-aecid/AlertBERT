import datetime as dt
import json
import os

import numpy as np
import pandas as pd

"""For each scenario of the AIT Alert Dataset, this script reads the raw alerts from
the files `alerts_json/{scenario}_light.json` and splits them into noise and attacks.
The noise alerts are separated into days and saved as individual JSON files in the
folder `aitads_augmented/data/` with the names `{scenario}-{day}.json`.
The attack alerts are saved as individual JSON files in the folder `aitads_augmented/`
with the names `{scenario}-{attack}.json`.
Additionally, the start times of the attacks are saved in a dictionary and printed to
the console for reconstruction of the original dataset and the constant part of the
hierarchical event labels is added to the attack alerts.
"""

scenarios = [
    "fox",
    "harrison",
    "russellmitchell",
    "santos",
    "shaw",
    "wardbeck",
    "wheeler",
    "wilson",
]

attack_event_labels = [
    "dirb",
    "wpscan",
    "service_scan",
    "escalated_sudo_command",
    "attacker_change_user",
    "webshell_cmd",
    "dnsteal_start",
    "dnsteal_active",
    "dnsteal_end",
    "crack_passwords",
    "dns_scan",
    "online_cracking",
]

dnsteal_stage_info = {
    "dnsteal_start": ("A-Dns-Val1", ".0."),
    "dnsteal_active": ("A-Dns-Frq", ".1."),
    "dnsteal_end": ("A-Aud-Com4", ".2."),
}

if __name__ == "__main__":
    # check if aitads_augmented[/data] directories exists
    if not os.path.exists("aitads_augmented/data"):
        os.makedirs("aitads_augmented/data")

    for scenario in scenarios:
        print(f"Processing {scenario}...")

        with open(f"alerts_json/{scenario}_light.json") as f:
            data = np.array(json.load(f))

        length = len(data)
        keys = list(data[0].keys())
        data = {k: np.array([d[k] for d in data]) for k in keys}

        print(f"Number of alerts: {length}")

        # separate noise
        noise_mask = data["event_label"] == "-"
        noise = {k: data[k][noise_mask] for k in keys}
        noise_length = len(noise["event_label"])
        print(f"Number of noise alerts: {noise_length}")

        # split noise into days and subtract initial times
        day0 = dt.date.fromtimestamp(noise["time"][0])
        time0 = dt.datetime(day0.year, day0.month, day0.day, 0, 0, 0).timestamp()

        noise["time"] = (noise["time"] - time0).astype(int)
        noise["raw_time"] = noise["raw_time"] - time0

        days = noise["time"] // 86400

        noise["time"] = noise["time"] % 86400
        noise["raw_time"] = noise["raw_time"] % 86400

        noise = [{k: noise[k][days == i] for k in keys} for i in range(days[-1] + 1)]

        # print number of alerts
        checksum = 0
        for i, n in enumerate(noise):
            checksum += len(n["event_label"])
            print(f"Number of noise alerts on day {i}: {len(n['event_label'])}")
            # print(f"| {scenario} | {i} | {len(n['event_label'])} |")  # for table in README
        assert checksum == noise_length, (
            f"We lost some noise alerts! {checksum} != {noise_length}"
        )

        # save noise
        noise = [pd.DataFrame(day) for day in noise]
        for df in noise:
            df.insert(7, "hierarchical_event_label", "-")
        noise = [df.to_dict(orient="records") for df in noise]

        for i, day in enumerate(noise):
            with open(f"aitads_augmented/data/{scenario}-{i}.json", "w") as f:
                json.dump(
                    day,
                    f,
                    indent=0,
                    separators=(",", ":"),
                )

        # split attacks
        checksum = 0
        attack_time_dict = {i: [] for i in range(days[-1] + 1)}
        for attack in attack_event_labels:
            if attack.startswith("dnsteal"):
                short, stage = dnsteal_stage_info[attack]
                attack_mask = (data["event_label"] == "dnsteal") & (
                    data["short"] == short
                )
                hierarchical_event_label = "dnsteal" + stage
            else:
                attack_mask = data["event_label"] == attack
                hierarchical_event_label = attack + ".0."

            if not np.any(attack_mask):
                continue

            attack_data = {k: data[k][attack_mask] for k in keys}

            # print number of alerts, start time and duration
            attack_length = len(attack_data["event_label"])
            checksum += attack_length
            day = (dt.date.fromtimestamp(attack_data["time"][0]) - day0).days
            start_time = dt.datetime.fromtimestamp(attack_data["time"][0])
            duration = dt.timedelta(
                seconds=float(attack_data["time"][-1] - attack_data["time"][0])
            )
            print(f"Number of {attack:<22} alerts: {attack_length}")
            # for table in README, not valid for dnsteal_active!!! (see if-block below)
            # print(f"| {scenario} | {attack} | {attack_length} | {duration} | {day} | {start_time.time().strftime("%H:%M:%S")} |") if attack != "dnsteal_active" else None

            # remove initial times
            attack_data["raw_time"] = attack_data["raw_time"] - attack_data["time"][0]
            attack_data["time"] = attack_data["time"] - attack_data["time"][0]

            # save attacks
            file_name = f"{scenario}-{attack}"

            if attack == "dnsteal_active":
                # special treatment of dnsteal_active because it is just a repetition of the same event
                # thus it is saved only once and its different occurrences are treated as repetitions of the same attack
                times = [
                    start_time + dt.timedelta(seconds=float(t))
                    for t in attack_data["time"]
                ]
                attack_data = {k: attack_data[k][:1] for k in keys}
                day = [str((d.date() - day0).days) for d in times]
                daytime = [d.time().strftime("%H:%M:%S") for d in times]
                for d, t in zip(day, daytime):
                    attack_time_dict[int(d)].append([file_name, t])
                # print(f"| {scenario} | {"dnsteal_active"} | {1} | {"0:00:00"} | {",".join(day)} | {",".join(daytime)} |")
            else:
                attack_time_dict[day].append(
                    [file_name, start_time.time().strftime("%H:%M:%S")]
                )

            attack_data = pd.DataFrame(attack_data)
            attack_data.insert(7, "hierarchical_event_label", hierarchical_event_label)
            attack_data = attack_data.to_dict(orient="records")

            with open(f"aitads_augmented/data/{file_name}.json", "w") as f:
                json.dump(
                    attack_data,
                    f,
                    indent=0,
                    separators=(",", ":"),
                )

        # print(*(f"{k}: {json.dumps(sorted(a, key=lambda x: x[1]))}" for k,a in attack_time_dict.items()), sep="\n")  # for config file of original AITADS
        assert checksum + noise_length == length, (
            f"We lost some alerts! {checksum + noise_length} != {length}"
        )
        print()
