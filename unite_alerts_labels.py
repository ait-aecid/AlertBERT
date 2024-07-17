import json
import pandas as pd
from datetime import datetime as dt
import pytz
from typing import Optional, Literal

"""For each scenario of the AIT Alert Dataset, this script reads the raw alerts from the files 
alerts/{scenario}_aminer.json and alerts/{scenario}_wazuh.json, and their lables from alerts_csv/{scenario}_alerts.csv, 
and merges the alerts and labels into a single JSON file alerts/{scenario}_alerts.json.
The resuting file contains one JSON object per line, each object representing one alert with its labels and being 
ordered by the raw timestamp of it.
"""


def get_time(j: dict, ids: Optional[Literal["a", "w"]] = None) -> float:
    """
    Get the timestamp from the given JSON object.

    Args:
        j (dict): The JSON object containing the timestamp.
        ids (Optional[Literal["a", "w"]]): The identifier for the source of the JSON object - "a" being AMiner and
            "w" being Wazuh.
            If None, the function will try to determine the source based on the keys in the JSON object.

    Returns:
        float: The timestamp as a floating-point number.

    Raises:
        KeyError: If the JSON object does not contain the required keys.

    """
    if ids is None:
        try:
            j["AMiner"]
            ids = "a"
        except:
            ids = "w"

    if ids == "a":
        return float(j["LogData"]["DetectionTimestamp"][-1])
    else:
        return float(
            dt.strptime(j["@timestamp"], "%Y-%m-%dT%H:%M:%S.%fZ")
            .replace(tzinfo=pytz.utc)
            .timestamp()
        )


scenarios = [
    "russellmitchell",
    "fox",
    "harrison",
    "santos",
    "shaw",
    "wardbeck",
    "wheeler",
    "wilson",
]

if __name__ == "__main__":
    for scenario in scenarios:
        print(f"Uniting alerts for scenario {scenario}...")

        csv_data = pd.read_csv(f"alerts_csv/{scenario}_alerts.csv")
        csv_data["raw_time"] = 0.0
        csv_data["raw_data"] = ""
        a_csv = (
            csv_data.where(csv_data.name.apply(
                lambda x: x.startswith("AMiner")))
            .dropna()
            .reset_index(drop=True)
        )
        w_csv = (
            csv_data.where(
                csv_data.name.apply(
                    lambda x: x.startswith(("Wazuh", "Suricata")))
            )
            .dropna()
            .reset_index(drop=True)
        )
        a_csv["time"] = a_csv["time"].astype("int64")
        w_csv["time"] = w_csv["time"].astype("int64")
        l = len(csv_data)
        assert l == len(a_csv) + len(w_csv)
        del csv_data
        data = {"aminer": a_csv, "wazuh": w_csv}

        for system in ["aminer", "wazuh"]:
            print(f"Labelling {system} alerts...")

            i = 0
            with open(f"alerts/{scenario}_{system}.json") as f:
                for line in f:
                    payload = json.loads(line)
                    time = get_time(payload, system[0])

                    assert data[system].at[i, "time"] == int(
                        time
                    ), f"{data[system].at[i, 'time']} != {int(time)}, {i}"
                    if system == "aminer":
                        assert (
                            data[system].at[i, "name"]
                            == payload["AnalysisComponent"]["AnalysisComponentName"]
                        )
                        assert data[system].at[i,
                                               "ip"] == payload["AMiner"]["ID"]
                    else:
                        assert (
                            data[system]
                            .at[i, "name"]
                            .endswith(payload["rule"]["description"])
                        ), f"{data[system].at[i, 'name']} != {payload['rule']['description']}, {i}"
                        assert data[system].at[i,
                                               "ip"] == payload["agent"]["ip"]

                    data[system].at[i, "raw_data"] = payload
                    data[system].at[i, "raw_time"] = time

                    i += 1
            assert i == len(data[system])

        print("Merging alerts...")

        all_data = pd.concat([a_csv, w_csv], ignore_index=True)
        assert l == len(all_data)
        all_data.sort_values(
            by="raw_time", kind="mergesort", inplace=True, ignore_index=True
        )

        all_data.to_json(f"alerts/{scenario}_alerts.json",
                         orient="records", lines=True)

        print(f"Processing {scenario} finished.")
        print()
