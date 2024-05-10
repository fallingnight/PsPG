"""
    this script include two func as the jsonl process example
    one for CheXpert and the other for PadChest

"""

import os
import shutil
import pandas as pd
import json


def extract_chexpert_data(
    csv_filepath,
    jsonl_filepath,
    class_filepath="labels.json",
    filepath_prefix="CheXpert-v1.0/train",
    type="train",
    is_output_classname=True,
):
    """
    This function is an example that used to process the metadata csv file of CheXpert dataset into our given jsonl format.

    For other datasets, you can refer to this.

    """
    df = pd.read_csv(csv_filepath)
    count = 0

    if type == "train" or type == "val":
        df = df[df["Frontal/Lateral"] == "Frontal"]
    else:
        df = df[df["Path"].str.contains("frontal")]
    if is_output_classname:
        categories = df.columns[5:].tolist()
        print(categories)
        categories_data = {"labels_name": categories}

    jsonl_data = []
    for index, row in df.iterrows():
        if type == "train" or type == "val":
            patient_id = row["Path"].split("/")[2]
            filepath = row["Path"].replace(filepath_prefix, type)
            labels = [
                row[column] if pd.notna(row[column]) else 0.0
                for column in df.columns[5:]
            ]
        else:
            patient_id = row["Path"].split("/")[1]
            filepath = row["Path"].replace("test", type)
            labels = [
                row[column] if pd.notna(row[column]) else 0.0
                for column in df.columns[1:]
            ]

        data = {
            "id": patient_id,
            "filepath": filepath,
            "text": "",
            "labels": labels,
        }

        jsonl_data.append(data)

    with open(jsonl_filepath, "w", encoding="utf-8") as jsonl_file:
        for data in jsonl_data:
            jsonl_file.write(json.dumps(data) + "\n")
    if is_output_classname:
        with open(class_filepath, "w", encoding="utf-8") as categories_file:
            json.dump(categories_data, categories_file)

    print("Conversion completed.")


def extract_padchest_data(
    input_file,
    source_folder="padchest",
    destination_folder="padchest",
    is_output_classname=True,
):
    """
    This function is an example that used to process the metadata file of PadChest dataset and select classes positive samples > 50.

    For other datasets, you can refer to this.
    """
    first_file = select_csv(input_file, source_folder, destination_folder)
    second_file = selected_labels(first_file)
    df = pd.read_csv(second_file)

    labels_names = list(df.columns[1:])

    if is_output_classname:
        labels_json = {"labels_name": labels_names}
        with open("labels_names.json", "w") as f:
            json.dump(labels_json, f)

    with open("output.jsonl", "w") as f:
        for index, row in df.iterrows():
            id_ = index + 1
            filepath = row["ImageID"]
            text = ""
            labels = list(row[1:])
            data = {
                "id": id_,
                "filepath": "padchest/" + filepath,
                "text": text,
                "labels": labels,
            }
            f.write(json.dumps(data) + "\n")

    print("Conversion completed.")


def select_csv(input_file, source_folder, destination_folder="padchest"):
    df = pd.read_csv(input_file)

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for image_id in df["ImageID"]:
        source_path = os.path.join(source_folder, image_id)
        destination_path = os.path.join(destination_folder, image_id)

        if os.path.exists(source_path):
            shutil.copy(source_path, destination_path)
        else:
            print(f"File {image_id} not found in the source folder.")

    print("Files copied successfully.")

    selected_rows = df[df["Projection"].str.contains("AP|PA")]

    selected_columns = selected_rows.iloc[
        :, [0] + list(range(35, len(selected_rows.columns)))
    ]
    output_file = "processed_file.csv"

    selected_columns.to_csv(output_file, index=False)
    return output_file


def selected_labels(input_file):
    df = pd.read_csv(input_file)

    ones_count = df.iloc[:, 1:].sum()
    l_count = list(ones_count[ones_count >= 50].sort_values(ascending=False).index)
    print(len(l_count))
    selected_columns = df.loc[:, ["ImageID"] + l_count]

    selected_columns.columns = ["ImageID"] + [
        col.replace("label_", "") for col in selected_columns.columns[1:]
    ]
    output_file = "further_processed_file.csv"

    selected_columns.to_csv(output_file, index=False)
    return output_file
