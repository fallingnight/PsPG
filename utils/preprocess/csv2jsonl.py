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
    # print("allzeros count: %d"%count)
    # output as jsonl
    with open(jsonl_filepath, "w", encoding="utf-8") as jsonl_file:
        for data in jsonl_data:
            jsonl_file.write(json.dumps(data) + "\n")
    if is_output_classname:
        with open(class_filepath, "w", encoding="utf-8") as categories_file:
            json.dump(categories_data, categories_file)


extract_chexpert_data(
    "val_labels.csv",
    "val_dataset.jsonl",
    type="val",
    filepath_prefix="CheXpert-v1.0/val",
)
