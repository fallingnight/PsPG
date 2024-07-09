"""
    this script is used for MIMIC-CXR.
    Merge the image's metadata and corresponding preprocessed text, uniquely identified by dicom_id, study_id, subject_id

"""

import csv
import json
import pandas as pd


def check_empty(file_path):
    jsonl_file_path = file_path

    empty_text_lines = []

    with open(jsonl_file_path, "r", encoding="utf-8") as jsonl_file:
        for line_number, line in enumerate(jsonl_file, start=1):
            data = json.loads(line.strip())
            text = data.get("text", "")

            if not text:
                empty_text_lines.append(line_number)

    if empty_text_lines:
        print("Lines with empty 'text' field:")
        for line_number in empty_text_lines:
            print(line_number)
    else:
        print("No lines with empty 'text' field.")


def merge():
    selected_rows = []

    with open(
        "mimic-cxr-2.0.0-metadata.csv/mimic-cxr-2.0.0-metadata.csv",
        "r",
        encoding="utf-8",
    ) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            if row["ViewPosition"] in ["AP", "PA"]:
                selected_rows.append(row)

    jsonl_data = []
    with open("processed_text.jsonl", "r", encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            json_data = json.loads(line.strip())
            jsonl_data.append(json_data)

    df1 = pd.DataFrame(selected_rows)
    df2 = pd.DataFrame(jsonl_data)

    matched_data = pd.merge(df1, df2, on=["study_id", "subject_id"])
    result = []
    for index, row in matched_data.iterrows():
        result.append(
            {
                "dicom_id": row["dicom_id"],
                "study_id": row["study_id"],
                "subject_id": row["subject_id"],
                "text": row["text"],
            }
        )

    with open("matched_results.jsonl", "w", encoding="utf-8") as output_file:
        for item in result:
            output_file.write(json.dumps(item, ensure_ascii=False) + "\n")
