"""
    this script is used for MIMIC-CXR.
    Extract the “IMPRESSION” or “CONCLUSION” fields,
    and omitted any physician identifiers or view positions sequences by simple matching rules.

"""

from concurrent.futures import ThreadPoolExecutor
import json
import os

from tqdm import tqdm
import spacy

import re

nlp = spacy.load("en_core_web_sm")


def get_all_files(directory, extensions):
    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                all_files.append(os.path.join(root, file))
    return all_files


def extract_between_markers(text):
    pattern = re.compile(
        r"(IMPRESSION:|CONCLUSION:|IMPRESSIONS:)\s*(.*?)\s*(NOTIFICATION:|NOTIFICATIONS:|RECOMMENDATION:|RECOMMENDATIONS:|RECOMMENDATION\(S\):).*?$",
        re.DOTALL,
    )

    match = pattern.search(text)

    if match:
        return cleansing(match.group(2).strip())
    else:
        pattern_without_end = re.compile(
            r"(IMPRESSION:|CONCLUSION:|IMPRESSIONS:)\s*(.*)", re.DOTALL
        )
        match = pattern_without_end.search(text)
        if match:
            return cleansing(match.group(2).strip())
        return None


def cleansing(text):
    content = re.sub("_", " ", text)
    inline_content = re.sub("\n", "", content)
    doc = nlp(inline_content)
    result = []
    for i, s in enumerate(doc.sents, start=1):
        if (
            "Dr." not in s.text
            and "PA and lateral" not in s.text
            and "AP chest" not in s.text
        ):
            result.append(s.text.strip())
    result = " ".join(result)

    return result


def process_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    result = extract_between_markers(content)

    return result


def process_files(filepaths):
    result_list = []

    for filepath in filepaths:
        subject_id = os.path.basename(os.path.dirname(filepath))
        subject_id = re.match(r"\D*(\d+)", subject_id).group(1)
        study_id = os.path.basename(filepath)
        study_id = re.match(r"\D*(\d+)", study_id).group(1)

        processed_text = process_file(filepath)

        if processed_text is not None and processed_text:
            result = {
                "subject_id": subject_id,
                "study_id": study_id,
                "text": processed_text,
            }
            result_list.append(result)

    return result_list


if __name__ == "__main__":
    mimic_cxr_reports_directory = "mimic-cxr-reports/files"
    output_jsonl_file = "processed_text.jsonl"

    all_files = get_all_files(mimic_cxr_reports_directory, [".txt"])

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(
            tqdm(
                executor.map(
                    process_files,
                    [all_files[i : i + 100] for i in range(0, len(all_files), 100)],
                ),
                total=len(all_files) // 100,
                desc="Processing files",
                unit="file",
            )
        )

    with open(output_jsonl_file, "w", encoding="utf-8") as jsonl_file:
        for result_list in results:
            for result in result_list:
                jsonl_file.write(json.dumps(result, ensure_ascii=False) + "\n")
