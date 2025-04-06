import med_minds as minds
import os
import pandas as pd

PATH_TO_SAVE_DATA = "./download"
MANIFEST_PATH = PATH_TO_SAVE_DATA + "/manifest.json"

if not os.path.exists(PATH_TO_SAVE_DATA):
    os.makedirs(PATH_TO_SAVE_DATA)

tables = minds.get_tables()
gdc_cohort = minds.build_cohort(
    gdc_cohort="cohort_hackathon_cohort.2025-03-20 (1).tsv",
    output_dir="./download",
    manifest=MANIFEST_PATH if os.path.exists(MANIFEST_PATH) else None,
)
# # to get the cohort details
print(gdc_cohort.stats())

# load the tsv file
ids = pd.read_csv("cohort_hackathon_cohort.2025-03-20 (1).tsv", sep="\t", dtype=str)

for table in tables:
    query_df = minds.query(
        "SELECT * FROM {} WHERE cases_case_id IN ({})".format(
            table, ",".join([f"'{i}'" for i in ids["id"]])
        )
    )
    query_df.to_csv(f"./download/{table}.csv", index=False)

gdc_cohort.download(threads=64)

