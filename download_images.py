import med_minds as minds

minds.update()

query = "SELECT * FROM minds.clinical WHERE project_id = 'TCGA-LUAD' LIMIT 20"

cohort = minds.build_cohort(query=query, output_dir="./data")

cohort.download(threads=4, exclude=["Clinical Data"])
