# All of Us analysis

Notebooks for running the heart failure clustering study on the
[All of Us Researcher Workbench](https://workbench.researchallofus.org/). They are separate from the
OHDSI R package: All of Us data is queried from the Workbench CDR, not through DatabaseConnector, so
`execute()` does not run them.

| Notebook | Step |
|---|---|
| `01_generate_hf_symptoms_concepts.ipynb` | Heart failure symptom concept sets |
| `02_build_cohort_aou.ipynb` | Heart failure cohort in women under 55 |
| `03_feature_extraction_aou.ipynb` | Feature matrix matching the JHU features |
| `04_jhu_model_transfer_aou.ipynb` | Assign clusters with the JHU-trained VaDE |
| `04_optimization_de_novo_aou.ipynb` | Optuna study and VaDE trained on All of Us |
| `05_feature_enrichment_analysis_aou.ipynb` | Per-cluster feature enrichment |
| `06_posthoc_analysis_aou.ipynb` | XGBoost surrogate and SHAP figures |
| `07_cox_model_aou.ipynb` | Cox and Kaplan-Meier models by cluster |

`aou_src/` holds the VaDE code the notebooks import, and `helpers/` the notebook helpers.
`data/` maps covariate ids to the feature names the JHU model was trained on.

Notebook outputs are stripped, and `output/` and `plots/` are git-ignored. The All of Us Data and
Statistics Dissemination Policy forbids sharing participant-level data or counts of 1 to 20 outside
the Workbench, so review anything before moving it off the Workbench.
