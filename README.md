# Agriculture Crop Pattern Mining

Data mining project: preprocessing, Apriori association rules, K-Means clustering,
outlier detection (IQR + Isolation Forest), with tables and visualizations.

## Run (VS Code)
1. Extract the zip and open the folder in VS Code (File > Open Folder).
2. Open a terminal (Ctrl + `) and run:
       pip install -r requirements.txt
       python crop_pattern_mining.py
   Or double-click `run.bat` (Windows) / run `./run.sh` (Mac/Linux).

Results (CSV tables + PNG charts) are written to the `output/` folder.

## Using your own dataset
Put your CSV at `data/crop_data.csv` (replace the sample), or run:
       python crop_pattern_mining.py path/to/file.csv
Needed: a Crop column. Optional: Season, Region/State/District, Soil, Rainfall,
Temperature, Area, Production, Yield (Yield = Production / Area if missing).

NOTE: `data/crop_data.csv` is a synthetic SAMPLE so the project runs out of the box.
Replace it with a real dataset (e.g. from Kaggle / data.gov.in) for your final report.

## Tuning
Edit MIN_SUPPORT, MIN_CONFIDENCE, MIN_LIFT at the top of crop_pattern_mining.py.

## Output files
cleaned_data.csv, frequent_itemsets.csv, association_rules.csv, crop_clusters.csv,
outliers.csv, final_results.csv, crop_by_season.csv and charts (*.png).
