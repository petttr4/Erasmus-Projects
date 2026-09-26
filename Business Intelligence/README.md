# Business Intelligence
# Wellness Tracker: Data-Driven Health & Fitness Insights

Business Intelligence project developed for the BI course at Universidade de Coimbra (Erasmus mobility program, 2025).

**Team:** Petroula Tsimpini, Georgia Poimenidou, David Blazheski

The project combines a full BI pipeline (ETL → Data Warehouse → OLAP dashboards) with a Data Mining phase (classification, clustering and a recommendation system) built on top of the [FitLife Health and Fitness Tracking dataset](https://www.kaggle.com/datasets/jijagallery/fitlife-health-and-fitness-tracking-dataset) from Kaggle.

## Project Idea

By analyzing demographic information, activity metrics, health indicators and lifestyle metrics, the project aims to find connections between them and provide insights that can help Health & Wellness centers and fitness apps improve user engagement and well-being.

## Dataset

- **Source:** Kaggle — "FitLife Health and Fitness Tracking"
- **Size:** ~78 MB
- **Participants:** 3,000 users
- **Period:** 1 year of daily health records (2024)
- Attributes include: age, gender, height, weight, BMI, activity type, duration, calories burned, heart rate, blood pressure, sleep hours, stress level, hydration level, daily steps, fitness level, smoking status.

## BI Pipeline (Part 1)

- **ETL:** Pentaho (`.ktr` job) + Python/Pandas for staging, cleaning, and categorization (e.g. bucketing heart rate and blood pressure into Low/Medium/High ranges).
- **Data Warehouse:** PostgreSQL, loaded via a Python script, modeled as a star schema (`Person`, `Activity`, `Health`, `Activity_type`, `TimeInfo`).
- **OLAP / Dashboards:** Tableau — two dashboards covering demographics vs. activity, smoking vs. fitness, stress vs. steps, blood pressure vs. stress, hydration vs. heart rate, and more.
  - Tableau profile: https://public.tableau.com/app/profile/david.blazheski/vizzes

## Data Mining (Part 2)

Three techniques were implemented as standalone Python scripts:

### 1. Classification — Churn Prediction (`Classify.py`)
Predicts whether a user is "At Risk of Quitting" or "Retained", based on aggregated per-user metrics (age, height, avg. weight, calories burned, heart rate, BMI, fitness level, activity/intensity proportions). Churn is defined as fewer than 230 recorded sessions.
- Model: **Random Forest Classifier**, with **SMOTE** to balance the training set.
- Results: ~0.79 accuracy, ~0.79 ROC-AUC.
- Outputs a churn summary bar chart and a confusion matrix.

### 2. Clustering — User Segmentation
Two separate clustering approaches were tried:
- **`Clusters-notdistinct.py`** — first attempt, clustering users by workout intensity and activity-type mix (K-Means, k=3). Clusters were not well separated (visualized with t-SNE), so a second approach was developed.
- **`Clusters2.py`** — final approach, clustering users' latest entries on age, BMI, duration, calories burned, heart rate, sleep, stress, steps, hydration, and fitness level (K-Means, k=4, visualized with PCA). Produced 4 interpretable segments: Young/High-Activity, Older/Consistent, Older/Low-Activity, Young/Inefficient.

### 3. Recommendation System — Collaborative Filtering (`App.py`)
A **Streamlit** app that recommends the top 3 workout types for a user based on the K-Nearest Neighbors (n=10) of similar users (age, gender, height, weight, BMI, stress level, preferred duration).

## Tech Stack

- **Python:** pandas, scikit-learn, imbalanced-learn (SMOTE), matplotlib, seaborn
- **Streamlit:** interactive recommender UI
- **Pentaho:** ETL
- **PostgreSQL:** data warehouse
- **Tableau:** OLAP dashboards

## Project Structure

```
.
├── App.py                      # Streamlit recommendation app (KNN)
├── Classify.py                 # Churn classification (Random Forest + SMOTE)
├── Clusters2.py                # Final clustering (K-Means, k=4, PCA)
├── Clusters-notdistinct.py     # Initial clustering attempt (K-Means, k=3, t-SNE)
├── Fitness_Pentaho.ktr         # Pentaho ETL job
├── LoadinPostgre.ipynb         # Notebook: loading transformed data into PostgreSQL
├── DataMining.ipynb            # Notebook: data mining exploration
├── dataset.ipynb               # Notebook: initial dataset exploration
├── health_fitness_dataset.csv  # Raw dataset used by the scripts
├── categorized_fitlife_data.csv# Transformed/categorized dataset (post-ETL)
├── LinktoTableau.txt           # Link to the Tableau dashboards

```

## Installation

```bash
git clone <this-repo-url>
cd <repo-folder>
pip install -r requirements.txt
```

## Usage

Run the classification script:
```bash
python Classify.py
```

Run the clustering script:
```bash
python Clusters2.py
```

Launch the workout recommender:
```bash
streamlit run App.py
```

## Notes

- The classification and clustering scripts are standalone analyses (they read the CSV directly and produce matplotlib/seaborn plots); they are not integrated into the Streamlit app.
- Only the recommendation system has an interactive UI (`App.py`).
- This project was built for educational purposes as part of a Business Intelligence course assignment. Results and recommendations are not intended for production or medical use.
