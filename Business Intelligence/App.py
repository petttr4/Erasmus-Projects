import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import streamlit as st

data = pd.read_csv("health_fitness_dataset.csv")

features = ['age', 'gender', 'height_cm', 'weight_kg', 'bmi', 'stress_level', 'duration_minutes']
target = 'activity_type'
data_filtered = data[features + [target]].dropna()

le_gender = LabelEncoder()
data_filtered['gender_encoded'] = le_gender.fit_transform(data_filtered['gender'])

X = data_filtered[['age', 'gender_encoded', 'height_cm', 'weight_kg', 'bmi', 'stress_level', 'duration_minutes']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

knn = NearestNeighbors(n_neighbors=10)
knn.fit(X_scaled)

def recommend_workouts(user_input, top_n=3):
    user_input_encoded = user_input.copy()
    user_input_encoded[1] = le_gender.transform([user_input[1]])[0]
    user_scaled = scaler.transform([user_input_encoded])
    distances, indices = knn.kneighbors(user_scaled)
    activities = data_filtered.iloc[indices[0]][target].values
    most_common = Counter(activities).most_common(top_n)
    return [activity for activity, _ in most_common]

st.title("💪 Personalized Workout Recommender")

st.write("Fill in your information to get the top 3 workout suggestions based on users like you.")

age = st.slider("Age", 15, 80, 25)
gender = st.selectbox("Gender", options=["M", "F"])
height = st.slider("Height (cm)", 140, 220, 175)
weight = st.slider("Weight (kg)", 40, 150, 70)
bmi = weight / ((height / 100) ** 2)
stress = st.slider("Stress Level (1=Low, 10=High)", 1, 10, 3)
duration = st.slider("Preferred Workout Duration (minutes)", 15, 120, 45)

if st.button("Get My Recommendations"):
    user_profile = [age, gender, height, weight, bmi, stress, duration]
    recommendations = recommend_workouts(user_profile)
    st.subheader("🏋️‍♂️ Top 3 Recommended Workouts:")
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"**{i}. {rec}**")