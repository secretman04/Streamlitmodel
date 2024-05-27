#!/usr/bin/env python
# coding: utf-8

# #### Import libraries and datasets

# In[116]:



import streamlit as st
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics
import statsmodels.api as sm
import matplotlib.pyplot as plt
import fsspec

fs = fsspec.filesystem('file')

# In[117]:


df = pd.read_csv("C://Users/zhiyu/Documents/IDA Sem 4/covid19.csv")




def calculate_incident_rate_difference(population_vaccinated, population_unvaccinated, vaccinated_cases, unvaccinated_cases):
    incident_rate_difference = (vaccinated_cases / population_vaccinated) - (unvaccinated_cases / population_unvaccinated)
    return incident_rate_difference

# Create Streamlit UI
st.title("Incident Rate Calculator")

# Input fields for user
st.write("Input Data")
col1, col2 = st.columns(2)
population_vaccinated = col1.number_input("Enter population of vaccinated individuals:",min_value=334402, max_value=22805163)
population_unvaccinated = col1.number_input("Enter population of unvaccinated individuals:",min_value=6810752,max_value=30550927)
vaccinated_cases = col2.number_input("Enter number of cases among vaccinated individuals:",min_value=5,max_value=3435)
unvaccinated_cases = col2.number_input("Enter number of cases among unvaccinated individuals:",min_value=442, max_value=13176)

# Display info
total_population = population_vaccinated + population_unvaccinated
total_cases = vaccinated_cases + unvaccinated_cases


st.write("Info")
col1,col2 = st.columns(2)
col1.metric(label="Total population",value=f"{total_population:,.2f}")
col2.metric(label="Total cases",value=f"{total_cases:,.2f}")


# Button to trigger calculation
if st.button("Calculate Incident Rate Difference"):
    # Calculate incident rate difference
    incident_rate_difference = calculate_incident_rate_difference(population_vaccinated, population_unvaccinated, vaccinated_cases, unvaccinated_cases)
    
    # Display result
    st.write("Incident Rate Difference:", incident_rate_difference)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.scatter(total_population, incident_rate_difference, color='blue')
    ax.set_xlabel('Total Population')
    ax.set_ylabel('Incident Rate Difference')
    ax.set_title('Incident Rate Difference vs Total Population')
    ax.grid(True)
    st.pyplot(fig)



# Modeling
X = df[['population_vaccinated', 'population_unvaccinated', 'vaccinated_cases', 'unvaccinated_cases']]
df['incident_rate_difference'] = df['vaccinated_cases'] / df['population_vaccinated'] - df['unvaccinated_cases'] / df['population_unvaccinated']
y = df['incident_rate_difference']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lm = LinearRegression()
lm.fit(X_train, y_train)

y_pred = lm.predict(X_test)

st.subheader("Visualize results:")
st.title("Incident Rate Difference Plot")

# Placeholder to display selected range
selected_range_placeholder = st.empty()

# Slider for range -0.0000 to 0.0000
amount = st.slider("Adjust the range of incident rate difference values:", min_value=-0.0008, max_value=0.0000, step=0.0001, value=[-0.0008, 0.0000])

# Update placeholder with selected range dynamically
selected_range_placeholder.write(f"Selected range: {amount[0]} to {amount[1]}")

# Plotting
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, label='Data Point', s=50)  # Increase marker size to 50
ax.set_xlabel('Actual Incident Rate Difference')
ax.set_ylabel('Predicted Incident Rate Difference')
ax.set_title('Actual vs Predicted Incident Rate Difference of Cases between Population')
ax.legend()

# Adjust the plot based on the range selected by the user
ax.set_xlim(amount[0], amount[1])  # Adjust the x-axis limit based on the range
ax.set_ylim(amount[0], amount[1])  # Adjust the y-axis limit based on the range

# Display the plot
st.pyplot(fig)


# Plotting
st.title("Incident Rate Difference Plot")

# Slider to adjust the number of data points
num_points = st.slider("Select number of data points:", min_value=5, max_value=len(df), value=10)

# Filter the dataframe based on the selected number of data points
filtered_df = df.head(num_points)

# Calculate incident rates
filtered_df['vaccinated_incident_rate'] = filtered_df['vaccinated_cases'] / filtered_df['population_vaccinated']
filtered_df['unvaccinated_incident_rate'] = filtered_df['unvaccinated_cases'] / filtered_df['population_unvaccinated']

# Plotting
plt.figure(figsize=(10, 6))
plt.bar(filtered_df.index, filtered_df['vaccinated_incident_rate'], label='Vaccinated Cases', alpha=0.7, color='blue')
plt.bar(filtered_df.index, filtered_df['unvaccinated_incident_rate'], label='Unvaccinated Cases', alpha=0.7, color='pink')

plt.xlabel('Data Points')
plt.ylabel('Incident Rate')
plt.title('Difference in Incident Rates between Vaccinated and Unvaccinated Cases')
plt.legend()

# Display the plot
st.pyplot(plt)