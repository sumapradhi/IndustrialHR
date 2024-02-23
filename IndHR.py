import os

import pandas as pd
import numpy as np
import requests

import streamlit as st
from streamlit_option_menu import option_menu
import plotly.express as px
from PIL import Image

import matplotlib.pyplot as plt

import seaborn as sns
import klib

import geopandas as gpd
import folium
from streamlit_folium import folium_static

# Here you should have some code that creates or reads a DataFrame and assigns it to merged_df
merged_df = pd.read_csv('C://Users//sumat//OneDrive//Documents//IND_HR//merged_file.csv')


#Drop columns which are not required

# After defining merged_df, you can use it as needed

merged_df.drop(columns=['State Code', 'District Code',
                        'Main Workers - Total -  Persons','Main Workers - Total - Males',
                       'Main Workers - Total - Females','Marginal Workers - Total -  Persons',
                       'Marginal Workers - Total - Males','Marginal Workers - Total - Females'], inplace=True)

merged_df.drop(columns=['Main Workers - Rural -  Persons','Main Workers - Urban -  Persons',
                       'Marginal Workers - Rural -  Persons','Marginal Workers - Urban -  Persons'], inplace=True)

#Filter data which are not required

filtered_rows = merged_df[merged_df['Districts'].str.contains('state', case=False)]
merged_df_filtered = merged_df.drop(filtered_rows.index)
merged_df_filtered.head()

# Create a boolean mask for rows containing '00' in 'division' column
total_rows_mask = merged_df_filtered['Division'].str.contains('00', case=False)

# Filter out rows containing '00' in 'division' column
merged_df_filterednew = merged_df_filtered[~total_rows_mask]

# Including Category column in the merged df
df_categories= pd.read_csv('C://Users//sumat//OneDrive//Documents//Categories.csv')
df_categories.head()

df_final=pd.merge(merged_df_filterednew,df_categories,on='Division',how='left')

#Clean the column names using klib library
merged_df=klib.clean_column_names(merged_df)


# Configuring Streamlit GUI
st.set_page_config(layout="wide")

st.markdown('<h1 style="color: blueviolet;text-align: center;">Industrial Human Resource Geo-Visualization</h1>', unsafe_allow_html=True)

with st.sidebar:
    selected = option_menu(None,
                       options = ["Home","Explore DATA","Reports"],
                       icons = ["rocket", "binoculars","clipboard"],
                       default_index=0,
                       orientation="vertical",
                       styles={"container": {"width": "100%"},
                               "icon": {"color": "white", "font-size": "24px"},
                               "nav-link": {"font-size": "24px", "text-align": "left", "margin": "-2px"},
                               "nav-link-selected": {"background-color": "#6F36AD"}})




# Streamlit UI

#Top 10 States Distribution in Streamlit
state_val = df_final['States'].value_counts().values
States = df_final['States'].value_counts().index.tolist()

st.title('Top 10 State wise Workers Distribution')

# Create the pie chart
fig, ax = plt.subplots()
ax.pie(state_val[:10], labels=States[:10], autopct='%1.2f%%')
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Display the chart in Streamlit
st.pyplot(fig)

#District wise Top 10 Workers Distribution in Streamlit
District_val = df_final['Districts'].value_counts().values
Districts = df_final['Districts'].value_counts().index.tolist()

st.title('Top 10 District wise Workers Distribution')

# Create the pie chart
fig, ax = plt.subplots()
ax.pie(District_val[:10], labels=Districts[:10], autopct='%1.2f%%')
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Display the chart in Streamlit
st.pyplot(fig)


# Display State and Districts vs WorkersCreate Streamlit app
st.title('Worker Counts by Category')

# Create dropdown widgets for selecting state and district
selected_state = st.selectbox('Select State', df_final['States'].unique())
selected_district = st.selectbox('Select District', df_final[df_final['States'] == selected_state]['Districts'].unique())

# Filter the DataFrame based on the selected state and district
filtered_data = df_final[(df_final['States'] == selected_state) & (df_final['Districts'] == selected_district)]

# Extract relevant columns for worker counts
worker_data = filtered_data[['Main Workers - Rural - Males', 'Main Workers - Rural - Females',
                             'Main Workers - Urban - Males', 'Main Workers - Urban - Females',
                             'Marginal Workers - Rural - Males', 'Marginal Workers - Rural - Females',
                             'Marginal Workers - Urban - Males', 'Marginal Workers - Urban - Females']]

# Sum the counts for each category
category_counts = worker_data.sum()

# Plot the bar chart
fig, ax = plt.subplots()
category_counts.plot(kind='bar', ax=ax)
plt.title('Counts of Workers by Category in {} - {}'.format(selected_state, selected_district))
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

# Display the graph in the Streamlit app
st.pyplot(fig)

json1 = f"C://Users//sumat//Downloads//states_india.geojson"
m = folium.Map(location=[23,47, 77.94], titles='CartoDB positron', name = "Light Map",
               zoom_start=5, attr="My Data attribution")

selected_state = st.selectbox('Select State', df_final['States'].unique())
selected_district = st.selectbox('Select District', df_final[df_final['States'] == selected_state]['Districts'].unique())

# Filter the DataFrame based on the selected state and district
filtered_data = df_final[(df_final['States'] == selected_state) & (df_final['Districts'] == selected_district)]

worker_data = ['Main Workers - Rural - Males', 'Main Workers - Rural - Females',
                             'Main Workers - Urban - Males', 'Main Workers - Urban - Females',
                             'Marginal Workers - Rural - Males', 'Marginal Workers - Rural - Females',
                             'Marginal Workers - Urban - Males', 'Marginal Workers - Urban - Females']

worker_dataselect = st.selectbox("Select Worker",worker_data)

folium.Choropleth(
        geo_data=json1,
        name='choropleth',
        data=df_final,
        columns=["States",worker_dataselect],  # Assuming the index contains state names and the values are worker counts
        key_on='feature.properties.States',  # Adjust key_on to match the structure of the GeoJSON data
        fill_color='YlGn',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='worker_dataselect'
    ).add_to(m)
folium.features.GeoJson('states_india.geojson',
                        name="States", popup=folium.features.GeoJsonPopup(field=["st_nm"])).add_to(m)
folium_static(m, width=1600, height=950)


