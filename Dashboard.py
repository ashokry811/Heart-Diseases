import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
file_path = 'cleaned_dataset.csv'  # Make sure this file is in the same directory
try:
    df = pd.read_csv(r'cleaned_dataset.csv')
except FileNotFoundError:
    st.error(f"File not found. Please upload the file: {file_path}")

# Handling missing values
if df.isnull().values.any():
    st.warning("Dataset contains missing values. Please check and clean your data if necessary.")

# Title and Description
st.title("Healthcare Predictive Analytics Dashboard")
st.markdown("Explore healthcare data with various metrics and visualizations.")

# Sidebar for Filters
st.sidebar.header("Filter the Data")

# Filter by age range
age_range = st.sidebar.slider('Select Age Range',
                              min_value=int(df['age'].min()),
                              max_value=int(df['age'].max()),
                              value=(30, 60))

# Filter by sex using multiselect
selected_genders = st.sidebar.multiselect('Select Gender',
                                           options=df['sex'].unique(),
                                           default=df['sex'].unique())

# Apply filters to dataset
filtered_df = df[(df['age'] >= age_range[0]) & (df['age'] <= age_range[1])]

# Apply gender filter
if selected_genders:
    filtered_df = filtered_df[filtered_df['sex'].isin(selected_genders)]

# Check if there is any data left after filtering
if filtered_df.empty:
    st.warning(f"No data available for the selected filters: Age Range = {age_range}, Gender = {selected_genders}")
else:
    # Total Records after filtering
    st.write(f"Total Records: {filtered_df.shape[0]}")

    # Visualizations
    st.subheader("Visualizations")

    # Chest Pain Type Distribution
    fig = px.bar(filtered_df, x='cp', title='Chest Pain Type Distribution',
                 labels={'cp': 'Chest Pain Type', 'count': 'Number of Patients'})
    st.plotly_chart(fig)

    # Gender Distribution Pie Chart
    gender_counts = filtered_df['sex'].value_counts()

    # Check if gender_counts has data to plot
    if not gender_counts.empty:
        fig = px.pie(values=gender_counts, names=gender_counts.index,
                     title='Gender Distribution')
        st.plotly_chart(fig)

    # Age vs Cholesterol Scatter Plot
    fig = px.scatter(filtered_df, x='age', y='chol', color='sex',
                     labels={'chol': 'Cholesterol (mg/dl)'}, title='Age vs Cholesterol')
    st.plotly_chart(fig)

    # Resting Blood Pressure by Gender
    fig = px.box(filtered_df, x='sex', y='trestbps',
                 labels={'sex': 'Gender', 'trestbps': 'Resting Blood Pressure (mm Hg)'},
                 title='Resting Blood Pressure by Gender')
    st.plotly_chart(fig)

    # Summary Statistics
    st.subheader("Summary Statistics")
    st.write(filtered_df[['age', 'chol', 'trestbps', 'thalch', 'oldpeak']].describe())

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")

    # Select only numeric columns for correlation
    numeric_df = filtered_df.select_dtypes(include='number')

    # Calculate the correlation matrix
    corr_matrix = numeric_df.corr()

    # Create the heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Download button for the filtered data
st.sidebar.download_button(
    label="Download Filtered Data",
    data=filtered_df.to_csv(index=False),
    file_name="filtered_healthcare_data.csv",
    mime='text/csv'
)
