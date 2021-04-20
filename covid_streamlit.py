import streamlit as st
import pandas as pd
import numpy as np

# ==== Headers ====
'''# Dashboard Covid19
datasearch: https://ourworldindata.org/covid-cases'''

# dataset
@st.cache(allow_output_mutation=True)
def get_data(path):
    data = pd.read_csv(path)
    return data
# Get data
path = 'dataset/owid-covid-data.csv'
df = get_data(path)

def get_describe(path2):
    describe = pd.read_csv(path2)
    return describe

path2 = 'dataset/describe.csv'
describe = get_describe(path2)

# ==== Create a column filter ====
st.sidebar.subheader('Filtering')
features = st.sidebar.multiselect(
    'Selecting columns',
    df.columns
)

country = st.sidebar.multiselect(
    'Selecting countries',
    df.iso_code.unique()
)

# Filtering

if (features != []) & (country != []):
    df = df.loc[df['iso_code'].isin(country), features]
    
elif (features != []) & (country == []):
    df = df.loc[:, features]
    
elif (features == []) & (country != []):
    df = df.loc[:, :]
    
else:
    df = df.copy()

st.dataframe(df.head(1000))

# ==== Selecting column describe ====
st.sidebar.subheader('Column Describe')
if st.sidebar.checkbox('See Describe'):
    st.dataframe(describe)


