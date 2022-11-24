import streamlit as st
import time
import numpy as np

st.set_page_config(page_title="Visualize a Network and Try to Solve")

st.markdown("# Visualize a Network and Try to Solve")
st.sidebar.header("Visualize")

col1, col2 = st.columns(2)

with col1:
    option = st.selectbox("Which network to visualize?",
                         ("Email", "Home phone", "Mobile phone")) #TODO get list of network ids
    

with col2:
    options = st.multiselect('Which strategy solution do you want to see?',
                             ['Myopic', 'Loss'],
                             ['Myopic'])

st.write("Insert custom visualization component here!")
    