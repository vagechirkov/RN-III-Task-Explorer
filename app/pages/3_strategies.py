import streamlit as st
import time
import numpy as np
import seaborn as sns
import pandas as pd
from solve import Rule_Agent

st.set_page_config(page_title="Compare strategies across all networks")
st.markdown("# Compare strategies across all networks")
st.sidebar.header("Strategies")

# Solve networks with strategies (TODO)
Myopic_agent = Rule_Agent("myopic")
Myopic_agent.solve()
Loss_agent = Rule_Agent("take_first_loss")
Loss_agent.solve()

st.markdown("## Strategies used")

# Display scores distribution
scores_melt = scores.melt(var_name='Experiment', value_name='Measurement')
fig = sns.displot(scores_melt,
                x='Measurement',
                binwidth=.2,
                hue='Experiment',
                aspect=2,
                element='step')
st.pyplot(fig)