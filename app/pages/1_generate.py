import streamlit as st
import time
import numpy as np
from generate import Network_Generator
from solve import Rule_Agent

st.set_page_config(page_title="Generate a Network")
st.markdown("# Generate a Network")
st.sidebar.header("Generate")

# submit parameters for generation
params = {}
with st.form("generate_form",clear_on_submit = False):

    st.write("Select the generation parameters")
    # how many networks to generate?
    params['n_networks'] = st.number_input(label='How many networks do you want to generate?',
                                          min_value=1,
                                          max_value=100000,
                                          value=1,
                                          step=10)
    # how many rewards do you want?
    params['n_rewards'] = st.number_input(label='How many rewards in the network?',
                                          min_value=2,
                                          max_value=5,
                                          value=5,
                                          step=1)
    # what are the reward values?
    rewards_str = st.text_input(label="Insert the reward values separated by a space",
                                value="-20 0 20")
    params['rewards'] = [int(i) for i in rewards_str.split(" ")]

    # how many nodes in each level?
    #TODO

    # Every form must have a submit button.
    submitted = st.form_submit_button("Generate")
    if submitted:
        st.info('Parameters submitted!')
        
        # Network_Generator class
        G = Network_Generator(params)
        save_path = "TODO"
        network = G.generate(save_path)
        with st.expander("See network in JSON format:"):
            st.json(network)

        # collect the network solution for differen strategies
        Myopic_agent = Rule_Agent(network,"myopic")
        Myopic_agent.solve()

        Loss_agent = Rule_Agent(network,"take_first_loss")
        Loss_agent.solve()
   

