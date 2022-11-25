import json

import streamlit as st
from network_component.network_component import network_component

from generate.generation import Network_Generator
from solve.rule_based import Rule_Agent

st.set_page_config(page_title="RN III Task Explorer")
st.write("""
            # RN III Task Explorer
            This is an interactive application to explore stimuli and task 
            design for the Reward Networks III project. 
         """)
networks = None

# ------------------- Sidebar -------------------
with st.sidebar:
    st.write("""
             ### Overview of sections
             * In the **Generate** section a user can specify stimuli parameters and generate a set of networks. The networks are saved in a JSON file
             * In the **Visualize section** the user can visualize the network and try to solve it, keeping track of moves and comparing the solution to other strategies' solutions.
             * In the **Compare** section we visualize the distribution of scores obtained over solving a collection of networks using different strategies 
             """)

    # -------------------
    # Generate
    # -------------------
    with st.expander("Generate"):
        # submit parameters for generation
        gen_params = {}
        data = None

        with st.form("generate_form", clear_on_submit=False):
            st.write("Select the generation parameters")
            # how many networks to generate?
            gen_params['n_networks'] = st.number_input(
                label='How many networks do you want to generate?',
                min_value=1,
                max_value=100_000,
                value=1,
                step=10)
            # how many rewards do you want?
            gen_params['n_rewards'] = st.number_input(
                label='How many rewards in the network?',
                min_value=2,
                max_value=5,
                value=5,
                step=1)
            # what are the reward values?
            rewards_str = st.text_input(
                label="Insert the reward values separated by a space",
                value="-100 -20 0 20 140")
            gen_params['rewards'] = [int(i) for i in rewards_str.split(" ")]
            gen_params['n_steps'] = st.number_input(
                label='How many steps to solve the network?',
                min_value=2,
                max_value=10,
                value=8,
                step=1)
            gen_params['n_levels'] = st.number_input(
                label='How many levels in the network?',
                min_value=1,
                max_value=4,
                value=4,
                step=1)

            # how many nodes in each level?
            # TODO

            # download the data yes or no?
            to_download_data = st.checkbox("Download the generated networks")   
            # download the data yes or no?
            to_download_solutions = st.checkbox("Download the generated networks' solutions")   

            # Every form must have a submit button.
            submitted = st.form_submit_button("Generate")
            if submitted:
                st.info('Parameters submitted!')
                if gen_params['n_rewards']!=len(gen_params['rewards']):
                    st.error("Number of rewards and rewards in the text field do not correspond, please submit again parameters")
                else:
                    st.info("Number of rewards and rewards in the text field correspond")

                # Network_Generator class
                G = Network_Generator(gen_params)
                save_path = "TODO"
                networks = G.generate(save_path)
                st.success("Networks generated!")
                if to_download_data:
                    data=G.save_as_json()

                # Solve networks with strategies
                Myopic_agent = Rule_Agent(networks,"myopic",gen_params)
                Myopic_agent.solve()
                Loss_agent = Rule_Agent(networks,"take_first_loss",gen_params)
                Loss_agent.solve()
                st.success("Solutions to networks calculated!")


        # download button cannot be used inside form        
        if to_download_data:
            st.download_button(
                label="Download data as JSON",
                data=data,
                file_name='networks.json')
        if to_download_solutions:
            st.download_button(
                label="Download solutions (myopic)",
                data=Myopic_agent.save_solutions_frontend(),
                file_name='solutions_myopic.json')
            st.download_button(
                label="Download solutions (loss)",
                data=Loss_agent.save_solutions_frontend(),
                file_name='solutions_loss.json')




    # -------------------
    # Visualize
    # -------------------

    with st.expander("Visualize"):
        col1, col2 = st.columns(2)

        with col1:
            # TODO get list of network ids
            network_id = st.selectbox("Which network to visualize?",
                                      ("Email", "Home phone",
                                       "Mobile phone"))

        with col2:
            strategies = st.multiselect(
                'Which strategy solution do you want to see?',
                ['Myopic', 'Loss'],
                ['Myopic'])

        st.write("Insert custom visualization component here!")


if networks is not None:
    # add starting node
    net_to_plot = networks[0]
    net_to_plot['nodes'][net_to_plot['starting_node']]['starting_node'] = True
    # convert dict to string
    # set separators=(',', ':') to remove spaces
    networks_str = json.dumps(net_to_plot, separators=(',', ':'))
    network_component(timer=100, network_args=networks_str)
else:
    network_component(100)


# -------------------
# Compare
# -------------------

with st.expander("Compare"):
    st.write("TODO")

# Display scores distribution
# scores_melt = scores.melt(var_name='Experiment', value_name='Measurement')
# fig = sns.displot(scores_melt,
#                   x='Measurement',
#                   binwidth=.2,
#                   hue='Experiment',
#                   aspect=2,
#                   element='step')
# st.pyplot(fig)
