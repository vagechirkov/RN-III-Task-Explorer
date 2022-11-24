import streamlit as st
# from generate.generation import Network_Generator
# from solve.rule_based import Rule_Agent

st.set_page_config(page_title="RN III Task Explorer")

st.write("""
         # RN III Task Explorer 

         This is an interactive application to explore stimuli and task design
         for the Reward Networks III project. 
         
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
    params = {}
    with st.form("generate_form", clear_on_submit=False):
        st.write("Select the generation parameters")
        # how many networks to generate?
        params['n_networks'] = st.number_input(
            label='How many networks do you want to generate?',
            min_value=1,
            max_value=100000,
            value=1,
            step=10)
        # how many rewards do you want?
        params['n_rewards'] = st.number_input(
            label='How many rewards in the network?',
            min_value=2,
            max_value=5,
            value=5,
            step=1)
        # what are the reward values?
        rewards_str = st.text_input(
            label="Insert the reward values separated by a space",
            value="-20 0 20")
        params['rewards'] = [int(i) for i in rewards_str.split(" ")]

        # how many nodes in each level?
        # TODO

        # Every form must have a submit button.
        submitted = st.form_submit_button("Generate")
        if submitted:
            st.info('Parameters submitted!')

            # Network_Generator class
            # G = Network_Generator(params)
            save_path = "TODO"
            # networks = G.generate(save_path)
            st.write("See network in JSON format:")
            # st.json(networks[0])

            # Solve networks with strategies (TODO)
            # Myopic_agent = Rule_Agent(networks,"myopic")
            # Myopic_agent.solve()
            # Loss_agent = Rule_Agent("take_first_loss")
            # Loss_agent.solve()

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
