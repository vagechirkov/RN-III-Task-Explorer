import json

import pandas as pd
import seaborn as sns
import streamlit as st

from generate.generation import Network_Generator
from network_component.network_component import network_component
from solve.rule_based import Rule_Agent

st.set_page_config(page_title="RN III Task Explorer", layout="wide")

st.write("""
            # RN III Task Explorer
            This is an interactive application to explore stimuli and task 
            design for the Reward Networks III project. 
         """)

# ------------------------------------------------------------------------------
#                      sidebar: generate and download options
# ------------------------------------------------------------------------------
with st.sidebar:
    st.write("## Generate Networks")
    gen_params = {}
    data = None

    # submit parameters for generation
    with st.form("generate_form", clear_on_submit=False):
        st.write("### Generate Parameters")
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

        # download title
        st.write("### Download Networks Options")

        # download the data yes or no?
        to_download_data = st.checkbox("Download the generated networks")
        # download the data yes or no?
        to_download_solutions = st.checkbox(
            "Download the generated networks' solutions")

        # Every form must have a submit button.
        submitted = st.form_submit_button("Generate")
        if submitted:
            st.info('Parameters submitted!')
            if gen_params['n_rewards'] != len(gen_params['rewards']):
                st.error(
                    "Number of rewards and rewards in the text field do not"
                    " correspond, please submit again parameters")
            else:
                st.info(
                    "Number of rewards and rewards in the text field "
                    "correspond")

            # Network_Generator class
            G = Network_Generator(gen_params)
            save_path = "TODO"
            networks = G.generate(save_path)
            st.session_state.networks = networks
            st.success("Networks generated!")
            if to_download_data:
                data = G.save_as_json()

            # Solve networks with strategies
            Myopic_agent = Rule_Agent(networks, "myopic", gen_params)
            Myopic_agent.solve()
            st.session_state.myopic_solutions = Myopic_agent.df
            Loss_agent = Rule_Agent(networks, "take_first_loss", gen_params)
            Loss_agent.solve()
            st.session_state.loss_solutions = Loss_agent.df
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

# ------------------------------------------------------------------------------
#                                   Compare
# ------------------------------------------------------------------------------
with st.expander("Compare strategies 🤖"):
    if "networks" in st.session_state:
        # create solution data file with all strategies in one file 
        strategy_data = pd.concat([st.session_state.myopic_solutions,
                                   st.session_state.loss_solutions],
                                  ignore_index=True)
        strategy_data_final = strategy_data[strategy_data['step'] == 8]

        g = sns.displot(data=strategy_data_final, x="total_reward",
                        hue="strategy", kind="hist")
        g.set(xlabel='Final total reward', ylabel='Count',
              title=f'Strategy final total reward comparison')
        # show figure in streamlit
        st.pyplot(g)

        # show figure in streamlit
        # st.pyplot(sns.boxplot(data=strategy_data_final,x="strategy", y="total_reward"))

        g3 = sns.relplot(
            data=strategy_data,
            x="step",
            y="reward",
            col='strategy',
            hue='strategy',
            height=4,
            aspect=.9,
            kind="line",
            palette={'myopic': 'skyblue', 'take_first_loss': 'orangered',
                     'random': 'springgreen'}
        )
        for ax in g3.axes.flat:
            labels = ax.get_xticklabels()  # get x labels
            ax.set_xticks(ticks=[1, 2, 3, 4, 5, 6, 7, 8])  # set new labels
            ax.set_xticklabels(fontsize=10,
                               labels=[str(i) for i in range(1, 9)])
        # show figure in streamlit
        st.pyplot(g3)

        # ---metrics----
        st.markdown(
            "### Average final reward obtained per strategy + "
            "average reward obtained at each step per strategy")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Myopic",
                value=st.session_state.myopic_solutions[
                    st.session_state.myopic_solutions, ['step'] == 8][
                    'total_reward'].mean())
            avg_step_reward = st.session_state.myopic_solutions.pivot_table(
                index="network_id",
                columns="step",
                values="reward").mean(
                axis=0)
            avg_step_reward.columns = ['Avg reward']
            st.dataframe(avg_step_reward)

        with col2:
            st.metric(
                "Take Loss then Myopic",
                value=st.session_state.loss_solutions[
                    st.session_state.loss_solutions['step'] == 8][
                    'total_reward'].mean())
            avg_step_reward = st.session_state.loss_solutions.pivot_table(
                index="network_id",
                columns="step",
                values="reward").mean(
                axis=0)
            avg_step_reward.columns = ['Avg reward']
            st.dataframe(avg_step_reward)

        with col3:
            st.metric("Random", "TODO")
    else:
        st.info("Please generate networks first!")

# ------------------------------------------------------------------------------
#                            Visualize Networks
# ------------------------------------------------------------------------------
with st.expander("Try yourself to solve the network 😎"):
    if "networks" in st.session_state:
        net_id = st.session_state.net_id if "net_id" in st.session_state else 1

        # add starting node
        net_to_plot = networks[net_id - 1]
        net_to_plot['nodes'][net_to_plot['starting_node']][
            'starting_node'] = True
        # convert dict to string
        # set separators=(',', ':') to remove spaces
        networks_str = json.dumps(net_to_plot, separators=(',', ':'))
        network_component(timer=100, network_args=networks_str)

        with st.form("vizualization_form", clear_on_submit=False):
            st.write("### Visualize Networks Options")
            net_id = st.number_input(
                label="Insert the network id to visualize",
                min_value=1,
                max_value=len(st.session_state.networks),
                value=1,
                step=1)
            updated = st.form_submit_button("Update Network")
            if updated:
                st.info('Parameters submitted!')
                st.session_state.net_id = net_id

        # visualization parameters
        # col1, col2 = st.columns(2)
        # with col1:
        #     network_id = st.selectbox("Which network to try?",
        #                               ("Email", "Home phone",
        #                                "Mobile phone"))
        # with col2:
        #     strategies = st.multiselect(
        #         'Which strategy solution do you want to see?',
        #         ['Myopic', 'Loss'],
        #         ['Myopic'])
    else:
        network_component(100)
