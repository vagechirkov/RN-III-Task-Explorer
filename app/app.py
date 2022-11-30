import pandas as pd
import streamlit as st

from generate.generation import NetworkGenerator
from models.environment import Environment
from network_component.network_component import network_component
from plotting.plotting_solutions import plot_final_rewards, \
    plot_avg_reward_per_step
from solve.rule_based import RuleAgent
from utils.dict_input import dict_input
from utils.io import load_yaml

st.write("""
            # RN III Task Explorer
            This is an interactive application to explore stimuli and task 
            design for the Reward Networks III project. 
         """)

if "gen_env" in st.session_state:
    environment = load_yaml("app/default_environment.yml")
    st.session_state.gen_env = Environment(**environment)

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
            value=11,
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
            max_value=16,
            value=8,
            step=1)
        gen_params['n_levels'] = st.number_input(
            label='How many levels in the network?',
            min_value=1,
            max_value=4,
            value=4,
            step=1)

        with st.expander("More Parameters"):
            changed_env = dict_input("Change more environment setting",
                                     st.session_state.gen_env.dict())

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

            st.session_state.gen_env = Environment(**changed_env)

            # Network_Generator class
            net_generator = NetworkGenerator(st.session_state.gen_env)
            networks = net_generator.generate(gen_params['n_networks'])
            networks = [n.dict() for n in networks]

            # check if the size of the networks is valid
            if len(networks) != gen_params['n_networks']:
                st.error(
                    f"The number of generated networks {len(networks)} is not "
                    f" equal to the number of networks requested "
                    f"{gen_params['n_networks']}")

            # update starting nodes
            for i in range(len(networks)):
                networks[i]['nodes'][networks[i]['starting_node']][
                    'starting_node'] = True
            print(f"N nets: {len(networks)}")
            st.session_state.networks = networks
            st.session_state.net_id = 1
            st.success("Networks generated!")
            if to_download_data:
                data = net_generator.save_as_json()

            # Solve networks with strategies
            myopic_agent = RuleAgent(networks, "myopic", gen_params)

            st.session_state.myopic_solutions = myopic_agent.solve()
            loss_agent = RuleAgent(networks, "take_first_loss", gen_params)
            st.session_state.loss_solutions = loss_agent.solve()
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
            data=myopic_agent.save_solutions_frontend(),
            file_name='solutions_myopic.json')
        st.download_button(
            label="Download solutions (loss)",
            data=loss_agent.save_solutions_frontend(),
            file_name='solutions_loss.json')

# ------------------------------------------------------------------------------
#                                   Compare
# ------------------------------------------------------------------------------
with st.expander("Compare strategies 🤖"):
    if "networks" in st.session_state:
        # create solution data file with all strategies in one file
        m_df = st.session_state.myopic_solutions
        l_df = st.session_state.loss_solutions
        strategy_data = pd.concat([m_df, l_df], ignore_index=True)
        strategy_data_final = strategy_data[strategy_data['step'] == 8]

        col1, col2 = st.columns([1, 2])
        g = plot_final_rewards(strategy_data_final)
        g3 = plot_avg_reward_per_step(strategy_data)
        with col1:
            st.pyplot(g)
        with col2:
            st.pyplot(g3)
    else:
        st.info("Please generate networks first!")

# ------------------------------------------------------------------------------
#                            Visualize Networks
# ------------------------------------------------------------------------------
with st.expander("Try yourself to solve the network 😎"):
    if "networks" in st.session_state:
        nets = st.session_state.networks
        net_id = st.session_state.net_id
        print(net_id)
        print(len(nets))

        with st.form("vizualization_form", clear_on_submit=False):
            col1, col2, _ = st.columns(3)
            with col1:
                prev_net = st.form_submit_button("Show previous network")
            with col2:
                next_net = st.form_submit_button("Show next network")

            if next_net:
                if st.session_state.net_id < len(nets):
                    net_id += 1
                    st.session_state.net_id = net_id
            if prev_net:
                if st.session_state.net_id > 0:
                    net_id -= 1
                    st.session_state.net_id = net_id

            network_component(
                timer=60,
                network=st.session_state.networks[st.session_state.net_id - 1])

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
        network_component(60)

with st.expander("Show solution dataframes 📊"):
    if "networks" in st.session_state:
        # ---metrics----
        st.markdown(
            "#### Average final reward obtained per strategy + "
            "average reward obtained at each step per strategy")
        col1, col2 = st.columns(2)
        with col1:
            avg_val1 = m_df[m_df['step'] == 8]['total_reward'].mean().round(0)
            st.metric(
                "Myopic",
                value=int(avg_val1))
            m_avg_step_reward = m_df.pivot_table(
                index="network_id",
                columns="step",
                values="reward").mean(axis=0)
            m_avg_step_reward.columns = ['Avg reward']
            st.dataframe(m_avg_step_reward)

        with col2:
            avg_val2 = l_df[l_df['step'] == 8]['total_reward'].mean().round(0)
            st.metric(
                "Take Loss then Myopic",
                value=int(avg_val2))
            l_avg_step_reward = l_df.pivot_table(
                index="network_id",
                columns="step",
                values="reward").mean(axis=0)
            l_avg_step_reward.columns = ['Avg reward']
            st.dataframe(l_avg_step_reward)
        #
        # with col3:
        #     st.metric("Random", "TODO")

        st.write("## Myopic solutions")
        st.dataframe(st.session_state.myopic_solutions)
        st.write("## Take first loss solutions")
        st.dataframe(st.session_state.loss_solutions)
    else:
        st.info("Please generate networks first!")
