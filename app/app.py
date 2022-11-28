import json

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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
    if networks is not None:
        # create solution data file with all strategies in one file 
        strategy_data = pd.concat([Myopic_agent.df,Loss_agent.df],ignore_index=True)
        strategy_data_final=strategy_data[strategy_data['step']==8]

        g = sns.displot(data=strategy_data_final,x="total_reward", hue="strategy", kind="hist")
        g.set(xlabel='Final total reward',ylabel='Count',title=f'Strategy final total reward comparison')
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
                        palette={'myopic':'skyblue','take_first_loss':'orangered','random':'springgreen'}
                        )
        for ax in g3.axes.flat:
            labels = ax.get_xticklabels() # get x labels
            ax.set_xticks(ticks=[1,2,3,4,5,6,7,8]) # set new labels
            ax.set_xticklabels(fontsize=10,labels=[str(i) for i in range(1,9)])
        # show figure in streamlit
        st.pyplot(g3)

        # ---metrics----
        st.markdown("### Average final reward obtained per strategy + average reward obtained at each step per strategy")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Myopic", 
                    value=Myopic_agent.df[Myopic_agent.df['step']==8]['total_reward'].mean())
            avg_step_reward = Myopic_agent.df.pivot_table(index="network_id", columns="step", values="reward").mean(axis=0)
            avg_step_reward.columns = ['Avg reward']
            st.dataframe(avg_step_reward)
        
        with col2:
            st.metric("Take Loss then Myopic",
                        value=Loss_agent.df[Loss_agent.df['step']==8]['total_reward'].mean())
            avg_step_reward = Loss_agent.df.pivot_table(index="network_id", columns="step", values="reward").mean(axis=0)
            avg_step_reward.columns = ['Avg reward']
            st.dataframe(avg_step_reward)

        with col3:
            st.metric("Random", "TODO")

