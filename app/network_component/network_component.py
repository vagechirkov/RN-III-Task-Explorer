import streamlit.components.v1 as components

BASE_URL = "https://631063bad8ec25de99f6946b-bmhultuvhx.chromatic.com/" \
           "iframe.html?"
STORY_ARGS = "id=utils-taskexplorer--default&viewMode=story"


def network_component(timer: int = 25):
    """Embeds a network component from Chromatic.

    Parameters
    ----------
    timer : int
        The time in ms for the component to wait before rendering.
    """
    url = BASE_URL + f"args=timer:{timer}&{STORY_ARGS}"
    components.iframe(url, height=700, width=800)

