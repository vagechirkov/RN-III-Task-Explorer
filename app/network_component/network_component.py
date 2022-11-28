import streamlit.components.v1 as components

BASE_URL = "https://631063bad8ec25de99f6946b-bnxaounhwz.chromatic.com/" \
           "iframe.html?"
STORY_COMPONENT = "id=utils-taskexplorer2--default&viewMode=story"


def network_component(timer: int = 25, network_args: str = None):
    """Embeds a network component from Chromatic.

    Parameters
    ----------
    timer : int
        The time in ms for the component to wait before rendering.
    """
    # args=timer:{timer}
    url = f"{BASE_URL}args=timer:{timer}" \
          f"&custom_args={network_args}" \
          f"&{STORY_COMPONENT}"
    components.iframe(url, height=700, width=800)
