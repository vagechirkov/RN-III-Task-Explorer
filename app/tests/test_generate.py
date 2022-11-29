from utils.io import load_yaml
from generate.generation import NetworkGenerator
from models.environment import Environment


def test_network_generator():
    environment_file = "tests/test_environment.yml"
    environment = load_yaml(environment_file)
    environment = Environment(**environment)
    network_generator = NetworkGenerator(environment)
    network_generator.generate(100)
