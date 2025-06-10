from .testbed import Testbed
import argparse
import yaml
from types import SimpleNamespace

# Function to "recursively" convert a dictionary into a SimpleNamespace objects
def dict_to_namespace(d):
    if isinstance(d, dict):
        # Convert all "nested" dictionaries to SimpleNamespace
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    return d

def get_configs(file_path):
    """
    Parses command-line arguments to load a configuration file and converts it into a namespace object.

    Returns:
    SimpleNamespace: Namespace object contatining configuration values from YAML file.
    """

    # Load the YAML file and convert it to an object
    try:
        with open(file_path, 'r') as file:
            config_dict = yaml.safe_load(file)
            config = dict_to_namespace(config_dict)
            return config
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")

    return 

# Main function to execute the training process
def main():
    # Define the argument parser
    parser = argparse.ArgumentParser(description="LiePose with Picard iteration to accelerate sampling on SO(3). This repo provides both the training and inference processes.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "test", "vis", "vis_video"],
                    help="Choose the mode to run: train, test, vis, vis_video.")
    
    # Parse the command-line arguments
    args = parser.parse_args()

    # Parse the arguments and load configuration
    config = get_configs(args.config)

    testbed = Testbed(config)
    if args.mode == "train":
        testbed.train()
    if args.mode == "test":
        testbed.test()
    if args.mode == "vis":
        testbed.visualize()
    if args.mode == "vis_video":
        testbed.visualize_video()


if __name__ == "__main__":
    main()

