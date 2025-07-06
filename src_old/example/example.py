from src_old.core import *
from src_old.config.config import CrossRingConfig
import numpy as np


def main():
    """
    This is an example script to run a simple CrossRing simulation.
    """
    # Define the model type
    model_type = "REQ_RSP"

    # Define the path to the configuration file
    config = CrossRingConfig()

    # Define the topology
    topo_type = "5x4"
    config.TOPO_TYPE = topo_type

    # Initialize the simulation model
    sim: BaseModel = eval(f"{model_type}_model")(
        model_type=model_type,
        config=config,
        topo_type=topo_type,
        traffic_file_path="",
        traffic_config=[["test_data.txt"]],
        result_save_path="../Result/",
        results_fig_save_path="",
        plot_flow_fig=0,
        plot_RN_BW_fig=0,
        plot_link_state=0,
        verbose=1,
    )

    sim.initial()
    # Set simulation parameters
    sim.end_time = 1000
    sim.print_interval = 1000

    # Run the simulation
    sim.run()


if __name__ == "__main__":
    main()
