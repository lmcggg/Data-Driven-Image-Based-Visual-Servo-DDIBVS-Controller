#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example script for running the Data-Driven Image-Based Visual Servo (DDIBVS) controller
with custom parameters and target motion patterns.
"""

import numpy as np
from ddibvs import DDIBVSController, DDIBVSSimulation


def run_custom_simulation(target_motion='circle', simulation_time=20.0):
    """
    Run a custom DDIBVS simulation with specified parameters
    
    Parameters:
    -----------
    target_motion : str
        Type of target motion ('circle', 'square', 'static')
    simulation_time : float
        Duration of the simulation in seconds
    """
    # Create controller with custom parameters
    controller = DDIBVSController(
        n_joints=2,                # Number of joints (pan-tilt system)
        learning_rate_mu=0.2,      # Learning rate for robot Jacobian
        learning_rate_beta=0.2,    # Learning rate for image Jacobian
        convergence_rate_delta=5.0, # Convergence rate for error
        update_rate_sigma=0.05,    # Update rate for auxiliary parameter
        joint_limit_kappa=0.8      # Joint limit coefficient
    )
    
    # Create simulation with the controller
    simulation = DDIBVSSimulation(controller)
    
    # Customize simulation parameters
    simulation.simulation_time = simulation_time
    simulation.target_motion = target_motion
    
    # Customize target motion parameters based on motion type
    if target_motion == 'circle':
        simulation.circle_radius = 0.5       # Radius of circular trajectory
        simulation.circle_frequency = 0.5    # Angular frequency in Hz
    elif target_motion == 'square':
        simulation.square_size = 1.0         # Size of square trajectory
        simulation.square_frequency = 0.2    # Frequency in Hz
    
    # Set joint limits (in radians)
    simulation.controller.set_joint_limits(
        phi_min=np.ones(controller.n_joints) * -np.pi,   # min joint angles
        phi_max=np.ones(controller.n_joints) * np.pi,    # max joint angles
        phi_dot_min=np.ones(controller.n_joints) * -2.0, # min velocities
        phi_dot_max=np.ones(controller.n_joints) * 2.0   # max velocities
    )
    
    # Run the simulation
    print(f"Running DDIBVS simulation with {target_motion} target motion for {simulation_time} seconds")
    simulation.run()


if __name__ == "__main__":
    # Example 1: Circle trajectory
    run_custom_simulation(target_motion='circle', simulation_time=20.0)
    
    # Example 2: Square trajectory
    # run_custom_simulation(target_motion='square', simulation_time=20.0)
    
    # Example 3: Static target
    # run_custom_simulation(target_motion='static', simulation_time=10.0) 