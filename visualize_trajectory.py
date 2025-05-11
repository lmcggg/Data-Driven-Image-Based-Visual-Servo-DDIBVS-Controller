#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to visualize different target trajectories for the DDIBVS controller
without running the full simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
import time

def generate_trajectory(trajectory_type, duration=20.0, dt=0.01):
    """
    Generate a trajectory based on the specified type
    
    Parameters:
    -----------
    trajectory_type : str
        Type of trajectory ('circle', 'square', 'static')
    duration : float
        Duration of the trajectory in seconds
    dt : float
        Time step
        
    Returns:
    --------
    time_points : numpy.ndarray
        Time points
    positions : numpy.ndarray
        Positions at each time point (x, y)
    velocities : numpy.ndarray
        Velocities at each time point (vx, vy)
    """
    time_points = np.arange(0, duration, dt)
    num_points = len(time_points)
    
    positions = np.zeros((num_points, 2))
    velocities = np.zeros((num_points, 2))
    
    if trajectory_type == 'circle':
        # Parameters
        radius = 0.5
        frequency = 0.5  # Hz
        omega = 2 * np.pi * frequency
        
        # Compute positions and velocities
        for i, t in enumerate(time_points):
            positions[i, 0] = radius * np.cos(omega * t)
            positions[i, 1] = radius * np.sin(omega * t)
            velocities[i, 0] = -radius * omega * np.sin(omega * t)
            velocities[i, 1] = radius * omega * np.cos(omega * t)
            
    elif trajectory_type == 'square':
        # Parameters
        square_size = 1.0
        frequency = 0.2  # Hz
        period = 1.0 / frequency
        half_size = square_size / 2
        
        # Compute positions and velocities
        for i, t in enumerate(time_points):
            t_mod = t % period
            segment = int(4 * t_mod / period)
            t_segment = t_mod - segment * period / 4
            
            if segment == 0:  # Right edge (bottom to top)
                positions[i, 0] = half_size
                positions[i, 1] = -half_size + square_size * t_segment / (period/4)
                velocities[i, 0] = 0
                velocities[i, 1] = square_size / (period/4)
            elif segment == 1:  # Top edge (right to left)
                positions[i, 0] = half_size - square_size * t_segment / (period/4)
                positions[i, 1] = half_size
                velocities[i, 0] = -square_size / (period/4)
                velocities[i, 1] = 0
            elif segment == 2:  # Left edge (top to bottom)
                positions[i, 0] = -half_size
                positions[i, 1] = half_size - square_size * t_segment / (period/4)
                velocities[i, 0] = 0
                velocities[i, 1] = -square_size / (period/4)
            else:  # Bottom edge (left to right)
                positions[i, 0] = -half_size + square_size * t_segment / (period/4)
                positions[i, 1] = -half_size
                velocities[i, 0] = square_size / (period/4)
                velocities[i, 1] = 0
                
    else:  # Static
        # No movement
        pass
    
    return time_points, positions, velocities

def visualize_trajectory(trajectory_type):
    """
    Visualize a trajectory
    
    Parameters:
    -----------
    trajectory_type : str
        Type of trajectory ('circle', 'square', 'static')
    """
    time_points, positions, velocities = generate_trajectory(trajectory_type)
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot trajectory
    plt.subplot(2, 2, 1)
    plt.plot(positions[:, 0], positions[:, 1], 'b-')
    plt.scatter(positions[0, 0], positions[0, 1], c='g', s=100, label='Start')
    plt.scatter(positions[-1, 0], positions[-1, 1], c='r', s=100, label='End')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title(f'{trajectory_type.capitalize()} Trajectory')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    
    # Plot X and Y positions over time
    plt.subplot(2, 2, 2)
    plt.plot(time_points, positions[:, 0], 'b-', label='X Position')
    plt.plot(time_points, positions[:, 1], 'r-', label='Y Position')
    plt.xlabel('Time (s)')
    plt.ylabel('Position')
    plt.title('Position vs Time')
    plt.grid(True)
    plt.legend()
    
    # Plot X and Y velocities over time
    plt.subplot(2, 2, 3)
    plt.plot(time_points, velocities[:, 0], 'b-', label='X Velocity')
    plt.plot(time_points, velocities[:, 1], 'r-', label='Y Velocity')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity')
    plt.title('Velocity vs Time')
    plt.grid(True)
    plt.legend()
    
    # Plot velocity magnitude over time
    plt.subplot(2, 2, 4)
    speed = np.sqrt(velocities[:, 0]**2 + velocities[:, 1]**2)
    plt.plot(time_points, speed, 'g-')
    plt.xlabel('Time (s)')
    plt.ylabel('Speed')
    plt.title('Speed vs Time')
    plt.grid(True)
    
    plt.tight_layout()
    plt.suptitle(f'{trajectory_type.capitalize()} Trajectory Analysis', fontsize=16)
    plt.subplots_adjust(top=0.9)
    plt.show()

def main():
    """
    Main function to visualize different trajectories
    """
    trajectory_types = ['circle', 'square', 'static']
    
    for traj_type in trajectory_types:
        print(f"Visualizing {traj_type} trajectory...")
        visualize_trajectory(traj_type)
        time.sleep(1)  # Pause between visualizations

if __name__ == "__main__":
    main() 