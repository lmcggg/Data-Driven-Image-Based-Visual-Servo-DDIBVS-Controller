# Data-Driven Image-Based Visual Servo (DDIBVS) Controller

This repository contains a Python implementation of a Data-Driven Image-Based Visual Servo (DDIBVS) controller. The DDIBVS method uses data-driven learning to estimate robot and image Jacobians and implements a neural dynamic controller (NDC) to generate control signals for visual servoing tasks.

## Description

The implementation simulates a pan-tilt camera system that tracks moving targets using visual servo control. The controller learns and adapts the system model online without requiring precise calibration.

Key features:
- Online estimation of robot Jacobian
- Online estimation of image Jacobian
- Neural dynamic controller with constraint handling
- Support for different target motion patterns (circle, square, static)
- Comprehensive visualization of controller performance

## Requirements

```
numpy
matplotlib
```

## Usage

To run the DDIBVS simulation:

```python
python ddibvs.py
```

The simulation will run for 20 seconds (configurable) and then display a set of plots showing the performance of the controller:
- Joint angles and velocities
- Feature positions and trajectories
- Tracking errors
- Jacobian norms
- Control effort

## Configuration

You can modify the following parameters in the code:

### Controller Parameters
```python
controller = DDIBVSController(
    n_joints=2,                   # Number of joints
    learning_rate_mu=0.2,         # Learning rate for robot Jacobian 
    learning_rate_beta=0.2,       # Learning rate for image Jacobian
    convergence_rate_delta=5.0,   # Convergence rate for feature error
    update_rate_sigma=0.05,       # Update rate for auxiliary parameter
    joint_limit_kappa=0.8         # Coefficient for joint angle limit
)
```

### Simulation Parameters
```python
# Target motion type
simulation.target_motion = 'circle'  # Options: 'circle', 'square', 'static'

# Circle trajectory parameters
simulation.circle_radius = 0.5
simulation.circle_frequency = 0.5  # Hz

# Square trajectory parameters
simulation.square_size = 1.0
simulation.square_frequency = 0.2  # Hz

# Simulation duration
simulation.simulation_time = 20.0  # seconds

# Measurement noise
simulation.noise_level = 0.005
```

## How It Works

The DDIBVS controller consists of three main components:

1. **Jacobian Estimation**: The controller continuously updates its estimate of the robot and image Jacobians based on observed motion.

2. **Neural Dynamic Controller**: Generates control signals that drive the feature error to zero while respecting joint constraints.

3. **Motion Planning**: Handles different target motion patterns for tracking.

The simulation environment provides a testing ground for the controller, simulating sensor measurements and robot dynamics.

## Results

After running the simulation, you'll see nine plots showing different aspects of the controller's performance:
- Pan-tilt joint angles
- Joint velocities
- Feature position tracking
- Feature trajectory in XY plane
- Feature tracking error
- Individual X and Y errors
- Jacobian norms
- Auxiliary parameter (Gamma) norm
- Total control effort

## References

This implementation is based on research in data-driven visual servoing and neural dynamic control approaches for robot control. 