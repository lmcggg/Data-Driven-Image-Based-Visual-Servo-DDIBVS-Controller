import numpy as np
import matplotlib.pyplot as plt
import time
import math

class DDIBVSController:
    """
    Data-Driven Image-Based Visual Servo (DDIBVS) Controller
    
    This class implements the DDIBVS control method as described in the paper,
    which uses data-driven learning to estimate the robot and image Jacobians
    and a neural dynamic controller (NDC) to generate control signals.
    """
    
    def __init__(self, n_joints=2, learning_rate_mu=0.2, learning_rate_beta=0.2, 
                 convergence_rate_delta=5.0, update_rate_sigma=0.05, 
                 joint_limit_kappa=0.8):
        """
        Initialize the DDIBVS controller
        
        Parameters:
        -----------
        n_joints : int
            Number of robot joints (default: 2 for pan-tilt system)
        learning_rate_mu : float
            Learning rate for robot Jacobian estimation (μ)
        learning_rate_beta : float
            Learning rate for image Jacobian estimation (β)
        convergence_rate_delta : float
            Convergence rate for feature error (δ)
        update_rate_sigma : float
            Update rate for auxiliary parameter Γ (σ)
        joint_limit_kappa : float
            Coefficient for joint angle limit tightening (κ)
        """
        self.n_joints = n_joints
        
        # Learning parameters
        self.mu = learning_rate_mu
        self.beta = learning_rate_beta
        self.delta = convergence_rate_delta
        self.sigma = update_rate_sigma
        self.kappa = joint_limit_kappa
        
        # Initialize estimated Jacobians
        self.J_m_hat = np.zeros((6, n_joints), dtype=np.float64)  # Robot Jacobian estimate (6x2 for pan-tilt)
        # Initialize image Jacobian with a reasonable guess
        self.J_im_hat = np.array([
            [1.0, 0.0],  # X image coordinate affected by pan
            [0.0, 1.0]   # Y image coordinate affected by tilt
        ], dtype=np.float64)  # Image Jacobian estimate for pan-tilt system
        
        # Initialize auxiliary parameter Γ (2x2 matrix)
        self.Gamma = np.eye(2, dtype=np.float64)
        
        # Joint limits
        self.phi_min = np.zeros(n_joints, dtype=np.float64)  # Joint angle lower limits
        self.phi_max = np.ones(n_joints, dtype=np.float64) * np.pi  # Joint angle upper limits
        self.phi_dot_min = -np.ones(n_joints, dtype=np.float64)  # Joint velocity lower limits
        self.phi_dot_max = np.ones(n_joints, dtype=np.float64)  # Joint velocity upper limits
        
        # Current state
        self.phi = np.zeros(n_joints, dtype=np.float64)  # Current joint angles
        self.phi_dot = np.zeros(n_joints, dtype=np.float64)  # Current joint velocities
        self.x = np.zeros(2, dtype=np.float64)  # Current feature position
        self.x_dot = np.zeros(2, dtype=np.float64)  # Current feature velocity
        self.x_d = np.zeros(2, dtype=np.float64)  # Desired feature position
        self.x_d_dot = np.zeros(2, dtype=np.float64)  # Desired feature velocity
        
        # For visualization
        self.history = {
            'time': [],
            'phi': [],
            'phi_dot': [],
            'x': [],
            'x_d': [],
            'error': [],
            'J_m_hat_norm': [],
            'J_im_hat_norm': [],
            'Gamma_norm': []
        }
    
    def update_robot_jacobian(self, phi_dot, r_dot):
        """
        Update the estimated robot Jacobian using formula (12)
        
        Parameters:
        -----------
        phi_dot : numpy.ndarray
            Joint velocities
        r_dot : numpy.ndarray
            End-effector velocity (6D)
        """
        # Formula (12): dJ_m_hat/dt = -μ(J_m_hat * phi_dot - r_dot) * phi_dot^T
        error = np.dot(self.J_m_hat, phi_dot) - r_dot
        self.J_m_hat -= self.mu * np.outer(error, phi_dot)
    
    def update_image_jacobian(self, phi_dot, x_dot):
        """
        Update the estimated image Jacobian using formula (13)
        
        Parameters:
        -----------
        phi_dot : numpy.ndarray
            Joint velocities
        x_dot : numpy.ndarray
            Feature velocity in image plane
        """
        # Formula (13): dJ_im_hat/dt = -β(J_im_hat * phi_dot - x_dot) * phi_dot^T
        error = np.dot(self.J_im_hat, phi_dot) - x_dot
        self.J_im_hat -= self.beta * np.outer(error, phi_dot)
    
    def update_auxiliary_parameter(self):
        """
        Update the auxiliary parameter Γ using formula (22b)
        """
        # Formula (22b): dΓ/dt = σ * J_im_hat * J_im_hat^T * (I - J_im_hat * J_im_hat^T * Γ)
        J_im_J_im_T = np.dot(self.J_im_hat, self.J_im_hat.T)
        self.Gamma += self.sigma * np.dot(J_im_J_im_T, np.eye(2) - np.dot(J_im_J_im_T, self.Gamma))
    
    def compute_joint_limits(self):
        """
        Compute dynamic joint limits using formula (15)
        
        Returns:
        --------
        omega_min : numpy.ndarray
            Lower limits for joint velocities
        omega_max : numpy.ndarray
            Upper limits for joint velocities
        """
        # Formula (15): ω^- = max{φ^-, κ(φ^- - φ)}, ω^+ = min{φ^+, κ(φ^+ - φ)}
        omega_min = np.maximum(
            self.phi_dot_min,
            self.kappa * (self.phi_min - self.phi)
        )
        omega_max = np.minimum(
            self.phi_dot_max,
            self.kappa * (self.phi_max - self.phi)
        )
        return omega_min, omega_max
    
    def project_to_limits(self, a):
        """
        Project a vector to the joint velocity limits using formula (16)
        
        Parameters:
        -----------
        a : numpy.ndarray
            Vector to be projected
        
        Returns:
        --------
        projected : numpy.ndarray
            Projected vector
        """
        omega_min, omega_max = self.compute_joint_limits()
        
        # Formula (16): F_Ω(a) = {ω^-, if a < ω^-; a, if ω^- ≤ a ≤ ω^+; ω^+, if a > ω^+}
        projected = np.zeros_like(a)
        for i in range(len(a)):
            if a[i] < omega_min[i]:
                projected[i] = omega_min[i]
            elif a[i] > omega_max[i]:
                projected[i] = omega_max[i]
            else:
                projected[i] = a[i]
        
        return projected
    
    def compute_control_signal(self):
        """
        Compute the control signal using formula (22a)
        
        Returns:
        --------
        phi_dot : numpy.ndarray
            Joint velocity control signal
        """
        # Formula (22a): φ_dot = F_Ω(J_im_hat^T * Γ * (-δ(x - x_d) + x_d_dot))
        error = self.x - self.x_d
        term1 = -self.delta * error + self.x_d_dot
        term2 = np.dot(self.Gamma, term1)
        term3 = np.dot(self.J_im_hat.T, term2)
        phi_dot = self.project_to_limits(term3)
        
        return phi_dot
    
    def update_state(self, dt, phi_dot, r_dot, x_dot):
        """
        Update the controller state
        
        Parameters:
        -----------
        dt : float
            Time step
        phi_dot : numpy.ndarray
            Joint velocities
        r_dot : numpy.ndarray
            End-effector velocity
        x_dot : numpy.ndarray
            Feature velocity in image plane
        """
        # Update joint angles
        self.phi += phi_dot * dt
        self.phi_dot = phi_dot
        
        # Update Jacobians
        self.update_robot_jacobian(phi_dot, r_dot)
        self.update_image_jacobian(phi_dot, x_dot)
        
        # Update auxiliary parameter
        self.update_auxiliary_parameter()
        
        # Update feature position (simplified)
        self.x += x_dot * dt
    
    def set_desired_feature(self, x_d, x_d_dot=None):
        """
        Set the desired feature position and velocity
        
        Parameters:
        -----------
        x_d : numpy.ndarray
            Desired feature position
        x_d_dot : numpy.ndarray, optional
            Desired feature velocity
        """
        self.x_d = x_d
        if x_d_dot is not None:
            self.x_d_dot = x_d_dot
    
    def set_joint_limits(self, phi_min, phi_max, phi_dot_min, phi_dot_max):
        """
        Set joint limits
        
        Parameters:
        -----------
        phi_min : numpy.ndarray
            Joint angle lower limits
        phi_max : numpy.ndarray
            Joint angle upper limits
        phi_dot_min : numpy.ndarray
            Joint velocity lower limits
        phi_dot_max : numpy.ndarray
            Joint velocity upper limits
        """
        self.phi_min = phi_min
        self.phi_max = phi_max
        self.phi_dot_min = phi_dot_min
        self.phi_dot_max = phi_dot_max
    
    def record_history(self, t):
        """
        Record the current state for visualization
        
        Parameters:
        -----------
        t : float
            Current time
        """
        self.history['time'].append(t)
        self.history['phi'].append(self.phi.copy())
        self.history['phi_dot'].append(self.phi_dot.copy())
        self.history['x'].append(self.x.copy())
        self.history['x_d'].append(self.x_d.copy())
        self.history['error'].append(np.linalg.norm(self.x - self.x_d))
        self.history['J_m_hat_norm'].append(np.linalg.norm(self.J_m_hat))
        self.history['J_im_hat_norm'].append(np.linalg.norm(self.J_im_hat))
        self.history['Gamma_norm'].append(np.linalg.norm(self.Gamma))
    
    def plot_results(self):
        """
        Plot the results of the DDIBVS control
        """
        # Create a figure with 3x3 subplots
        plt.figure(figsize=(20, 15))
        
        # Plot joint angles
        plt.subplot(3, 3, 1)
        joint_labels = ['Pan (horizontal)', 'Tilt (vertical)']  # Labels for pan-tilt system
        for i in range(self.n_joints):
            plt.plot(self.history['time'], [phi[i] for phi in self.history['phi']], 
                    label=joint_labels[i])
        plt.xlabel('Time (s)')
        plt.ylabel('Joint Angles (rad)')
        plt.title('Pan-Tilt Joint Angles')
        plt.grid(True)
        plt.legend()
        
        # Plot joint velocities
        plt.subplot(3, 3, 2)
        for i in range(self.n_joints):
            plt.plot(self.history['time'], [phi_dot[i] for phi_dot in self.history['phi_dot']], 
                    label=joint_labels[i])
        plt.xlabel('Time (s)')
        plt.ylabel('Joint Velocities (rad/s)')
        plt.title('Pan-Tilt Joint Velocities')
        plt.grid(True)
        plt.legend()
        
        # Plot feature position
        plt.subplot(3, 3, 3)
        plt.plot(self.history['time'], [x[0] for x in self.history['x']], 'b-', label='x')
        plt.plot(self.history['time'], [x[1] for x in self.history['x']], 'g-', label='y')
        plt.plot(self.history['time'], [x_d[0] for x_d in self.history['x_d']], 'b--', label='x_d')
        plt.plot(self.history['time'], [x_d[1] for x_d in self.history['x_d']], 'g--', label='y_d')
        plt.xlabel('Time (s)')
        plt.ylabel('Feature Position')
        plt.title('Feature Position')
        plt.grid(True)
        plt.legend()
        
        # Plot feature trajectory in XY plane
        plt.subplot(3, 3, 4)
        plt.plot([x[0] for x in self.history['x']], [x[1] for x in self.history['x']], 'r-', label='Actual')
        plt.plot([x_d[0] for x_d in self.history['x_d']], [x_d[1] for x_d in self.history['x_d']], 'b--', label='Desired')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Feature Trajectory')
        plt.grid(True)
        plt.legend()
        plt.axis('equal')
        
        # Plot feature error
        plt.subplot(3, 3, 5)
        plt.plot(self.history['time'], self.history['error'], 'r-')
        plt.xlabel('Time (s)')
        plt.ylabel('Feature Error')
        plt.title('Feature Error')
        plt.grid(True)
        
        # Plot individual feature errors
        plt.subplot(3, 3, 6)
        x_errors = [x[0] - x_d[0] for x, x_d in zip(self.history['x'], self.history['x_d'])]
        y_errors = [x[1] - x_d[1] for x, x_d in zip(self.history['x'], self.history['x_d'])]
        plt.plot(self.history['time'], x_errors, 'b-', label='X Error')
        plt.plot(self.history['time'], y_errors, 'g-', label='Y Error')
        plt.xlabel('Time (s)')
        plt.ylabel('Position Error')
        plt.title('Individual Feature Errors')
        plt.grid(True)
        plt.legend()
        
        # Plot Jacobian norms
        plt.subplot(3, 3, 7)
        plt.plot(self.history['time'], self.history['J_m_hat_norm'], 'b-', label='Robot Jacobian')
        plt.plot(self.history['time'], self.history['J_im_hat_norm'], 'g-', label='Image Jacobian')
        plt.xlabel('Time (s)')
        plt.ylabel('Norm')
        plt.title('Jacobian Norms')
        plt.grid(True)
        plt.legend()
        
        # Plot Gamma norm
        plt.subplot(3, 3, 8)
        plt.plot(self.history['time'], self.history['Gamma_norm'], 'r-')
        plt.xlabel('Time (s)')
        plt.ylabel('Norm')
        plt.title('Gamma Norm')
        plt.grid(True)
        
        # Plot control effort (sum of absolute joint velocities)
        plt.subplot(3, 3, 9)
        control_effort = [np.sum(np.abs(phi_dot)) for phi_dot in self.history['phi_dot']]
        plt.plot(self.history['time'], control_effort, 'k-')
        plt.xlabel('Time (s)')
        plt.ylabel('Control Effort')
        plt.title('Total Control Effort')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()


class DDIBVSSimulation:
    """
    Simulation environment for testing the DDIBVS controller
    """
    
    def __init__(self, controller):
        """
        Initialize the simulation
        
        Parameters:
        -----------
        controller : DDIBVSController
            The DDIBVS controller to test
        """
        self.controller = controller
        self.dt = 0.01  # Time step
        self.t = 0.0  # Current time
        
        # Simulation parameters
        self.simulation_time = 20.0  # Total simulation time
        self.noise_level = 0.005  # Reduced noise level for better stability
        
        # Robot parameters (pan-tilt camera system)
        # For a pan-tilt system:
        # - Joint 1 (pan) rotates the camera about the vertical axis
        # - Joint 2 (tilt) rotates the camera about the horizontal axis
        self.true_J_m = np.array([
            [0.0, 0.0],  # X position (not directly affected)
            [0.0, 0.0],  # Y position (not directly affected)
            [0.0, 0.0],  # Z position (not directly affected)
            [0.0, 1.0],  # Rotation about X axis (tilt)
            [1.0, 0.0],  # Rotation about Y axis (pan)
            [0.0, 0.0]   # Rotation about Z axis (not used)
        ], dtype=np.float64)  # True robot Jacobian for pan-tilt system
        
        # Image Jacobian for a pan-tilt camera system
        # - Pan joint affects horizontal image coordinate (x)
        # - Tilt joint affects vertical image coordinate (y)
        self.true_J_im = np.array([
            [1.0, 0.0],  # X image coordinate affected by pan
            [0.0, 1.0]   # Y image coordinate affected by tilt
        ], dtype=np.float64)  # True image Jacobian for pan-tilt camera
        
        # Target motion parameters
        self.target_motion = 'circle'  # 'circle', 'square', or 'static'
        self.circle_radius = 0.5
        self.circle_frequency = 0.5  # Hz
        self.square_size = 1.0
        self.square_frequency = 0.2  # Hz
    
    def add_noise(self, value):
        """
        Add noise to a value
        
        Parameters:
        -----------
        value : numpy.ndarray
            Value to add noise to
        
        Returns:
        --------
        noisy_value : numpy.ndarray
            Value with noise added
        """
        return value + self.noise_level * np.random.randn(*value.shape)
    
    def compute_true_velocities(self, phi_dot):
        """
        Compute true end-effector and feature velocities
        
        Parameters:
        -----------
        phi_dot : numpy.ndarray
            Joint velocities
        
        Returns:
        --------
        r_dot : numpy.ndarray
            True end-effector velocity
        x_dot : numpy.ndarray
            True feature velocity
        """
        # Compute true end-effector velocity
        r_dot = np.dot(self.true_J_m, phi_dot)
        
        # Compute true feature velocity
        x_dot = np.dot(self.true_J_im, phi_dot)
        
        # Add noise to measurements
        r_dot_noisy = self.add_noise(r_dot)
        x_dot_noisy = self.add_noise(x_dot)
        
        return r_dot_noisy, x_dot_noisy
    
    def update_target_position(self):
        """
        Update the target position based on the selected motion pattern
        
        Returns:
        --------
        x_d : numpy.ndarray
            New desired feature position
        x_d_dot : numpy.ndarray
            New desired feature velocity
        """
        if self.target_motion == 'circle':
            # Circular motion
            omega = 2 * np.pi * self.circle_frequency
            x_d = self.circle_radius * np.array([
                np.cos(omega * self.t),
                np.sin(omega * self.t)
            ])
            x_d_dot = self.circle_radius * omega * np.array([
                -np.sin(omega * self.t),
                np.cos(omega * self.t)
            ])
        elif self.target_motion == 'square':
            # Square motion
            period = 1.0 / self.square_frequency
            t_mod = self.t % period
            segment = int(4 * t_mod / period)
            t_segment = t_mod - segment * period / 4
            
            if segment == 0:  # Right edge
                x_d = np.array([self.square_size/2, -self.square_size/2 + self.square_size * t_segment / (period/4)])
                x_d_dot = np.array([0, self.square_size / (period/4)])
            elif segment == 1:  # Top edge
                x_d = np.array([self.square_size/2 - self.square_size * t_segment / (period/4), self.square_size/2])
                x_d_dot = np.array([-self.square_size / (period/4), 0])
            elif segment == 2:  # Left edge
                x_d = np.array([-self.square_size/2, self.square_size/2 - self.square_size * t_segment / (period/4)])
                x_d_dot = np.array([0, -self.square_size / (period/4)])
            else:  # Bottom edge
                x_d = np.array([-self.square_size/2 + self.square_size * t_segment / (period/4), -self.square_size/2])
                x_d_dot = np.array([self.square_size / (period/4), 0])
        else:  # Static target
            x_d = np.array([0.0, 0.0])
            x_d_dot = np.array([0.0, 0.0])
        
        return x_d, x_d_dot
    
    def run(self):
        """
        Run the simulation
        """
        # Set initial desired feature position
        x_d, x_d_dot = self.update_target_position()
        self.controller.set_desired_feature(x_d, x_d_dot)
        
        # Set joint limits with larger velocity range
        self.controller.set_joint_limits(
            np.ones(self.controller.n_joints) * -np.pi,   # phi_min
            np.ones(self.controller.n_joints) * np.pi,    # phi_max
            np.ones(self.controller.n_joints) * -2.0,     # velocity range
            np.ones(self.controller.n_joints) * 2.0       # velocity range
        )
        
        # Main simulation loop
        while self.t < self.simulation_time:
            # Update target position
            x_d, x_d_dot = self.update_target_position()
            self.controller.set_desired_feature(x_d, x_d_dot)
            
            # Compute control signal
            phi_dot = self.controller.compute_control_signal()
            
            # Compute true velocities
            r_dot, x_dot = self.compute_true_velocities(phi_dot)
            
            # Update controller state
            self.controller.update_state(self.dt, phi_dot, r_dot, x_dot)
            
            # Record history
            self.controller.record_history(self.t)
            
            # Update time
            self.t += self.dt
        
        # Plot results
        self.controller.plot_results()


def main():
    """
    Main function to run the DDIBVS simulation
    """
    # Create controller with optimized parameters
    controller = DDIBVSController(
        n_joints=2,              # Changed from 6 to 2 joints
        learning_rate_mu=0.2,    
        learning_rate_beta=0.2,  
        convergence_rate_delta=5.0,
        update_rate_sigma=0.05,  
        joint_limit_kappa=0.8    
    )
    
    # Create simulation
    simulation = DDIBVSSimulation(controller)
    
    # Set joint limits with larger velocity range
    simulation.controller.set_joint_limits(
        np.ones(simulation.controller.n_joints) * -np.pi,   # phi_min
        np.ones(simulation.controller.n_joints) * np.pi,    # phi_max
        np.ones(simulation.controller.n_joints) * -2.0,     # velocity range
        np.ones(simulation.controller.n_joints) * 2.0       # velocity range
    )
    
    # Run simulation
    simulation.run()


if __name__ == "__main__":
    main()
