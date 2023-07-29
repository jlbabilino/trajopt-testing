from math import hypot, pi
from casadi import *

class SwerveDrive:
    """
        This class represents a swerve drive robot. It includes the physical properties necessary to
        accurately model the dynamics of the system. To understand this implementation, some definitions
        are required.
        
        When properties for each module are listed, they are always in the order of
        [front_left, front_right, rear_left, rear_right].

        Related to the 2D coordinate system of the field, the robot coordinate system is defined with its
        center placed on the center of the robot, and the x-axis points directly towards the front face
        of the robot. The y-axis points 90 degress counter-clockwise. The following diagram shows the
        coordinate system and some related dimensions:

         _________________________________________                    \ 
        /                                         \                   |
        |   00                              00    |                   |
        |   00  rear_left        front_left  00   |   \               |
        |   00               y                00  |   |               |
        |                    ^                    |   | wheelbase_y   |
        |                    |                    |   |               |
        |                    +----> x             |   /               | width
        |                                         |                   |
        |                                         |                   |
        |  00                                00   |                   |
        |   00  rear_right      front_right  00   |                   |
        |    00                              00   |                   |
        |                                         |                   |
        \_________________________________________/                   /
                             <- wheelbase_x ->
        <----------------- length ---------------->
        The nonrotating robot coordinate system is defined with the same center as the robot coordinate
        system, but it does not rotate and its axes always point in the same directions as the field
        coordinate system.
    """
    def __init__(self,
            wheelbase_x: float,
            wheelbase_y: float,
            length: float,
            width: float,
            mass: float,
            moi: float,
            omega_max: float,
            tau_max: float,
            wheel_radius: float):
        """
            Initializes a swerve drive model with given characteristics.

            Arguments:
                wheelbase_x  -- when facing side of robot, half the horizontal distance between modules
                wheelbase_y  -- when facing front of robot, half the horizontal distance between modules
                length       -- when facing side of robot, the horizontal distance across bumpers
                width        -- when facing front of robot, the horizontal distance across bumpers
                mass         -- mass of robot
                moi          -- moment of inertia of robot about axis of rotation (currently through
                                center of robot coordinate system)
                omega_max    -- maximum angular velocity of wheels (not related to direction controlling
                                motor; motor that controls speed)
                tau_max      -- maximum torque of wheels (similar to omega_max)
                wheel_radius -- radius of wheels
        """
        self.wheelbase_x = wheelbase_x
        self.wheelbase_y = wheelbase_y
        self.length = length
        self.width = width
        self.mass = mass
        self.moi = moi
        self.omega_max = omega_max
        self.tau_max = tau_max
        self.wheel_radius = wheel_radius
        self.force_max = tau_max / wheel_radius

    def solve_module_positions(self, theta: MX):
        """
            Calculates the position of the the modules relative to the nonrotating robot coordinate system
            at an instant. Constructs expressions for the position of each module in terms of the angle
            variable.

            Arguments:
                theta -- the variable representing the heading of the robot at one sample point.
            Returns:
                Returns a list containing the the four positions in the order specified in the class.
        """
        # module_angles are angles between x-axis of robot coordinate system and vector
        # pointing from center of robot to module.
        module_angle = atan2(self.wheelbase_y, self.wheelbase_x)
        module_angles = (module_angle, -module_angle, pi - module_angle, -(pi - module_angle))
        diagonal = hypot(self.wheelbase_x, self.wheelbase_y)
        module_positions = []
        for module_angle in module_angles:
            module_positions.append([diagonal*cos(module_angle+theta), diagonal*sin(module_angle+theta)])
        return module_positions

    def add_kinematics_constraint(self,
            solver: OptiSol,
            theta: MX,
            vx: MX,
            vy: MX,
            omega: MX,
            ax: MX,
            ay: MX,
            alpha: MX,
            N_total: int):
        """
            Arguments:
                solver -- the solver to add constraints to
                theta  -- the list of angles throughout the trajectory (size N+1)
                vx     -- the list of x-components of velocity (size N+1)
                vy     -- the list of y-components of velocity (size N+1)
                omega  -- the list of angular velocities (size N+1)
                ax     -- the list of x-components of acceleration (size N)
                ay     -- the list of y-components of acceleration (size N)
                alpha  -- the list of angular acceleration (size N)
                N      -- the total number of segments in the path
        """
        max_wheel_velocity_squared = self.omega_max * self.wheel_radius
        max_wheel_velocity_squared = max_wheel_velocity_squared * max_wheel_velocity_squared
        max_force_squared = self.force_max * self.force_max
        for k in range(N_total):
            module_positions = self.solve_module_positions(theta[k])

            for module_position in module_positions:
                # swerve kinematics: add the component of velocity of the robot
                # to the velocity to cause rotation. This gives the total velocity
                # of the module to both rotate and translate.
                m_vx = vx[k] + module_position[0] * omega[k]
                m_vy = vy[k] + module_position[1] * omega[k]
                solver.subject_to(m_vx * m_vx + m_vy * m_vy < max_wheel_velocity_squared)

            # Components of force caused by each wheel
            Fx = solver.variable(4)
            Fy = solver.variable(4)

            # Components of torque by each wheel
            # Does not need to be a solver variable since each torque
            # can be expressed in terms of force
            tau = []
            for j in range(4):
                # 2D version of cross product (torque = r x force)
                tau.append(module_positions[j][1] * Fx[j] - module_positions[j][0] * Fy[j])
                solver.subject_to(Fx[j] * Fx[j] + Fy[j] * Fy[j] < max_force_squared)

            # Newton's second law
            solver.subject_to(ax[k] * self.mass == Fx[0] + Fx[1] + Fx[2] + Fx[3])
            solver.subject_to(ay[k] * self.mass == Fy[0] + Fy[1] + Fy[2] + Fy[3])
            solver.subject_to(alpha[k] * self.moi == sum(tau))
