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

        ╭─────────────────────────────────────────╮                   ↑ 
        │                                         │                   │
        │   ╭╮                             ╭╮     │                   │
        │   ││  rear_left       front_left ││     │   ↑               │
        │   ╰╯              y              ╰╯     │   │               │
        │                   ↑                     │   │               │
        │                   │                     │   │               │
        │                   └────→ x              │   │ track_width   │ bumpers_width
        │                                         │   │               │
        │                                         │   │               │
        │   ╭╮                             ╭╮     │   │               │
        │   ││   rear_right    front_right ││     │   ↓               │
        │   ╰╯                             ╰╯     │                   │
        │                                         │                   │
        ╰─────────────────────────────────────────╯                   ↓
             ←───────── wheelbase ─────────→
        ←─────────── bumpers_length ──────────────→

        The nonrotating robot coordinate system is defined with the same center as the robot coordinate
        system, but it does not rotate and its axes always point in the same directions as the field
        coordinate system.
    """
    def __init__(self,
            wheelbase: float,
            track_width: float,
            mass: float,
            moi: float,
            kV: float,
            kF: float,
            mu_s: float):
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
        self.wheelbase = wheelbase
        self.track_width = track_width
        self.mass = mass
        self.moi = moi
        self.kV = kV
        self.kF = kF
        self.mu_s = mu_s

    def solve_module_positions(self, theta: MX) -> MX:
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
        module_angle = atan2(self.track_width / 2, self.wheelbase / 2)
        module_angles = (module_angle, -module_angle, pi - module_angle, -(pi - module_angle))
        diagonal = hypot(self.track_width / 2, self.wheelbase / 2)
        module_positions = MX(2, 4)
        for module_idx in range(4):
            module_positions[0, module_idx] = diagonal*cos(module_angles[module_idx]+theta)
            module_positions[1, module_idx] = diagonal*sin(module_angles[module_idx]+theta)

        return module_positions

    def add_constraints(self,
            opti: Opti,
            theta: MX,
            module_rotation: MX,
            vx: MX,
            vy: MX,
            omega: MX,
            ax: MX,
            ay: MX,
            alpha: MX,
            F_prime: MX,
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

        for k in range(N_total):
            # Calculate positions of each module relative to field coordinate system
            module_positions = self.solve_module_positions(theta[k])

            # Unit vectors of module rotation in module's coordinate system

            module_rotations = vertcat(module_rotation[:4, k].T, module_rotation[4:8, k].T)

            # Magnitude of velocity of each module
            module_velocity_magnitudes = opti.variable(1, 4)

            # collect velocity variables
            for module_idx in range(4):
                # Use IK to calculate module velocity vectors
                v_m = vertcat(vx[k] - module_positions[1,module_idx] * omega[k],
                              vy[k] + module_positions[0,module_idx] * omega[k])

                v_m_norm = module_velocity_magnitudes[module_idx]

                v_m_hat = module_rotations[:,module_idx]

                opti.set_initial(v_m_hat, DM([1.0, 0.0]))

                apply_unit_vector_constraint(opti, v_m_hat)

                opti.subject_to(v_m == v_m_norm * v_m_hat)

                # Force module velocity unit vector to point in direction of
                # module velocity
                opti.subject_to(v_m_norm >= 0.0)

            # Force applied by each module, separated into longitudinal and
            # lateral components, respectively
            # "prime" means in rotated reference frame of wheel
            samp_F_prime = horzcat(F_prime[:4, k], F_prime[4:8, k]).T

            F = MX(2, 4)
            tau = MX(0.0)

            # Collect force variables
            for module_idx in range(4):
                # Assume the robot lives on planet earth
                # Assume even weight distribution (low cg helps with this)
                F_N = (self.mass * 9.8) / 4

                # Constrain F within the "friction circle"
                constrain_vector_norm(opti, samp_F_prime, F_N)

                v_m_hat = module_rotations[:,module_idx]
                v_perp_m_hat = calculate_perpendicular_vector(v_m_hat)

                F_m_prime = samp_F_prime[:,module_idx]

                # Force components as vectors in field frame
                F_m_longitudinal = F_m_prime[0] * v_m_hat
                F_m_lateral = F_m_prime[1] * v_perp_m_hat

                r_m = module_positions[:,module_idx]
                F_m = F_m_longitudinal + F_m_lateral

                F[:, module_idx] = F_m
                tau += cross_product(r_m, F_m)

            # Apply power constraints
            for module_idx in range(4):
                F_m_longitudinal = samp_F_prime[0, module_idx]
                v_m_norm = module_velocity_magnitudes[module_idx]

                # The motor power equation, accounting for the case where
                # longitudinal force is in the opposite direction of velocity
                # https://www.desmos.com/calculator/rfrh8elbsx
                opti.subject_to(self.kV * v_m_norm + self.kF * F_m_longitudinal <= 12)
                opti.subject_to(self.kV * v_m_norm - self.kF * F_m_longitudinal <= 12)

            # Newton's second law
            opti.subject_to(ax[k] * self.mass == F[0, 0] + F[0, 1] + F[0, 2] + F[0, 3])
            opti.subject_to(ay[k] * self.mass == F[1, 0] + F[1, 1] + F[1, 2] + F[1, 3])
            opti.subject_to(alpha[k] * self.moi == tau)

def apply_unit_vector_constraint(opti: Opti, vec: MX):
    opti.subject_to(vec[0] * vec[0] + vec[1] * vec[1] == 1.0)

def calculate_perpendicular_vector(vec: MX):
    return vertcat(-vec[1], vec[0])

def constrain_vector_norm(opti: Opti, vec: MX, norm: float):
    opti.subject_to(vec[0] * vec[0] + vec[1] * vec[1] <= norm * norm)

def cross_product(a: MX, b: MX):
    return a[0] * b[1] - a[1] * b[0]
