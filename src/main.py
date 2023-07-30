from trajectory_generator import TrajectoryGenerator, Waypoint, Trajectory
from drive.swerve_drive import SwerveDrive
import trajectory_viz
import math
import matplotlib.pyplot as plt

def main():
    drive = SwerveDrive(
        # Wheelbase x/y
        0.622 * 2,  0.572 * 2,
        # Mass/moi
        46.7, 5.6,
        # Max velocity/force
        3.0, 1.2,
        # 20.0, 1.2,
        # static friction
        1.1)
    
    waypoints = [
        Waypoint(0.0, 0.0, 0.0),
        # Waypoint(5.0, 3.0, 2.0, False),
        Waypoint(3.0, 3.0, 0.0)]

    generator = TrajectoryGenerator(drive, waypoints)

    trajectory = generator.generate()

    trajectory_viz.animate_trajectory(trajectory.x, trajectory.y, trajectory.theta, waypoints, drive, 0.01, "hi")


    # fig, ax = trajectory_viz.draw_field()

    # for waypoint in waypoints:
    #     trajectory_viz.draw_robot(ax, waypoint, drive)

    # plt.plot(trajectory["x"], trajectory["y"])

    # plt.show()




main()