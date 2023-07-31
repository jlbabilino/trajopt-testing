from trajectory_generator import TrajectoryGenerator, Waypoint, Trajectory
from drive.swerve_drive import SwerveDrive
import trajectory_viz
import math
import matplotlib.pyplot as plt

def main():
    drive = SwerveDrive(
        # Wheelbase x/y
        0.622,0.572,
        # Bumper length/width
        0.954,0.903,
        # Mass/moi
        46.7,5.6,
        # Max velocity/force
        0.0000003, 0.0000003,
        # 73, 1,
        # 50, 1,
        # Wheel radius
        0.051)
    
    waypoints = [
        Waypoint(0.0, 0.0, 0.0),
        Waypoint(5.0, 3.0, 2.0),
        Waypoint(3.0, 1.0, -1.0),
        Waypoint(2.0, 3.0, 5.0),
        Waypoint(7.0, 4.0, 3.0)]

    generator = TrajectoryGenerator(drive, waypoints)

    trajectory = generator.generate()

    trajectory_viz.animate_trajectory(trajectory.x, trajectory.y, trajectory.theta, waypoints, drive, 0.01, "hi")


    # fig, ax = trajectory_viz.draw_field()

    # for waypoint in waypoints:
    #     trajectory_viz.draw_robot(ax, waypoint, drive)

    # plt.plot(trajectory["x"], trajectory["y"])

    # plt.show()




main()