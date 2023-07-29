import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
import math
import os
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.transforms as transforms

def draw_field():
    plt.style.use("classic")
    fig, ax = plt.subplots()
    ax.add_patch(mpl.patches.Rectangle(
        (0, 0),
        30,
        15,
        lw=4,
        edgecolor="black",
        facecolor="none",
    ))
    plt.title("Trajectory")
    plt.xlabel("X Position (meters)")
    plt.ylabel("y Position (meters)")
    plt.ylim(-2.0,8.23)
    plt.xlim(-1.0,16.46)
    plt.gca().set_aspect("equal", adjustable="box")
    return fig, ax

def draw_trajectory(x_coords, y_coords, angular_coords, waypoints, drive, title):
    fig, ax = draw_field()
    # draw_robot(ax,[x_coords[0],y_coords[0],angular_coords[0]],drive)
    # draw_robot(ax,[x_coords[-1],y_coords[-1],angular_coords[-1]],drive)
    plt.plot(x_coords,y_coords,color="b")
    plt.title(title)
    for waypoint in waypoints:
        # draw_robot(ax, waypoint, drive)
        pass
    # for pose in zip(x_coords, y_coords, angular_coords):
    #     draw_robot(ax,pose, drive)

def animate_trajectory(
    x_coords,
    y_coords,
    angular_coords,
    waypoints,
    drive,
    dt,
    title
):
    
    fig, ax = draw_field()

    for waypoint in waypoints:
        # draw_robot(ax, waypoint, drive)
        pass

    num_states = len(x_coords)
    plt.plot(x_coords, y_coords)

    myrect = patches.Rectangle(
        (0, 0),
        drive.length, drive.width,
        fc="y",
        rotation_point="center")
    
    myrect.set_fill(False)
    

    def init():
        ax.add_patch(myrect)
        return myrect,

    def animate(i):
        # myrect.set_width(drive.length)
        # myrect.set_height(drive.width)

        robot_transform = transforms.Affine2D().translate(-drive.length / 2, -drive.width / 2).rotate(angular_coords[i]).translate(x_coords[i], y_coords[i]) + ax.transData 
        # robot_transform.clear()


        myrect.set_transform(robot_transform)
        myrect.set_x(0)
        myrect.set_y(0)
        myrect.set_angle(0)
        return myrect,

    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=num_states, interval=10, blit=True, repeat=True
    )

    plt.show()

    if not os.path.exists("animations"):
        os.makedirs("animations")
    anim.save(
        os.path.join("animations", "{}.gif".format(title)),
        writer="pillow",
        dpi=100,
        fps=(int)(1 / dt),
    )
    return anim
