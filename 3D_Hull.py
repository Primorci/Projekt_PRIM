import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Generate random points
    num_points = 20
    x = np.random.rand(num_points)  # Random x coordinates
    y = np.random.rand(num_points)  # Random y coordinates
    z = np.random.rand(num_points)  # Random z coordinates

    points = np.column_stack((x,y,z))

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # Show points
    ax.scatter3D(points[:, 0], points[:, 1], points[:, 2], c='r')

    # Set labels and title
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    # ax.set_title('Random 3D Scatter Plot')

    # Show the plot
    plt.show()