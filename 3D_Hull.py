import numpy as np
import matplotlib.pyplot as plt

class Triangle:
    def __init__(self, p1, p2, p3, points):
        # Assuming p1, p2, and p3 are indices
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3

        self.p1_id = np.where(points == p1)[0][0]
        self.p2_id = np.where(points == p2)[0][0]
        self.p3_id = np.where(points == p3)[0][0]
        self.normal = np.cross(
            points[self.p2_id] - points[self.p1_id], points[self.p3_id] - points[self.p1_id])
        self.normal /= np.linalg.norm(self.normal)
        self.center = (points[self.p1_id] +
                       points[self.p2_id] + points[self.p3_id]) / 3
        self.positive_points = []

    def __str__(self):
        return "Triangle: p1 = " + str(self.p1) + ", p2 = " + str(self.p2) + ", p3 = " + str(self.p3)

    def __eq__(self, other):
        return np.array_equal(points[self.p1_id], points[other.p1_id]) \
            and np.array_equal(points[self.p2_id], points[other.p2_id]) \
            and np.array_equal(points[self.p3_id], points[other.p3_id])

    def __hash__(self):
        return hash((self.p1_id, self.p2_id, self.p3_id))

def extremePoints(points):
    # Find extreme points
    max_x_point = points[np.argmax(points[:, 0])]
    min_x_point = points[np.argmin(points[:, 0])]

    max_y_point = points[np.argmax(points[:, 1])]
    min_y_point = points[np.argmin(points[:, 1])]

    max_z_point = points[np.argmax(points[:, 2])]
    min_z_point = points[np.argmin(points[:, 2])]

    return max_x_point, min_x_point, max_y_point, min_y_point, max_z_point, min_z_point

def normalize_vector(vector):
    magnitude = np.linalg.norm(vector)  # Calculate the magnitude of the vector
    # Normalize the vector by dividing each component by the magnitude
    normalized_vector = vector / magnitude
    return normalized_vector

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def extreme_points_distances(points):
    max_x_point, min_x_point, max_y_point, min_y_point, max_z_point, min_z_point = extremePoints(
        points)

    # Calculate distances between extreme points
    distances = {
        euclidean_distance(max_x_point, min_x_point): (max_x_point, min_x_point),
        euclidean_distance(max_y_point, min_y_point): (max_y_point, min_y_point),
        euclidean_distance(max_z_point, min_z_point): (max_z_point, min_z_point),
        euclidean_distance(max_x_point, max_y_point): (max_x_point, max_y_point),
        euclidean_distance(max_x_point, min_y_point): (max_x_point, min_y_point),
        euclidean_distance(max_x_point, max_z_point): (max_x_point, max_z_point),
        euclidean_distance(max_x_point, min_z_point): (max_x_point, min_z_point),
        euclidean_distance(min_x_point, max_y_point): (min_x_point, max_y_point),
        euclidean_distance(min_x_point, min_y_point): (min_x_point, min_y_point),
        euclidean_distance(min_x_point, max_z_point): (min_x_point, max_z_point),
        euclidean_distance(min_x_point, min_z_point): (min_x_point, min_z_point),
        euclidean_distance(max_y_point, max_z_point): (max_y_point, max_z_point),
        euclidean_distance(max_y_point, min_z_point): (max_y_point, min_z_point),
        euclidean_distance(min_y_point, max_z_point): (min_y_point, max_z_point),
        euclidean_distance(min_y_point, min_z_point): (min_y_point, min_z_point),
    }

    max_distance = max(distances)
    max_distance_points = distances[max_distance]

    return max_distance_points

def projection(point, line):
    v1 = line[1] - line[0]
    v2 = point - line[0]

    vn = normalize_vector(v1)

    sp = np.dot(vn, v2)
    tp = line[0] + vn * sp

    if (0 <= sp).all() and (sp <= np.linalg.norm(v1)).all():
        return tp
    else:
        return None
def tetrahedron(points):
    extrems = extremePoints(points)
    max_distance_points = extreme_points_distances(points)

    tetraeder = [max_distance_points[0], max_distance_points[1]]

    maxDis = 0
    point = None

    for extrem in extrems:
        if all(np.any(extrem != line) for line in tetraeder):
            projected_point = projection(extrem, tetraeder)
            dis = 0
            if projected_point is not None:
                dis = euclidean_distance(extrem, projected_point)
            else:
                dis = min(euclidean_distance(extrem, tetraeder[0]), euclidean_distance(extrem, tetraeder[1]))
            if maxDis < dis:
                maxDis = dis
                point = extrem
    tetraeder.append(point)

    # Calculate two vectors representing two edges of the triangle
    v1 = tetraeder[1] - tetraeder[0]
    v2 = tetraeder[2] - tetraeder[0]

    # Take the cross product of the two vectors to find the normal vector
    vn = np.cross(v1, v2)

    # Normalize the normal vector
    vn /= np.linalg.norm(vn)

    maxAbs = 0
    furthest_point = None
    for p in points:
        vp = p - tetraeder[0]
        n = abs(np.dot(vn, vp))
        if maxAbs < n:
            maxAbs = n
            furthest_point = p

    tetraeder.append(furthest_point)

    return tetraeder

def draw_tetrahedron(ax, tetra):
    for i in range(len(tetra)):
        for j in range(i + 1, len(tetra)):
            ax.plot([tetra[i][0], tetra[j][0]], [tetra[i][1], tetra[j][1]], [tetra[i][2], tetra[j][2]], 'b')

if __name__ == '__main__':
    # Generate random points
    num_points = 20
    x = np.random.rand(num_points)  # Random x coordinates
    y = np.random.rand(num_points)  # Random y coordinates
    z = np.random.rand(num_points)  # Random z coordinates

    points = np.column_stack((x,y,z))
    tetra = tetrahedron(points)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # Show points
    ax.scatter3D(points[:, 0], points[:, 1], points[:, 2], c='r')

    draw_tetrahedron(ax, tetra)

    # Set labels and title
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    # ax.set_title('Random 3D Scatter Plot')

    # Show the plot
    plt.show()