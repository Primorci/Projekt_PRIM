import matplotlib.pyplot as plt
import numpy as np

class Triangle:
    def __init__(self, p1, p2, p3, points):
        # Initialize a triangle with given points
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3

        # Get the indices of the points in the points array
        self.p1_id = np.where(points == p1)[0][0]
        self.p2_id = np.where(points == p2)[0][0]
        self.p3_id = np.where(points == p3)[0][0]

        # Calculate the normal vector of the triangle
        self.normal = np.cross(
            points[self.p2_id] - points[self.p1_id], points[self.p3_id] - points[self.p1_id])
        self.normal /= np.linalg.norm(self.normal)

        # Calculate the center of the triangle
        self.center = (points[self.p1_id] +
                       points[self.p2_id] + points[self.p3_id]) / 3

        # List to store points that are above the triangle
        self.positive_points = []

    def __str__(self):
        return "Triangle: p1 = " + str(self.p1) + ", p2 = " + str(self.p2) + ", p3 = " + str(self.p3)

    def __eq__(self, other):
        # Compare two triangles for equality
        return np.array_equal(points[self.p1_id], points[other.p1_id]) \
            and np.array_equal(points[self.p2_id], points[other.p2_id]) \
            and np.array_equal(points[self.p3_id], points[other.p3_id])

    def __hash__(self):
        # Generate a hash value for a triangle
        return hash((self.p1_id, self.p2_id, self.p3_id))

def extremePoints(points):
    # Find extreme points in x, y, and z directions
    max_x_point = points[np.argmax(points[:, 0])]
    min_x_point = points[np.argmin(points[:, 0])]

    max_y_point = points[np.argmax(points[:, 1])]
    min_y_point = points[np.argmin(points[:, 1])]

    max_z_point = points[np.argmax(points[:, 2])]
    min_z_point = points[np.argmin(points[:, 2])]

    return max_x_point, min_x_point, max_y_point, min_y_point, max_z_point, min_z_point

def euclidean_distance(point1, point2):
    # Calculate Euclidean distance between two points
    return np.sqrt(np.sum((point1 - point2) ** 2))

def extreme_points_distances(points):
    # Calculate distances between all pairs of extreme points
    max_x_point, min_x_point, max_y_point, min_y_point, max_z_point, min_z_point = extremePoints(points)

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

def normalize_vector(vector):
    # Normalize a vector
    magnitude = np.linalg.norm(vector)
    normalized_vector = vector / magnitude
    return normalized_vector

def projection(point, line):
    # Project a point onto a line
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
    # Find a tetrahedron that encloses the points
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

    v1 = tetraeder[1] - tetraeder[0]
    v2 = tetraeder[2] - tetraeder[0]

    vn = np.cross(v1, v2)
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

def hull(points):
    # Construct the convex hull using the gift wrapping algorithm
    def adjust_normals():
        # Adjust the normals to ensure they point outward
        for new_triangle in newTriangles:
            op = 0
            for triangle_point in [triangle.p1_id, triangle.p2_id, triangle.p3_id]:
                if triangle_point != new_triangle.p1_id and triangle_point != new_triangle.p2_id and triangle_point != new_triangle.p3_id:
                    op = triangle_point
                    break
            if np.dot(new_triangle.normal, new_triangle.center - points[op]) < 0:
                new_triangle.normal *= -1

    def find_pos_points():
        # Find points that are above the triangle
        for new_triangle in newTriangles:
            for p in points:
                if (p != new_triangle.p1).all() and (p != new_triangle.p2).all() and (p != new_triangle.p3).all() \
                    and np.dot(new_triangle.normal, p - new_triangle.center) > 0:
                        new_triangle.positive_points.append([np.where(points == p)[0][0], 
                                                             np.dot(new_triangle.normal, 
                                                            p - new_triangle.center)])
    
    tetraeder = tetrahedron(points)

    triangles = [Triangle(tetraeder[0], tetraeder[1], tetraeder[2], points),
                 Triangle(tetraeder[0], tetraeder[1], tetraeder[3], points),
                 Triangle(tetraeder[0], tetraeder[2], tetraeder[3], points),
                 Triangle(tetraeder[1], tetraeder[2], tetraeder[3], points)]

    for triangle in triangles:
       op = 0
       for i in range(4):
           if i != np.where(tetraeder == triangle.p1)[0][0] \
                   and i != np.where(tetraeder == triangle.p2)[0][0] \
                   and i != np.where(tetraeder == triangle.p3)[0][0]:
               op = i
               break
       if np.dot(triangle.normal, triangle.center - tetraeder[op]) < 0:
            triangle.normal *= -1

    lines_normals = []
    for triangle in triangles:
        lines_normals.append(
            [triangle.center, triangle.center + triangle.normal])
        
    poz_points = []
    for triangle in triangles:
        for i in range(len(points)):
            if i != triangle.p1_id and i != triangle.p2_id and i != triangle.p3_id:
                p = np.dot(triangle.normal, points[i] - triangle.center)
                if p > 0:
                    triangle.positive_points.append([i, p])
                    poz_points.append(points[i])
                    
    triangleStack = []
    for triangle in triangles:
        triangleStack.append(triangle)
        
    hull = []
    for triangle in triangles:
        hull.append(triangle)
        
    while triangleStack:
        hull = list(set(hull))
        
        currTriangle = None
        for tri in triangleStack:
            if not tri.positive_points:
                currTriangle = tri
                break
    
        if currTriangle is None:
            currTriangle = triangleStack.pop()
        else:
            triangleStack.remove(currTriangle)
            if currTriangle in hull:
                hull.remove(currTriangle)
    
        if not currTriangle.positive_points:
            hull.append(currTriangle)
            continue
    
        maxp = max(currTriangle.positive_points, key=lambda pt: pt[1])
    
        posTriangles = [
            stack_tri for stack_tri in triangleStack
            if np.dot(stack_tri.normal, points[maxp[0]] - stack_tri.center) > 0
        ]
    
        if not posTriangles:
            newTriangles = [
                Triangle(currTriangle.p1, currTriangle.p2, points[maxp[0]], points),
                Triangle(currTriangle.p1, currTriangle.p3, points[maxp[0]], points),
                Triangle(currTriangle.p2, currTriangle.p3, points[maxp[0]], points)
            ]
    
            adjust_normals()
            find_pos_points()
    
            for new_tri in newTriangles:
                triangleStack.append(new_tri)
                hull.append(new_tri)
            if currTriangle in hull:
                hull.remove(currTriangle)
    
        else:
            posTriangles.append(currTriangle)
            triangle_dict = {}
    
            for pos_tri in posTriangles:
                for point in [pos_tri.p1_id, pos_tri.p2_id, pos_tri.p3_id]:
                    triangle_dict[point] = triangle_dict.get(point, 0) + 1
    
            for pos_tri in posTriangles:
                if pos_tri in hull:
                    hull.remove(pos_tri)
    
            group = list(triangle_dict.keys())
    
            edges = [
                edge for hull_tri in hull
                for edge in [[hull_tri.p1_id, hull_tri.p2_id], [hull_tri.p1_id, hull_tri.p3_id], [hull_tri.p2_id, hull_tri.p3_id]]
                if edge[0] in group and edge[1] in group
            ]
    
            cEdge = {}
            for edge in edges:
                edge_key = tuple(sorted(edge))
                cEdge[edge_key] = cEdge.get(edge_key, 0) + 1
    
            uEdges = [list(edge) for edge, count in cEdge.items() if count == 1]
    
            newTriangles = [
                Triangle(points[edge[0]], points[edge[1]], points[maxp[0]], points)
                for edge in uEdges
            ]
    
            adjust_normals()
            find_pos_points()
    
            for new_tri in newTriangles:
                triangleStack.append(new_tri)
                hull.append(new_tri)
    
            for pos_tri in posTriangles:
                if pos_tri in triangleStack:
                    triangleStack.remove(pos_tri)
    
    return list(set(hull))

if __name__ == '__main__':
    # Generate random points
    num_points = 20
    x = np.random.rand(num_points)  # Random x coordinates
    y = np.random.rand(num_points)  # Random y coordinates
    z = np.random.rand(num_points)  # Random z coordinates

    points = np.column_stack((x, y, z))

    # Calculate the convex hull of the points
    hull = hull(points)
        
    # Create a triangle mesh from the hull
    shell_lines = []
    for triangle in hull:
        shell_lines.append([triangle.p1, triangle.p2])
        shell_lines.append([triangle.p1, triangle.p3])
        shell_lines.append([triangle.p2, triangle.p3])

    # Plot the points and the convex hull
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(points[:, 0], points[:, 1], points[:, 2], c='r')  # Plot the points
    for line in shell_lines:
        ax.plot3D([line[0][0], line[1][0]],
                  [line[0][1], line[1][1]],
                  [line[0][2], line[1][2]],
                  c='g')  # Plot the hull edges

    # Set labels
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # Show the plot
    plt.show()
