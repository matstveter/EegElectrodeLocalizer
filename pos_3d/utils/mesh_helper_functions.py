import pyvista as pv
import numpy as np
import trimesh


def manual_point_selection(obj_file_path, jpg_file_path, point_to_be_picked, point_transform, rotation_matrix=None,
                           suggested_point=None):
    """ Function to man

    Parameters
    ----------
    rotation_matrix: np.ndarray
        rotation matrix if that exists for the mesh
    point_transform: np.ndarray
        if the points should be shifted
    obj_file_path: str
        path to the .obj file
    jpg_file_path: str
        path to the jpg file
    point_to_be_picked: str
        String that specifies which point we are choosing
    suggested_point: np.ndarray
        If there are any suggested points

    Returns
    -------
    point: np.ndarray
        The last picked point in mesh

    """
    picked_point = []

    def get_point(event):
        picked_point.append(plotter.picked_point)

    plotter = pv.Plotter()
    # Read the mesh and texture using the pyvista library
    mesh = pv.read(obj_file_path)
    texture = pv.read_texture(jpg_file_path)
    # Transform the mesh according to the rotation matrix

    mesh.points = np.dot(mesh.points, rotation_matrix.T)
    if point_transform is not None:
        mesh.points = mesh.points - point_transform

    plotter.add_mesh(mesh=mesh, texture=texture)

    if suggested_point is not None:
        sphere = pv.Sphere(radius=0.002, center=suggested_point)
        plotter.add_mesh(sphere, color="r")

    plotter.enable_surface_point_picking(show_point=True, color="blue", tolerance=0.002, callback=get_point,
                                         font_size=15,
                                         show_message=f"Right Mouse Button to pick point: {point_to_be_picked}")
    # plotter.enable_point_picking(show_point=True, color="blue", tolerance=0.01, callback=get_point,
    #                                      font_size=15,
    #                                      show_message=f"Right Mouse Button to pick point: {point_to_be_picked}")
    plotter.show(window_size=[5000, 5000])
    # plotter.show()
    plotter.close()

    # Assumes that the last point is the point that should be selected
    if picked_point:
        return picked_point[-1]
    else:
        return None


def get_ray_trace_point(mesh, point, direction):
    """
    Traces a ray from a given point in a specified direction and returns the intersecting point on the mesh.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The mesh object that the ray is being cast against.
    point : array_like
        The starting point of the ray. Must be a 3-element array or list [x, y, z].
    direction : array_like
        The direction in which to cast the ray. Must be a 3-element array or list [dx, dy, dz].

    Returns
    -------
    array_like or None
        Returns the intersecting point on the mesh if found. If multiple intersections are found,
        returns the point that is furthest away from the starting point.
        Returns None if no valid intersection points are found.

    Notes
    -----
    - This function first tries tracing the ray in the given direction. If no intersection is found,
      it tries the opposite direction.
    - It also accounts for the sign of the x-coordinate of the point to adjust the ray direction.
    """
    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)

    # Go the opposite way of the sign of the x-axis
    if np.sign(point[0]) > 0:
        direction = np.array([-1, 0, 0])

    _, _, locations = intersector.intersects_id(ray_origins=[point], ray_directions=[direction],
                                                return_locations=True)
    if len(locations) == 0:

        # No points found in this direction, try other, and return the first point which is not the original...
        _, _, locations = intersector.intersects_id(ray_origins=[point], ray_directions=[-1 * np.array(direction)],
                                                    return_locations=True)
        return [p for p in locations if not np.allclose(point, p)][0]
    elif len(locations) == 1:
        if not np.allclose(point, locations[0], atol=1e-2):
            return locations[0]
        else:
            return None
    else:
        # Return the point that is the furthest away the point
        distances = [np.linalg.norm(np.array(p) - np.array(point)) for p in locations]
        return locations[np.argmax(distances)]


def transform_mesh(mesh, rotation_matrix, point_transform=None):
    """ Rotate the mesh instance using the rotation matrix and transform points if it is not None

    Parameters
    ----------
    mesh: Trimesh.trimesh
        mesh
    rotation_matrix: np.ndarray
        rotation matrix
    point_transform: np.ndarray
        transformation of points

    Returns
    -------
    trimesh.Mesh

    """
    mesh.vertices = mesh.vertices.dot(rotation_matrix.T)
    if point_transform is not None:
        # Apply the actual transformation of points
        mesh.apply_translation(-point_transform)
    return mesh


def find_white_gray_vertices(mesh, vertex_colors, point, distance_threshold=0.025, proximity_threshold=0.01,
                             not_use_original_point=False):
    """
    Identify candidate vertices on a mesh that are close to white or gray and
    within a specified distance from a given point. Group these candidates based
    on proximity and find the centroid of each group. Return the centroid that is
    closest to the given point, considering both the Euclidean distance and the
    x-coordinate difference.

    Parameters:
    mesh: trimesh.base.Trimesh
        The mesh containing the vertices.
    vertex_colors: np.ndarray
        An array of HSV colors for each vertex in the mesh.
    point: np.ndarray
        The reference point (numpy array of shape (3,)).
    distance_threshold: float, optional
        Maximum distance from the point to consider a vertex. Defaults to 0.025.
    proximity_threshold: float, optional
        Maximum distance between vertices to consider them as part of the same group. Defaults to 0.01.

    Returns:
    numpy.ndarray:
        The centroid of the vertex group that best matches the criteria.
    """
    # Thresholds for identifying gray-white vertices
    saturation_threshold = 0.4
    value_threshold = 120

    # List to store candidate vertices
    candidate_vertices = []

    closest_vertex_index = None
    min_distance = float('inf')

    # Iterate through each vertex in the mesh
    for i, vertex in enumerate(mesh.vertices):
        distance = np.linalg.norm(vertex - point)
        if distance < min_distance:
            min_distance = distance
            closest_vertex_index = i

    # If the point is indeed within the range, no need to look at other vertices surrounding it...
    _, s, v = vertex_colors[closest_vertex_index]
    if s <= saturation_threshold and v >= value_threshold and not_use_original_point:
        return [point]

    # Iterate through each vertex in the mesh
    for i, vertex in enumerate(mesh.vertices):
        _, sat, val = vertex_colors[i]

        # Check if the vertex color is within the gray-white HSV range
        if sat <= saturation_threshold and val >= value_threshold:
            # Calculate the distance from the current vertex to the given point
            distance = np.linalg.norm(vertex - point)

            # Check if the distance is within the specified threshold (2.5 cm or 0.025)
            if distance <= distance_threshold:
                candidate_vertices.append(vertex)

    # Return the original point if no candidates are found
    if not candidate_vertices:
        return []

    # Group vertices based on proximity
    groups = []
    for pos in candidate_vertices:
        found_group = False
        # Check proximity to vertices in existing groups
        for group in groups:
            if any(np.linalg.norm(pos - other_pos) <= proximity_threshold for other_pos in group):
                group.append(pos)
                found_group = True
                break
        if not found_group:
            groups.append([pos])

    # Calculate the centroid of each group
    centroids = [np.mean(group, axis=0) for group in groups]
    return centroids

    # # If no centroids is found, return the point, if 1 return that, else return the closest in x-axis
    # if len(centroids) == 0:
    #     return point
    # elif len(centroids) == 1:
    #     return centroids[0]
    # else:
    #     dist_weight = 0.5
    #     dists = [np.linalg.norm(point - centre) for centre in centroids]
    #     x_diffs = [np.abs(point[0] - centre[0]) for centre in centroids]
    #
    #     combined_scores = [x_diff + dist_weight * dist for x_diff, dist in zip(x_diffs, dists)]
    #
    #     return centroids[combined_scores.index(min(combined_scores))]
