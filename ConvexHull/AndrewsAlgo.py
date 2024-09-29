import utils
import numpy as np

def check_det(points, current_hull, q):
    return utils.det(points[current_hull[-2]], points[current_hull[-1]], points[q]) < 0

def andrews_hull_step(points, current_hull, p, q):
    """
    Performs one step in the andrews algorithm
    :param points: all points of the current simulation
    :param current_hull: points that make up the convex hull
    :param p: index of current point
    :param q: index of next point
    :return: next point of the hull
    """
    # Update q to be the new farthest point
    while len(current_hull) >= 2 and check_det(points, current_hull, q):
        current_hull.pop()  # Remove the last point from the hull
    current_hull.append(q)  # Add the new point to the hull
    return q  # Return the index of the new point added
