def det(p1, p2, p3):
    """
    > 0: CCW turn
    < 0 CW turn
    = 0: colinear
    """
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) -(p2[1] - p1[1]) * (p3[0] - p1[0]) #Kreuzprodukt der Vektoren p1p2 und p1p3

def distance(p1, p2):
    """Calculates the distance between 2 points"""
    return (p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2