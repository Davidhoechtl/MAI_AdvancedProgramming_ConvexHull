import pygame
import numpy as np
import time

# Initialize Pygame
pygame.init()

# Window dimensions
width, height = 800, 600
window = pygame.display.set_mode((width, height))
pygame.display.set_caption('Convex Hull Visualization')
pygame.font.init() # you have to call this at the start, 
                   # if you want to use this module.
my_font = pygame.font.SysFont('Comic Sans MS', 30)

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Generate random points
def generate_points(n_points=5):
    """Generate random points in 2D space."""
    points = np.random.rand(n_points, 2)
    points[:, 0] = points[:, 0] * width
    points[:, 1] = points[:, 1] * height
    return points


def det(p1, p2, p3):
    """ 
    > 0: CCW turn
    < 0 CW turn
    = 0: colinear
    """
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) \
        -(p2[1] - p1[1]) * (p3[0] - p1[0])
# Gift Wrapping (Jarvis March) Algorithm (Step by Step)
def gift_wrapping_step(points, current_hull, p, q):
    """Compute the next point in the convex hull for the step visualization."""
    n = len(points)
    for i in range(n):
        # If point i is more counterclockwise than current q, update q
        if det(points[q], points[p], points[i]) > 0:
            q = i
    #print(f'Next point: {q}')
    current_hull.append(q)
    return q

# Draw points and hull
def draw(points, hull, current_point=None, next_point=None):
    """Draw the points, current hull, and next candidate point."""
    window.fill(WHITE)
    
    # Draw points
    for point in points:
        pygame.draw.circle(window, BLACK, point.astype(int), 2)
    
    # Draw convex hull
    if len(hull) > 1:
        for i in range(len(hull) - 1):
            pygame.draw.line(window, GREEN, points[hull[i]].astype(int), points[hull[i + 1]].astype(int), 2)
        pygame.draw.line(window, GREEN, points[hull[-1]].astype(int), points[hull[0]].astype(int), 2)
    
    # Draw current point
    if current_point is not None:
        pygame.draw.circle(window, RED, points[current_point].astype(int), 8)
    
    # Draw next point
    if next_point is not None:
        pygame.draw.circle(window, GREEN, points[next_point].astype(int), 8)
    
    pygame.display.update()

# Main function to step through the algorithm
def visualize_convex_hull_pygame():

    points = generate_points(n_points=50000)
    n = len(points)
    
    # Find the leftmost point to start the algorithm
    l = np.argmin(points[:, 0])
    p = l
    current_hull = [p]
    
    # Initial draw
    draw(points, current_hull, current_point=p)
    
    running = True
    next_step = False
    run = False
    q = (p + 1) % n

    text = my_font.render('Simulation running', True, GREEN, BLUE)
    textRect = text.get_rect()
    # set the center of the rectangular object.
    textRect.center = (width // 2, height // 2)
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    next_step = True
                    #print('Next step...')
                if event.key == pygame.K_r:
                    time1 = time.time()
                    run = True
                    next_step = True
                    #print('Next step...')

        if next_step:
            #print('Computing next step...')
            q = gift_wrapping_step(points, current_hull, p, q)
            p = q
            q = (p + 1) % n
            
            if p == l:  # When we return to the first point, the hull is complete
                #running = False
                #print('Hull complete!')
                run = False
                time2 = time.time()
                print(f'Time: {time2 - time1}')
            window.blit(text, textRect)
            pygame.display.update()
            if (not run):
                draw(points, current_hull, current_point=p, next_point=q)
            #next_step = False if run is not True else True  # Wait for the next key press
            next_step = True if run else False
            
        #pygame.time.wait(100)

    pygame.quit()
# Run the visualization
visualize_convex_hull_pygame()
