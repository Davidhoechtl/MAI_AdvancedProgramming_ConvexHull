import pygame
import numpy as np
import time
import matplotlib.pyplot as plt



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

def generate_circular_points(n_points=5):
    """Generate random points in 2D space."""
    points = []
    for i in range(n_points):
        angle = np.random.rand() * 2 * np.pi
        r = np.random.rand() * 200
        x = width // 2 + r * np.cos(angle)
        y = height // 2 + r * np.sin(angle)
        points.append([x, y])
    return np.array(points)

def det(p1, p2, p3):
    """ 
    > 0: CCW turn
    < 0 CW turn
    = 0: colinear
    """
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) \
        -(p2[1] - p1[1]) * (p3[0] - p1[0])

# Gift Wrapping (Jarvis March) Algorithm (Step by Step)
# Andere Algorithmen sollten die selbe Signatur haben. Sie akzeptieren die Punkte, die aktuelle Hülle und die beiden Punkte p und q, und geben den nächsten Punkt zurück.
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

def convex_hull(n_points, algorithm, points, start_running=False):
    if points == 'square':
        points = generate_points(n_points=n_points)
    elif points == 'circular':
        points = generate_circular_points(n_points=n_points)
    else:
        raise ValueError('Invalid point distribution')
    n = len(points)
    if algorithm == 'gift_wrapping':
        step_function = gift_wrapping_step
    else:
        raise ValueError('Invalid algorithm')
    
    # Find the leftmost point to start the algorithm
    l = np.argmin(points[:, 0])
    p = l
    current_hull = [p]

    running = True
    next_step = False
    run = False
    q = (p + 1) % n
    
    # Initial draw
    draw(points, current_hull, current_point=p)

    time1 = time.time()
    while running:
            if start_running:
                run = True
                next_step = True
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        next_step = True
                        #print('Next step...')
                    if event.key == pygame.K_r:
                        run = True

            if next_step or run:
                #print('Computing next step...')
                q = step_function(points, current_hull, p, q)
                p = q
                q = (p + 1) % n
                
                if p == l:  # When we return to the first point, the hull is complete
                    #running = False
                    #print('Hull complete!')
                    #run = False
                    running = False
                    
                #window.blit(text, textRect)
                #pygame.display.update()
                if (not run):
                #    pass
                    draw(points, current_hull, current_point=p, next_point=q)
                    next_step = False if run is not True else True  # Wait for the next key press
                #next_step = True if run else False
    time2 = time.time()
    time_elapsed = time2 - time1
    print(f'Time: {time_elapsed:.2f} seconds for {n} points.  KiloPoints per seconds : {(n/1000)/(time_elapsed):.2f}. Hull size: {len(current_hull)}')
    #speeds.append((n,(n/1000)/(time_elapsed)))
    return (n,time_elapsed, len(current_hull))


speeds = []
# Main function to step through the algorithm
def benchmark_convex_hull(algorithm, points, max_points):
    for i in (range(1000, max_points, 1000)):
        n, time_elapsed, hull_size = convex_hull(n_points=i, algorithm=algorithm, points=points, start_running=True)
        speeds.append((n,n/time_elapsed, hull_size))
        
    pygame.quit()

def plot_speeds(speeds, algorithm, points):
    x, y, h = zip(*speeds)
    fig, ax1 = plt.subplots()
    ax1.set_title(f'{algorithm} - {points}')
    ax1.set_xlabel('Number of Points')
    ax1.set_ylabel('KiloPoints per seconds', color='blue')
    ax1.plot(x, y, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Hull Size', color='red')
    ax2.plot(x, h, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    fig.tight_layout()
    plt.savefig(f'{algorithm}_{points}.png')
    plt.show()


# Run the visualization
algorithm = 'gift_wrapping'
points = 'square'
max_points = 20000

#convex_hull(n_points=100, algorithm=algorithm, points=points, start_running=False)

benchmark_convex_hull(algorithm, points, max_points)
plot_speeds(speeds, algorithm, points)
