import pygame
import numpy as np
import time
import matplotlib.pyplot as plt
from pyparsing import White

import utils
import AndrewsAlgo

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

def generate_ring_points(n_points=5):
    """Generate random points on a ring in 2D space."""
    points = []
    for i in range(n_points):
        angle = np.random.rand() * 2 * np.pi
        r = 200
        x = width // 2 + r * np.cos(angle)
        y = height // 2 + r * np.sin(angle)
        points.append([x, y])
    return np.array(points)

def generate_rectangular_ring_points(n_points=5):
    """Generate random points on the sides of a rectangle in 2D space."""
    points = []
    for i in range(n_points):
        side = np.random.randint(4)
        if side == 0:  # Top side
            x = np.random.rand() * (width*0.9)
            y = height*0.1
        elif side == 1:  # Right side
            x = width*0.9
            y = np.random.rand() * (height*0.9)
        elif side == 2:  # Bottom side
            x = np.random.rand() * (width*0.9)
            y = height*0.9
        else:  # Left side
            x = width*0.1
            y = np.random.rand() * height*0.9
        points.append([x, y])
    return np.array(points)

def import_points_from_file(file_path):
    """Import points from a file with the given format."""
    points = []
    with open(file_path, 'r') as file:
        n_points = int(file.readline().strip())  # Erste Zeile: Anzahl der Punkte
        for line in file:
            x, y = map(float, line.strip().split(','))  # Trenne die x- und y-Koordinaten
            points.append([x, y])
    return np.array(points)

# Gift Wrapping (Jarvis March) Algorithm (Step by Step)
# Andere Algorithmen sollten die selbe Signatur haben. Sie akzeptieren die Punkte, die aktuelle Hülle und die beiden Punkte p und q, und geben den nächsten Punkt zurück.
def gift_wrapping_step(points, current_hull, p, q):
    """Compute the next point in the convex hull for the step visualization."""
    n = len(points)
    for i in range(n):
        #draw(points, current_hull, current_point=p, next_point=i) #für bessere Visualisierung
        #pygame.time.delay(500) #für bessere Visualisierung

           # Berechne das Kreuzprodukt
        orientation = utils.det(points[p], points[q], points[i])

        # Wenn Punkt i gegen den Uhrzeigersinn von Punkt q liegt, aktualisiere q
        if orientation > 0:
            q = i
        # Wenn Punkt i kollinear mit p und q ist, wähle den weiter entfernten Punkt
        elif orientation == 0:
            if utils.distance(points[p], points[i]) > utils.distance(points[p], points[q]):#
                q = i
         
    current_hull.append(q)
    return q

# Draw points and hull
def draw(points, hull, current_point=None, next_point=None, clearWindow = True, hull_color=GREEN):
    """Draw the points, current hull, and next candidate point."""

    if clearWindow:
        window.fill(WHITE)
    
    # Draw points
    for point in points:
        pygame.draw.circle(window, BLACK, point.astype(int), 2)
    
    # Draw convex hull
    if len(hull) > 1:
        for i in range(len(hull) - 1):
            pygame.draw.line(window, hull_color, points[hull[i]].astype(int), points[hull[i + 1]].astype(int), 2)
        pygame.draw.line(window, hull_color, points[hull[-1]].astype(int), points[hull[0]].astype(int), 2)
    
    # Draw current point
    if current_point is not None:
        pygame.draw.circle(window, RED, points[current_point].astype(int), 8)
    
    # Draw next point
    if next_point is not None:
        pygame.draw.circle(window, GREEN, points[next_point].astype(int), 8)
    
    pygame.display.update()

def init_Points(points, n_points, file_path = None):
    if points == 'square':
        points = generate_points(n_points=n_points)
    elif points == 'circular':
        points = generate_circular_points(n_points=n_points)
    elif points == 'ring':
        points = generate_ring_points(n_points=n_points)
    elif points == 'square-ring':
        points = generate_rectangular_ring_points(n_points=n_points)
    elif points == 'file':
        if file_path is None:
            raise ValueError("No file path provided for importing points")
        points = import_points_from_file(file_path)
    else:
        raise ValueError('Invalid point distribution')

    return points

def convex_hull_giftwarpping(n_points, points, start_running=False, file_path=None): #Thomas: neu: , file_path=None
    points = init_Points(points, n_points, file_path)
    n = len(points)
    step_function = gift_wrapping_step

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
                    # running = False
                    # print('Hull complete!')
                    # run = False
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

def convex_hull_andrews(n_points, points, start_running=False, file_path=None):
    points = init_Points(points, n_points, file_path)
    n = len(points)
    step_function = AndrewsAlgo.andrews_hull_step

    points = np.array(sorted(points, key=lambda x: x[0])) # sort by x value (from left to right)

    # Find the leftmost point to start the algorithm
    current = 0
    top_hull = [current]
    bottom_hull = []
    top = True  # top hull

    running = True
    next_step = False
    run = False
    next = (current + 1) % n

    # Initial draw
    draw(points, top_hull, current_point=current)

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
                    # print('Next step...')
                if event.key == pygame.K_r:
                    run = True

        if next_step or run:
            # print('Computing next step...')
            if top:
                next = step_function(points, top_hull, current, next)
                current = next
                next = (current + 1) % n
            else:
                next = step_function(points, bottom_hull, current, next)
                current = next
                next = current - 1

            if current == len(points) - 1: # reached right side ?
                bottom_hull = [current]
                next = (current - 1) % n
                top = False
            elif current == 0:
                running = False

            # window.blit(text, textRect)
            # pygame.display.update()
            if (not run):
                #    pass
                window.fill(WHITE)
                draw(points, top_hull, current_point=current, next_point=next, clearWindow=False)
                draw(points, bottom_hull, current_point=current, next_point=next, clearWindow=False, hull_color=RED)
                next_step = False if run is not True else True  # Wait for the next key press
            # next_step = True if run else False
    time2 = time.time()
    time_elapsed = time2 - time1

    top_hull.extend(bottom_hull)

    print(
        f'Time: {time_elapsed:.2f} seconds for {n} points.  KiloPoints per seconds : {(n / 1000) / (time_elapsed):.2f}. Hull size: {len(top_hull)}')
    # speeds.append((n,(n/1000)/(time_elapsed)))

    return (n, time_elapsed, len(top_hull))

speeds = []
# Main function to step through the algorithm
def benchmark_convex_hull(algorithm, points, max_points, step_size=1000):
    results = []
    for i in (range(1000, max_points, step_size)):
        if algorithm == "andrews":
            hull_function = convex_hull_andrews
        else:
            hull_function = convex_hull_giftwarpping
       
        n, time_elapsed, hull_size = hull_function(n_points=i, points=points, start_running=True)
        results.append((n, time_elapsed, hull_size))
        speeds.append((n, n / time_elapsed, hull_size))        
    #pygame.quit()
    return (results)

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
algorithm = 'andrews'
points = 'square-ring'
max_points = 500001

#convex_hull_giftwarpping(n_points=300, points=points, start_running=False, file_path=None)
#convex_hull_andrews(n_points=40, points=points, start_running=False, file_path=None)
import itertools
for algo, points in itertools.product(['andrews', 'giftwrapping'], ['circular', 'square', 'ring']):
#for algo, points in itertools.product(['giftwrapping'], ['ring']):
    print(f'Running {algo} for {points} points')
    if algo == "giftwrapping" and points == "ring":
        continue
        speeds = benchmark_convex_hull(algo, points, 10001, 500)
    else:
        speeds = benchmark_convex_hull(algo, points, max_points, 10000)
    with open(f'{algo}_{points}_results.csv', 'w') as file:
        for speed in speeds:
            file.write(f'{speed[0]},{speed[1]},{speed[2]}\n')

#plot_speeds(speeds, algorithm, points)

####file
#points = 'file'
#file_path = 'input100k.txt'
#file_path = 'square_10000.txt'
#convex_hull_giftwarpping(n_points=19, points=points, start_running=True, file_path=file_path)


#tests to run:
# beide algorthmen
# circular und square, ring, square-ring