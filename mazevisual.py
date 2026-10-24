import numpy as np
import pygame
from collections import deque
ROWS, COLS = 10, 10
CELL = 60
go = np.array([
    (0, 1),   # Right
    (1, 0),   # Down
    (0, -1),  # Left
    (-1, 0)   # Up
])
def solvable(maze):
    queue = deque([(0, 0)])
    visited = {(0, 0)}
    while queue:
        y, x = queue.popleft()
        if (y, x) == (ROWS - 1, COLS - 1):
            return True
        for dy, dx in go:
            v, u = y + dy, x + dx
            if (0 <= v < ROWS and 0 <= u < COLS
                    and maze[v][u] != '#'
                    and (v, u) not in visited):
                visited.add((v, u))
                queue.append((v, u))
    return False
def generate_maze():
    while True:
        maze = np.random.choice(
            ['.', '#'],
            size=(ROWS, COLS),
            p=[0.65, 0.35]
        )
        maze[0, 0] = 'S'
        maze[ROWS - 1, COLS - 1] = 'G'
        if solvable(maze):
            return maze
def train(maze):
    Q = np.zeros((ROWS, COLS, 4))
    alpha = 0.1
    gamma = 0.95
    epsilon = 1.0
    for episode in range(10000):
        y, x = 0, 0
        for step in range(300):
            if np.random.random() < epsilon:
                a = np.random.randint(4)
            else:
                a = np.argmax(Q[y, x])
            v, u = np.array([y, x]) + go[a]
            if (v < 0 or v >= ROWS or
                u < 0 or u >= COLS or
                maze[v][u] == '#'):
                v, u = y, x
                reward = -5
            elif maze[v][u] == 'G':
                reward = 100
            else:
                reward = -1
            done = maze[v][u] == 'G'
            target = reward
            if not done:
                target += gamma * np.max(Q[v, u])
            Q[y, x, a] += alpha * (target - Q[y, x, a])
            y, x = v, u
            if done:
                break
        epsilon = max(0.05, epsilon * 0.999)
    return Q
def create_model():
    maze = generate_maze()
    Q = train(maze)
    return maze, Q
pygame.init()
screen = pygame.display.set_mode((COLS * CELL, ROWS * CELL))
pygame.display.set_caption("Random Maze AI Solver")
clock = pygame.time.Clock()
maze, Q = create_model()
y, x = 0, 0
visited = {(0, 0)}
steps = 0
finished = False
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                maze, Q = create_model()
                y, x = 0, 0
                visited = {(0, 0)}
                steps = 0
                finished = False
    screen.fill((255, 255, 255))
    for r in range(ROWS):
        for c in range(COLS):
            rect = pygame.Rect(c * CELL, r * CELL,CELL, CELL)
            if maze[r][c] == '#':
                color = (40, 40, 40)
            elif maze[r][c] == 'G':
                color = (0, 200, 0)
            elif (r, c) in visited:
                color = (170, 210, 255)
            else:
                color = (255, 255, 255)
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, (150, 150, 150), rect, 1)
    pygame.draw.circle(screen,(255, 50, 50),(x * CELL + CELL // 2, y * CELL + CELL // 2),CELL // 3)
    pygame.display.flip()
    if not finished:
        a = np.argmax(Q[y, x])
        v, u = np.array([y, x]) + go[a]
        if (v < 0 or v >= ROWS or
            u < 0 or u >= COLS or
            maze[v][u] == '#' or
            (v, u) in visited):
            print("AI got stuck. Press R to retry.")
            finished = True
        else:
            y, x = v, u
            visited.add((y, x))
            steps += 1
            if maze[y][x] == 'G':
                print("Goal reached in", steps, "steps!")
                finished = True
    clock.tick(5)
pygame.quit()