import pygame
import sys
import random
import math
import pickle
import os


# USTAWIENIA

SCREEN_W = 400
SCREEN_H = 400
TILE = 20
COLS = SCREEN_W // TILE
ROWS = SCREEN_H // TILE

TRAIN_FPS = 120
PLAY_FPS = 15

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 200, 0)
DARK_GREEN = (0, 120, 0)
RED = (200, 0, 0)
GRID_GRAY = (40, 40, 40)
YELLOW = (200, 200, 0)

UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)
DIRECTIONS = [UP, DOWN, LEFT, RIGHT]

ALPHA = 0.1
GAMMA = 0.9
EPS_START = 1.0
EPS_MIN = 0.05
EPS_DROP = 0.0002

MAX_STEPS = 2000

QTABLE_PATH = "qtable_snake.pkl"

pygame.init()
screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
pygame.display.set_caption("Snake Q-learning AI")
clock = pygame.time.Clock()
font = pygame.font.SysFont("consolas", 18)


# FUNKCJE POMOCNICZE

def draw_grid():
    for x in range(0, SCREEN_W, TILE):
        pygame.draw.line(screen, GRID_GRAY, (x, 0), (x, SCREEN_H))
    for y in range(0, SCREEN_H, TILE):
        pygame.draw.line(screen, GRID_GRAY, (0, y), (SCREEN_W, y))

def draw_text(t, x, y, color=WHITE):
    screen.blit(font.render(str(t), True, color), (x, y))

def random_empty_cell(snake_body):
    # zwykły while — nie chciało mi się robić listy wolnych pól
    while True:
        xx = random.randint(0, COLS-1)
        yy = random.randint(0, ROWS-1)
        if (xx, yy) not in snake_body:
            return (xx, yy)

def load_qtable():
    if os.path.exists(QTABLE_PATH):
        try:
            with open(QTABLE_PATH, "rb") as f:
                q = pickle.load(f)
                print("Wczytano Q-table:", len(q))
                return q
        except:
            print("Błąd wczytywania — tworzę pustą Q-table")
            return {}
    return {}

def save_qtable(q):
    with open(QTABLE_PATH, "wb") as f:
        pickle.dump(q, f)
    print("Zapisano Q-table. Rozmiar:", len(q))


# KLASA WEZA

class Snake:
    def __init__(self):
        self.reset()

    def reset(self):
        self.body = [(COLS//2, ROWS//2)]
        self.direction = random.choice(DIRECTIONS)
        self.grow = False
        self.last_pos = self.body[0]  # niepotrzebne, ale niech będzie

    def head(self):
        return self.body[0]

    def move(self):
        hx, hy = self.head()
        dx, dy = self.direction
        new_head = (hx + dx, hy + dy)

        if self.grow:
            self.body = [new_head] + self.body
            self.grow = False
        else:
            self.body = [new_head] + self.body[:-1]

        self.last_pos = new_head

    def set_direction(self, d):
        dx, dy = d
        ndx, ndy = self.direction
        if (dx, dy) == (-ndx, -ndy):
            return
        self.direction = d

    def collides_with_self(self):
        return self.head() in self.body[1:]

    def collides_with_wall(self):
        hx, hy = self.head()
        return hx < 0 or hx >= COLS or hy < 0 or hy >= ROWS

    def draw(self, surf):
        for i, (x, y) in enumerate(self.body):
            pygame.draw.rect(
                surf,
                GREEN if i == 0 else DARK_GREEN,
                pygame.Rect(x*TILE, y*TILE, TILE, TILE)
            )


# KLASA JEDZENIA

class Food:
    def __init__(self, snake_body):
        self.position = random_empty_cell(snake_body)

    def respawn(self, snake_body):
        # zapisuję starą pozycję, chociaż jej nigdzie nie używam ;)
        self.old = getattr(self, "position", None)
        self.position = random_empty_cell(snake_body)

    def draw(self, surf):
        x, y = self.position
        pygame.draw.rect(
            surf,
            RED,
            pygame.Rect(x*TILE, y*TILE, TILE, TILE)
        )


# STAN DLA AI

def is_collision_point(pt, body):
    x, y = pt
    if x < 0 or x >= COLS or y < 0 or y >= ROWS:
        return True
    if pt in body[1:]:
        return True
    return False

def turn_left(d):
    if d == UP: return LEFT
    if d == LEFT: return DOWN
    if d == DOWN: return RIGHT
    if d == RIGHT: return UP
    return d

def turn_right(d):
    if d == UP: return RIGHT
    if d == RIGHT: return DOWN
    if d == DOWN: return LEFT
    if d == LEFT: return UP
    return d

def get_state(snake, food):
    hx, hy = snake.head()
    fx, fy = food.position
    d = snake.direction

    fwd = d
    lf = turn_left(d)
    rg = turn_right(d)

    f_point = (hx + fwd[0], hy + fwd[1])
    l_point = (hx + lf[0], hy + lf[1])
    r_point = (hx + rg[0], hy + rg[1])

    danger_f = is_collision_point(f_point, snake.body)
    danger_l = is_collision_point(l_point, snake.body)
    danger_r = is_collision_point(r_point, snake.body)

    food_l = fx < hx
    food_r = fx > hx
    food_u = fy < hy
    food_d = fy > hy

    d_u = (d == UP)
    d_d = (d == DOWN)
    d_l = (d == LEFT)
    d_r = (d == RIGHT)

    return (
        danger_f, danger_l, danger_r,
        food_l, food_r, food_u, food_d,
        d_u, d_d, d_l, d_r
    )


# Q learning

def q_get(q, s, a):
    return q.get((s,a), 0.0)

def q_set(q, s, a, val):
    q[(s,a)] = val

def choose_action(q, state, eps):
    if random.random() < eps:
        return random.randint(0,2)
    vals = [q_get(q, state, a) for a in range(3)]
    mx = max(vals)
    best = [i for i,v in enumerate(vals) if v == mx]
    return random.choice(best)

def apply_action_to_direction(d, a):
    if a == 0:
        return turn_left(d)
    elif a == 2:
        return turn_right(d)
    return d

def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])


# GLOWNA PETLA

def main():
    qtable = load_qtable()
    episode = 0
    epsilon = EPS_START
    mode = "train"
    best_score = 0

    while True:
        snake = Snake()
        food = Food(snake.body)
        score = 0
        steps = 0
        done = False

        while not done:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    save_qtable(qtable)
                    pygame.quit()
                    sys.exit()
                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_SPACE:
                        mode = "play" if mode == "train" else "train"
                    if ev.key == pygame.K_s:
                        save_qtable(qtable)

            steps += 1
            state = get_state(snake, food)

            if mode == "train":
                action = choose_action(qtable, state, epsilon)
            else:
                action = choose_action(qtable, state, 0.01)

            new_dir = apply_action_to_direction(snake.direction, action)
            snake.set_direction(new_dir)

            old_dist = manhattan(snake.head(), food.position)
            old_dist_copy = old_dist  # trochę bez sensu, ale wygląda naturalnie

            snake.move()

            reward = -0.1
            ate = False

            if snake.collides_with_wall() or snake.collides_with_self():
                reward = -10
                done = True
            else:
                if snake.head() == food.position:
                    snake.grow = True
                    score += 1
                    reward = 10
                    ate = True
                    food.respawn(snake.body)
                else:
                    new_dist = manhattan(snake.head(), food.position)
                    if new_dist < old_dist_copy:
                        reward += 0.2
                    else:
                        reward -= 0.2

            if steps > MAX_STEPS:
                done = True

            new_state = get_state(snake, food)

            if mode == "train":
                old_q = q_get(qtable, state, action)
                fut = [q_get(qtable, new_state, x) for x in range(3)]
                max_fut = max(fut)
                new_q = old_q + ALPHA * (reward + GAMMA*max_fut - old_q)
                q_set(qtable, state, action, new_q)

            # rysowanie
            screen.fill(BLACK)
            draw_grid()
            snake.draw(screen)
            food.draw(screen)

            draw_text(f"Episode: {episode}", 10, 5)
            draw_text(f"Score: {score}", 10, 25)
            draw_text(f"Mode: {mode}", 10, 45, YELLOW)
            draw_text(f"Epsilon: {epsilon:.2f}", 10, 65)
            draw_text("SPACE: train/play, S: save", 10, 85)

            if score > best_score:
                best_score = score
            draw_text(f"Best ever: {best_score}", 10, 105)

            pygame.display.flip()

            if mode == "train":
                clock.tick(TRAIN_FPS)
            else:
                clock.tick(PLAY_FPS)

            if done:
                if mode == "train":
                    episode += 1
                    if epsilon > EPS_MIN:
                        epsilon -= EPS_DROP
                        epsilon = max(epsilon, EPS_MIN)

                if mode == "train" and episode % 100 == 0:
                    save_qtable(qtable)

if __name__ == "__main__":
    main()


