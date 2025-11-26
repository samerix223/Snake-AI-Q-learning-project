
import pygame
import sys
import random
import math
import pickle
import os

# ----------------------
# settings
# ----------------------
WIDTH = 400
HEIGHT = 400
TILE_SIZE = 20
GRID_WIDTH = WIDTH // TILE_SIZE
GRID_HEIGHT = HEIGHT // TILE_SIZE

FPS_TRAIN = 120   # high speed for training
FPS_PLAY = 15     # slower for viewing

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 200, 0)
DARK_GREEN = (0, 120, 0)
RED = (200, 0, 0)
GRAY = (40, 40, 40)
YELLOW = (200, 200, 0)

UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)
DIRECTIONS = [UP, DOWN, LEFT, RIGHT]

# Q-learning parameters
ALPHA = 0.1
GAMMA = 0.9
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.0002

MAX_STEPS_PER_EPISODE = 1000

QTABLE_FILE = "qtable_snake.pkl"

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake Q-learning AI")
clock = pygame.time.Clock()
font = pygame.font.SysFont("consolas", 18)


# ----------------------
# helpful
# ----------------------
def draw_grid():
    for x in range(0, WIDTH, TILE_SIZE):
        pygame.draw.line(screen, GRAY, (x, 0), (x, HEIGHT))
    for y in range(0, HEIGHT, TILE_SIZE):
        pygame.draw.line(screen, GRAY, (0, y), (WIDTH, y))


def draw_text(text, x, y, color=WHITE):
    img = font.render(text, True, color)
    screen.blit(img, (x, y))


def random_empty_cell(snake_body):
    while True:
        x = random.randint(0, GRID_WIDTH - 1)
        y = random.randint(0, GRID_HEIGHT - 1)
        if (x, y) not in snake_body:
            return (x, y)


def load_qtable():
    if os.path.exists(QTABLE_FILE):
        with open(QTABLE_FILE, "rb") as f:
            try:
                q = pickle.load(f)
                print("Q-table wczytane, rozmiar:", len(q))
                return q
            except Exception as e:
                print("Błąd wczytywania Q-table:", e)
    return {}


def save_qtable(qtable):
    with open(QTABLE_FILE, "wb") as f:
        pickle.dump(qtable, f)
    print("Q-table zapisane. Rozmiar:", len(qtable))


# ----------------------
# GAME CLASSES
# ----------------------
class Snake:
    def __init__(self):
        self.reset()

    def reset(self):
        self.body = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
        self.direction = random.choice(DIRECTIONS)
        self.grow = False

    def head(self):
        return self.body[0]

    def move(self):
        dx, dy = self.direction
        hx, hy = self.head()
        new_head = (hx + dx, hy + dy)

        if self.grow:
            self.body = [new_head] + self.body
            self.grow = False
        else:
            self.body = [new_head] + self.body[:-1]

    def set_direction(self, dir_):
        dx, dy = dir_
        cdx, cdy = self.direction
        if (dx, dy) == (-cdx, -cdy):
            return
        self.direction = dir_

    def collides_with_self(self):
        return self.head() in self.body[1:]

    def collides_with_wall(self):
        hx, hy = self.head()
        return hx < 0 or hx >= GRID_WIDTH or hy < 0 or hy >= GRID_HEIGHT

    def draw(self, surface):
        for i, (x, y) in enumerate(self.body):
            rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            color = GREEN if i == 0 else DARK_GREEN
            pygame.draw.rect(surface, color, rect)


class Food:
    def __init__(self, snake_body):
        self.position = random_empty_cell(snake_body)

    def respawn(self, snake_body):
        self.position = random_empty_cell(snake_body)

    def draw(self, surface):
        x, y = self.position
        rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
        pygame.draw.rect(surface, RED, rect)


# ----------------------
# DEFINITION OF STATE FOR AI
# ----------------------
def is_collision_point(point, snake_body):
    x, y = point
    if x < 0 or x >= GRID_WIDTH or y < 0 or y >= GRID_HEIGHT:
        return True
    if point in snake_body[1:]:
        return True
    return False


def turn_left(direction):
    dx, dy = direction
    # (0,-1)->(-1,0), (-1,0)->(0,1), (0,1)->(1,0), (1,0)->(0,-1)
    if direction == UP:
        return LEFT
    if direction == LEFT:
        return DOWN
    if direction == DOWN:
        return RIGHT
    if direction == RIGHT:
        return UP


def turn_right(direction):
    if direction == UP:
        return RIGHT
    if direction == RIGHT:
        return DOWN
    if direction == DOWN:
        return LEFT
    if direction == LEFT:
        return UP


def get_state(snake, food):
    head = snake.head()
    hx, hy = head
    fx, fy = food.position
    dir_ = snake.direction


    forward = dir_
    left = turn_left(dir_)
    right = turn_right(dir_)


    f_point = (hx + forward[0], hy + forward[1])
    l_point = (hx + left[0], hy + left[1])
    r_point = (hx + right[0], hy + right[1])

    danger_straight = is_collision_point(f_point, snake.body)
    danger_left = is_collision_point(l_point, snake.body)
    danger_right = is_collision_point(r_point, snake.body)


    food_left = fx < hx
    food_right = fx > hx
    food_up = fy < hy
    food_down = fy > hy


    dir_up = (dir_ == UP)
    dir_down = (dir_ == DOWN)
    dir_left = (dir_ == LEFT)
    dir_right = (dir_ == RIGHT)

    # We encode the state as a tuple of boolean/simple values so that it is possible
    state = (
        danger_straight,
        danger_left,
        danger_right,
        food_left,
        food_right,
        food_up,
        food_down,
        dir_up,
        dir_down,
        dir_left,
        dir_right
    )
    return state


# ----------------------
# AI – Q-LEARNING
# ----------------------
def get_q(qtable, state, action):
    return qtable.get((state, action), 0.0)


def set_q(qtable, state, action, value):
    qtable[(state, action)] = value


def choose_action(qtable, state, epsilon):

    if random.random() < epsilon:
        return random.randint(0, 2)
    else:
        qs = [get_q(qtable, state, a) for a in range(3)]
        max_q = max(qs)

        best_actions = [i for i, q in enumerate(qs) if q == max_q]
        return random.choice(best_actions)


def apply_action_to_direction(direction, action):
    # 0 = left, 1 = straight, 2 = right
    if action == 0:
        return turn_left(direction)
    elif action == 2:
        return turn_right(direction)
    else:
        return direction


def distance(a, b):
    (ax, ay) = a
    (bx, by) = b
    return abs(ax - bx) + abs(ay - by)


# ----------------------
# MAIN LOOP – TRAINING + GAME
# ----------------------
def main():
    qtable = load_qtable()

    episode = 0
    epsilon = EPSILON_START

    mode = "train"

    best_score_ever = 0

    while True:
        snake = Snake()
        food = Food(snake.body)
        score = 0
        steps = 0
        done = False


        while not done:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    save_qtable(qtable)
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:

                        if mode == "train":
                            mode = "play"
                        else:
                            mode = "train"
                    if event.key == pygame.K_s:
                        save_qtable(qtable)

            steps += 1


            state = get_state(snake, food)
            if mode == "train":
                action = choose_action(qtable, state, epsilon)
            else:

                action = choose_action(qtable, state, 0.01)


            new_dir = apply_action_to_direction(snake.direction, action)
            snake.set_direction(new_dir)


            old_distance = distance(snake.head(), food.position)


            snake.move()

            reward = -0.1  #
            done = False
            ate_food = False


            if snake.collides_with_wall() or snake.collides_with_self():
                reward = -10.0
                done = True
            else:

                if snake.head() == food.position:
                    snake.grow = True
                    score += 1
                    reward = 10.0
                    ate_food = True
                    food.respawn(snake.body)
                else:

                    new_distance = distance(snake.head(), food.position)
                    if new_distance < old_distance:
                        reward += 0.2
                    else:
                        reward -= 0.2


            if steps > MAX_STEPS_PER_EPISODE:
                done = True


            new_state = get_state(snake, food)

            # update Q
            if mode == "train":
                old_q = get_q(qtable, state, action)
                future_qs = [get_q(qtable, new_state, a) for a in range(3)]
                max_future_q = max(future_qs)
                new_q = old_q + ALPHA * (reward + GAMMA * max_future_q - old_q)
                set_q(qtable, state, action, new_q)

            # drawing
            screen.fill(BLACK)
            draw_grid()
            snake.draw(screen)
            food.draw(screen)

            draw_text(f"Episode: {episode}", 10, 5, WHITE)
            draw_text(f"Score: {score}", 10, 25, WHITE)
            draw_text(f"Mode: {mode}", 10, 45, YELLOW)
            draw_text(f"Epsilon: {epsilon:.2f}", 10, 65, WHITE)
            draw_text("SPACE: train/play, S: save", 10, 85, WHITE)

            if score > best_score_ever:
                best_score_ever = score
            draw_text(f"Best ever: {best_score_ever}", 10, 105, WHITE)

            pygame.display.flip()

            if mode == "train":
                clock.tick(FPS_TRAIN)
            else:
                clock.tick(FPS_PLAY)

            if done:
                # end episode
                if mode == "train":
                    episode += 1

                    if epsilon > EPSILON_MIN:
                        epsilon -= EPSILON_DECAY
                        if epsilon < EPSILON_MIN:
                            epsilon = EPSILON_MIN


                if mode == "train" and episode % 100 == 0:
                    save_qtable(qtable)




if __name__ == "__main__":
    main()
