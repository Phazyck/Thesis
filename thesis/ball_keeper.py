#!/usr/bin/python3

# pylint: disable=wildcard-import
# pylint: disable=too-many-instance-attributes
# pylint: disable=missing-docstring
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
# pylint: disable=too-many-branches
# pylint: disable=too-many-statements

import random as rnd
import math
import MultiNEAT as NEAT
import pygame
from pygame.locals import QUIT, KEYDOWN, KEYUP
from pygame.locals import K_ESCAPE, K_f, K_LEFT, K_RIGHT, K_SPACE
from pygame.color import THECOLORS

import pymunk as pm
from pymunk import Vec2d
from pymunk.pygame_util import draw

import draw_net

import config

_COLLISION_TYPE_WALL = 0
_COLLISION_TYPE_NN = 1
_COLLISION_TYPE_BALL = 2
_COLLISION_TYPE_FLOOR = 3

_MAX_TIMESTEPS = 15000

FEATURE_NAMES = [
    "total_frames_passed",
    "total_collisions_ball_player",
    "total_collisions_ball_wall",
    "total_player_jumps",
    "total_travel_distance_player",
    "total_travel_distance_ball",

    "average_x_position_player",
    "average_x_position_ball",
    "average_y_position_ball",
    "average_distance_ball_player",

    "max_ball_velocity",
    "max_ball_y_position",
    "max_distance_player_ball",
    "final_distance_ball_player",
    
    "player_jumps_per_frame",
    "collisions_ball_player_per_frame",
    "collisions_ball_wall_per_frame",
    "travel_distance_player_per_second",
    "travel_distance_ball_per_second"
]


class _NeuralNetworkAgent(object):

    def __init__(self, space, brain, start_x):
        self.jump_count = 0
        self.startpos = (start_x, 80)
        self.radius = 20
        self.mass = 50000

        self.inertia = pm.moment_for_circle(self.mass, 0, self.radius, (0, 0))
        self.body = pm.Body(self.mass, self.inertia)
        self.shape = pm.Circle(self.body, self.radius)
        self.shape.collision_type = _COLLISION_TYPE_NN
        self.shape.elasticity = 1.0
        self.body.position = self.startpos

        space.add(self.body, self.shape)

        self.body.velocity_limit = 1500

        self.body.velocity = (230, 0)
        self.force = (0, 0)

        self.brain = brain
        self.in_air = False

    def touch_floor(self, *_):
        self.in_air = False
        return True

    def leave_floor(self, *_):
        self.in_air = True

    def jump(self):
        if not self.in_air:
            cur_vel = self.body.velocity
            self.body.velocity = (cur_vel[0], 300)
            self.jump_count += 1

    def move(self, horizontal_movement):
        # if not self.in_air:
        # self.body.force = (horizontal_movement, 0)
        velocity_x = self.body.velocity[0] + horizontal_movement
        max_velocity = 500
        if velocity_x > max_velocity:
            velocity_x = max_velocity
        elif velocity_x < -max_velocity:
            velocity_x = -max_velocity
        self.body.velocity = (velocity_x, self.body.velocity[1])

    def move_left(self):
        self.move(-100)

    def move_right(self):
        self.move(100)

    def move_none(self):
        velocity_x = self.body.velocity[0]
        self.move(- (velocity_x * 0.5))

    def interact(self, ball):
        """
        inputs: x - ball_x, log(ball_y), log(y), ball_vx, ball_vy, in_air, 1
        output: x velocity [-1 .. 1]*const, jump (if > 0.5 )
        """
        inputs = [(self.body.position[0] - ball.body.position[0]) / 300,
                  #                  math.log(ball.body.position[1]),
                  math.log(self.body.position[1]),
                  ball.body.velocity[0] / 300,
                  ball.body.velocity[1] / 300,
                  self.in_air,
                  1.0
                  ]

        self.brain.Input(inputs)
        self.brain.Activate()
        outputs = self.brain.Output()

        horizontal_movement = outputs[0]

        if horizontal_movement < -0.5:
            self.move_left()
        elif horizontal_movement > 0.5:
            self.move_right()
        else:
            self.move_none()

        # self.move(outputs[0] * 500)

        if outputs[1] > 0.5:
            self.jump()


class _Ball(object):

    def __init__(self, space, start_x, start_vx):
        self.mass = 1500
        self.radius = 30
        self.inertia = pm.moment_for_circle(self.mass, 0, self.radius, (0, 0))
        self.body = pm.Body(self.mass, self.inertia)
        self.shape = pm.Circle(self.body, self.radius)
        self.shape.collision_type = _COLLISION_TYPE_BALL
        self.shape.elasticity = 1.0
        self.shape.friction = 0.0
        self.body.position = (start_x, 450)
        space.add(self.body, self.shape)
        self.body.velocity = (start_vx, 0)
        self.body.velocity_limit = 500
        self.in_air = True

        self.wall_collisions = 0
        self.player_collisions = 0

    def touch_floor(self, *_):
        self.in_air = False
        return True

    def leave_floor(self, *_):
        self.in_air = True
        return True

    def collide_wall(self, *_):
        self.wall_collisions += 1
        return True

    def collide_player(self, *_):
        self.player_collisions += 1
        return True


class Average(object):

    def __init__(self):
        self.sum = 0.0
        self.count = 0.0

    def push(self, value):
        self.sum += value
        self.count += 1.0

    def get_average(self):
        average = (self.sum / self.count)
        return average

def _evaluate(genome, space, screen, fast_mode, start_x, start_vx, bot_startx):

    if (not fast_mode) and (screen is None):
        screen = pygame.display.set_mode((600, 600))
        title_string = ("Extracted ball keeper " +
                        "[when in play mode, arrow keys to move player]")
        pygame.display.set_caption(title_string)

    # Setup the environment
    clock = pygame.time.Clock()

    # The agents - the brain and the ball
    net = None
    if genome is not None:
        net = NEAT.NeuralNetwork()
        genome.BuildPhenotype(net)

    agent = _NeuralNetworkAgent(space, net, bot_startx)
    ball = _Ball(space, start_x, start_vx)

    # Feature counting
    total_frames_passed = 0.0
    total_collisions_ball_player = 0.0
    total_collisions_ball_wall = 0.0
    total_player_jumps = 0.0
    total_travel_distance_player = 0.0
    total_travel_distance_ball = 0.0
    average_x_position_player = Average()
    average_x_position_ball = Average()
    average_y_position_ball = Average()
    average_distance_ball_player = Average()
    max_ball_velocity = 0.0
    max_ball_y_position = 0.0
    max_distance_player_ball = 0.0
    final_distance_ball_player = 0.0

    def distance(pos1, pos2):
        return math.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)

    def magnitude(vector):
        return math.sqrt(vector[0]**2 + vector[1]**2)

    def get_position(entity):
        pos = entity.body.position
        return (pos[0], pos[1])

    prev_position_player = get_position(agent)
    prev_position_ball = get_position(ball)

    # Collision Handlers
    space.add_collision_handler(
        _COLLISION_TYPE_NN,
        _COLLISION_TYPE_FLOOR,
        agent.touch_floor,
        None,
        None,
        agent.leave_floor)
    space.add_collision_handler(_COLLISION_TYPE_BALL, _COLLISION_TYPE_FLOOR,
                                ball.touch_floor, None, None, ball.leave_floor)

    space.add_collision_handler(_COLLISION_TYPE_BALL, _COLLISION_TYPE_NN,
                                ball.collide_player, None, None, None)

    space.add_collision_handler(_COLLISION_TYPE_BALL, _COLLISION_TYPE_WALL,
                                ball.collide_wall, None, None, None)

    left_down = False
    right_down = False
    go_left = False

    while total_frames_passed < _MAX_TIMESTEPS:
        total_frames_passed += 1
        do_jump = False
        for event in pygame.event.get():
            if event.type == QUIT:
                exit()
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                exit()
            # NOTE: should we remove this?
            elif event.type == KEYDOWN and event.key == K_f:
                fast_mode = not fast_mode
            elif genome is None:
                if event.type == KEYUP and event.key == K_LEFT:
                    left_down = False
                elif event.type == KEYUP and event.key == K_RIGHT:
                    right_down = False
                elif (event.type == KEYDOWN
                      and event.key == K_LEFT
                      and not fast_mode):
                    left_down = True
                    go_left = True
                elif (event.type == KEYDOWN
                      and event.key == K_RIGHT
                      and not fast_mode):
                    right_down = True
                    go_left = False
                elif (event.type == KEYDOWN
                      and event.key == K_SPACE
                      and not fast_mode):
                    do_jump = True

        # The NN interacts with the world on each 5 timesteps
        #
        if genome is not None:
            if (total_frames_passed % 5) == 0:
                agent.interact(ball)
        else:
            if left_down or right_down:
                if go_left:
                    agent.move_left()
                else:
                    agent.move_right()
            else:
                agent.move_none()

            if do_jump:
                agent.jump()

        # Update physics
        delta_time = 1.0 / 50.0
        space.step(delta_time)

        # Feature calculations

        current = get_position(agent)
        total_travel_distance_player += distance(prev_position_player, current)
        prev_position_player = current

        current = get_position(ball)
        total_travel_distance_ball += distance(prev_position_ball, current)
        prev_position_ball = current

        distance_ball_player = distance(
            prev_position_player, prev_position_ball)
        final_distance_ball_player = distance_ball_player
        average_distance_ball_player.push(distance_ball_player)
        max_distance_player_ball = max(
            max_distance_player_ball,
            distance_ball_player)

        max_ball_velocity = max(
            max_ball_velocity,
            magnitude(ball.body.velocity))

        max_ball_y_position = max(
            max_ball_y_position,
            ball.body.position[1])

        total_player_jumps = agent.jump_count
        total_collisions_ball_player = ball.player_collisions
        total_collisions_ball_wall = ball.wall_collisions

        average_x_position_player.push(agent.body.position[0])
        average_x_position_ball.push(ball.body.position[0])
        average_y_position_ball.push(ball.body.position[1])

        # stopping conditions
        if not ball.in_air:
            break
            
        if ball.body.position[1] < 0:
            break
        
        if not fast_mode:
            if net != None: 
                draw_net.DrawNet(net)
        
            # Draw stuff
            screen.fill(THECOLORS["black"])

            # Draw stuff
            draw(screen, space)

            # Flip screen
            pygame.display.flip()
            clock.tick(50)

    # remove objects from space
    space.remove(agent.shape, agent.body)
    space.remove(ball.shape, ball.body)

    frames = float(total_frames_passed)
    player_jumps_per_frame = float(total_player_jumps) / frames
    collisions_ball_player_per_frame = float(total_collisions_ball_player) / frames
    collisions_ball_wall_per_frame = float(total_collisions_ball_wall) / frames
    travel_distance_player_per_second = float(total_travel_distance_player) / frames
    travel_distance_ball_per_second = float(total_travel_distance_ball) / frames
    
    feature_set = FeatureSet()
    feature_set.insert_many({
        "total_frames_passed": total_frames_passed,
        "total_collisions_ball_player": total_collisions_ball_player,
        "total_collisions_ball_wall": total_collisions_ball_wall,
        "total_player_jumps": total_player_jumps,
        "total_travel_distance_player": total_travel_distance_player,
        "total_travel_distance_ball": total_travel_distance_ball,
        "average_x_position_player": average_x_position_player.get_average(),
        "average_x_position_ball": average_x_position_ball.get_average(),
        "average_y_position_ball": average_y_position_ball.get_average(),
        "average_distance_ball_player":
            average_distance_ball_player.get_average(),
        "max_ball_velocity": max_ball_velocity,
        "max_ball_y_position": max_ball_y_position,
        "max_distance_player_ball": max_distance_player_ball,
        "final_distance_ball_player": final_distance_ball_player,
        "player_jumps_per_frame": player_jumps_per_frame,
        "collisions_ball_player_per_frame": collisions_ball_player_per_frame,
        "collisions_ball_wall_per_frame": collisions_ball_wall_per_frame,
        "travel_distance_player_per_second": travel_distance_player_per_second,
        "travel_distance_ball_per_second": travel_distance_ball_per_second
    })

    feature_set.verify()

    return feature_set


class FeatureSet(object):

    def __init__(self):
        self.feature_map = {}

    def insert(self, key, value):
        feature_map = self.feature_map
        if key in feature_map:
            raise Exception(
                "key '%s' already has a value '%s'" %
                key, str(
                    feature_map[key]))
        else:
            feature_map[key] = value

    def insert_many(self, feature_map):
        for (key, value) in feature_map.items():
            self.insert(key, value)

    def verify(self):
        feature_map = self.feature_map
        for feature_name in FEATURE_NAMES:
            if feature_name not in feature_map:
                raise Exception("key '%s' has not been assigned!")

    def get_features(self, features):
        return tuple(self.feature_map[feature] for feature in features)

    def debug_print(self):
        feature_map = self.feature_map

        width_key = max(len(key) for key in FEATURE_NAMES)
        width_val_b = 2
        width_val_a = max(len(str(int(val)))
                          for val in feature_map.values()) + 1 + width_val_b

        fmt = "%-" + str(width_key) + "s : %" + \
            str(width_val_a) + "." + str(width_val_b) + "f"

        for key in FEATURE_NAMES:
            value = feature_map[key]
            print fmt % (key, value)

    def all_features(self):
        return [self.feature_map[key] for key in FEATURE_NAMES]

def _get_start_x(use_random=False):
    if use_random:
        return rnd.randint(80, 400)
    else:
        return config.ball_start_x

def _get_start_vx(use_random=False):
    if use_random:
        return rnd.randint(-200, 200)
    else:
        return config.ball_start_velocity_x


def _get_bot_start_x(use_random=False):
    if use_random:
        return rnd.randint(80, 400)
    else:
        return config.player_start_x


def evaluate_genome(xxx_todo_changeme, genome, use_random=False):
    (space, screen) = xxx_todo_changeme
    return _evaluate(
        genome,
        space,
        screen,
        True,
        _get_start_x(use_random),
        _get_start_vx(use_random),
        _get_bot_start_x(use_random))


def play_genome(xxx_todo_changeme1, genome, use_random=False):
    (space, screen) = xxx_todo_changeme1
    return _evaluate(
        genome,
        space,
        screen,
        False,
        _get_start_x(use_random),
        _get_start_vx(use_random),
        _get_bot_start_x(use_random))


def play_human(xxx_todo_changeme2, use_random=False):
    (space, screen) = xxx_todo_changeme2
    return _evaluate(
        None,
        space,
        screen,
        False,
        _get_start_x(use_random),
        _get_start_vx(use_random),
        _get_bot_start_x(use_random))


def init(display=False):
    pygame.init()

    screen = None

    if display:
        screen = pygame.display.set_mode((600, 600))
        pygame.display.set_caption(
            "Extracted ball keeper " +
            "[when in play mode, arrow keys to move player]")

    # Physics stuff
    space = pm.Space()
    space.gravity = Vec2d(0.0, -500.0)

    # walls - the left-top-right walls
    body = pm.Body()
    walls = [
        pm.Segment(body, (50, 50), (50, 1550), 10), 
        pm.Segment(body, (50, 1550), (560, 1550), 10), 
        pm.Segment(body, (560, 1550), (560, 50), 10)
    ]

    floor = pm.Segment(body, (50, 50), (560, 50), 10)
    floor.friction = 1.0
    floor.elasticity = 0.0
    floor.collision_type = _COLLISION_TYPE_FLOOR

    for segment in walls:
        segment.friction = 0
        segment.elasticity = 0.99
        segment.collision_type = _COLLISION_TYPE_WALL
    space.add(walls)
    space.add(floor)

    return (space, screen)

def _main():
    game = init()
    while True:
        print "-" * 64
        play_human(game).debug_print()
            

# If run as script, not module
if __name__ == "__main__":
    _main()
