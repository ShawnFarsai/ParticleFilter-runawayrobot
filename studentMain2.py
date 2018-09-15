
from robot import *  # Check the robot.py tab to see how this works.
from math import *
from matrix import *  # Check the matrix.py tab to see how this works.
import random


def initialize_particles(measurement):
    particles = []
    for i in range(50):
        # x = random.uniform(-20, 20)
        # y = random.uniform(-20, 20)
        x = random.uniform(measurement[0] - 2, measurement[0] + 2)
        y = random.uniform(measurement[1] - 2, measurement[1] + 2)
        # heading = random.uniform(-1 * pi, pi)
        heading = 0.5
        turning = 2*pi / 34.0
        distance = 1.5
        new_robot = robot(x, y, heading, turning, distance)
        new_robot.set_noise(0, 0, measurement_noise)
        particles.append(new_robot)

    return particles


def initializeData(measurement):
    particles = []
    averageHeading = 0
    averageTurning = 0
    averageDistance = 0
    first = True
    for i in range(500):
        x = random.uniform(measurement[0] - 2, measurement[0] + 2)
        y = random.uniform(measurement[1] - 2, measurement[1] + 2)
        heading = 0.5
        turning = 2*pi / 34.0
        distance = 1.5
        new_robot = robot(x, y, heading, turning, distance)
        new_robot.set_noise(0, 0, measurement_noise)
        particles.append(new_robot)

    data = {
        "particles": particles,
        "averageHeading": averageHeading,
        "averageTurning": averageTurning,
        "averageDistance": averageDistance,
        "first": first
    }

    return data


def initializeParticles(OTHER, measurement):
    particles = []
    avgTurn = 0
    avgDistance = 0

    distances = []
    headings = []
    turns = []
    for i in range(1, len(OTHER)):
        p1 = (OTHER[i][0], OTHER[i][1])
        p2 = (OTHER[i-1][0], OTHER[i-1][1])
        distances.append(distance_between(p1, p2))
        heading = atan2(p1[1] - p2[1], p1[0] - p2[0])
        headings.append(heading)

        if i > 1:
            turns.append(headings[i-1] - headings[i-2])

    avgDistance = sum(distances) / len(distances)
    avgTurn = sum(turns) / len(turns)
    print(avgTurn, avgDistance)
    predictedHeading = angle_trunc(headings[-1] + avgTurn)

    for i in range(500):
        x = random.uniform(measurement[0] - 2, measurement[0] + 2)
        y = random.uniform(measurement[1] - 2, measurement[1] + 2)
        heading = headings[-1]
        turning = avgTurn
        distance = avgDistance
        new_robot = robot(x, y, heading, turning, distance)
        new_robot.set_noise(0, 0, measurement_noise)
        particles.append(new_robot)

    return {
        'particles': particles,
        'first': True
    }


def Gaussian(mu, sigma, x):

    # calculates the probability of x for 1-dim Gaussian with mean mu and var. sigma
    return exp(- ((mu - x) ** 2) / (sigma ** 2) / 2.0) / sqrt(2.0 * pi * (sigma ** 2))


def measurement_prob(measurement_coord, particle):
    prob = 1.0
    predicted_measurements = particle.sense()

    prob *= Gaussian(predicted_measurements[0],
                     particle.measurement_noise, measurement_coord[0])
    prob *= Gaussian(predicted_measurements[1],
                     particle.measurement_noise, measurement_coord[1])

    return prob


def get_position(particles):
    x = 0.0
    y = 0.0
    bearing = 0.0
    for p in particles:
        x += p.x
        y += p.y
    return (x / len(particles), y / len(particles))


def getAvgs(particles):
    x = 0.0
    y = 0.0
    heading = 0.0
    turning = 0.0
    distance = 0.0
    N = len(particles)
    for p in particles:
        x += p.x
        y += p.y
        heading += p.heading
        turning += p.turning
        distance += p.distance
    return [x / N, y / N, heading / N, turning / N, distance / N]


def estimate_next_pos(measurement, OTHER=None):
    """Estimate the next (x, y) position of the wandering Traxbot
    based on noisy (x, y) measurements."""

    if not OTHER:
        # OTHER = initializeData(measurement)
        OTHER = []
        OTHER.append(measurement)
    elif (type(OTHER) == list) & (len(OTHER) <= 10):
        OTHER.append(measurement)
    elif (type(OTHER) == list) & (len(OTHER) == 11):
        OTHER = initializeParticles(OTHER, measurement)
    else:
        particles = OTHER["particles"]
        N = len(particles)

        if not OTHER['first']:
            for p in particles:
                p.move(p.turning, p.distance)
        else:
            OTHER['first'] = False

        # weights
        w = []
        for p in particles:
            w.append(measurement_prob(measurement, p))

        # normalize weights
        w2 = []
        s = sum(w)
        for i in w:
            w2.append(i/s)
        w = w2

        # resampling
        resample = []
        index = int(random.random() * N)
        beta = 0.0
        mw = max(w)
        for i in range(N):
            beta += random.random() * 2.0 * mw
            while beta > w[index]:
                beta -= w[index]
                index = (index + 1) % N
            new_particle = robot(particles[index].x, particles[index].y,
                                 particles[index].heading + random.gauss(0, measurement_noise), particles[index].turning, particles[index].distance)
            new_particle.set_noise(0, 0, measurement_noise)
            resample.append(new_particle)

        particles = resample

        averageXY = get_position(particles)
        a = getAvgs(particles)
        estimate_particle = robot(a[0], a[1], a[2], a[3], a[4])
        # estimate_particle.set_noise(0, 0, measurement_noise)
        estimate_particle.move(estimate_particle.turning,
                               estimate_particle.distance)

        xy_estimate = (estimate_particle.x, estimate_particle.y)
        OTHER["particles"] = particles
        return xy_estimate, OTHER
    xy_estimate = (0, 0)
    return xy_estimate, OTHER

# A helper function you may find useful.


def distance_between(point1, point2):
    """Computes distance between point1 and point2. Points are (x, y) pairs."""
    x1, y1 = point1
    x2, y2 = point2
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# This is here to give you a sense for how we will be running and grading
# your code. Note that the OTHER variable allows you to store any
# information that you want.


def demo_grading(estimate_next_pos_fcn, target_bot, OTHER=None):
    localized = False
    distance_tolerance = 0.01 * target_bot.distance
    ctr = 0
    # if you haven't localized the target bot, make a guess about the next
    # position, then we move the bot and compare your guess to the true
    # next position. When you are close enough, we stop checking.
    while not localized and ctr <= 1000:
        ctr += 1
        print ctr
        measurement = target_bot.sense()
        position_guess, OTHER = estimate_next_pos_fcn(measurement, OTHER)
        target_bot.move_in_circle()
        true_position = (target_bot.x, target_bot.y)
        error = distance_between(position_guess, true_position)
        if error <= distance_tolerance:
            print "You got it right! It took you ", ctr, " steps to localize."
            localized = True
        if ctr == 1000:
            print "Sorry, it took you too many steps to localize the target."
    return localized

# def demo_grading(estimate_next_pos_fcn, target_bot, OTHER=None):
#     localized = False
#     distance_tolerance = 0.01 * target_bot.distance
#     ctr = 0
#     # if you haven't localized the target bot, make a guess about the next
#     # position, then we move the bot and compare your guess to the true
#     # next position. When you are close enough, we stop checking.
#     # For Visualization
#     import turtle  # You need to run this locally to use the turtle module
#     window = turtle.Screen()
#     window.bgcolor('white')
#     size_multiplier = 25.0  # change Size of animation
#     broken_robot = turtle.Turtle()
#     broken_robot.shape('turtle')
#     broken_robot.color('green')
#     broken_robot.resizemode('user')
#     broken_robot.shapesize(0.1, 0.1, 0.1)
#     measured_broken_robot = turtle.Turtle()
#     measured_broken_robot.shape('circle')
#     measured_broken_robot.color('red')
#     measured_broken_robot.resizemode('user')
#     measured_broken_robot.shapesize(0.1, 0.1, 0.1)
#     prediction = turtle.Turtle()
#     prediction.shape('arrow')
#     prediction.color('blue')
#     prediction.resizemode('user')
#     prediction.shapesize(0.1, 0.1, 0.1)
#     prediction.penup()
#     broken_robot.penup()
#     measured_broken_robot.penup()
#     # End of Visualization
#     while not localized and ctr <= 10000:
#         ctr += 1
#         print ctr
#         measurement = target_bot.sense()
#         position_guess, OTHER = estimate_next_pos_fcn(measurement, OTHER)
#         target_bot.move_in_circle()
#         true_position = (target_bot.x, target_bot.y)
#         error = distance_between(position_guess, true_position)
#         if error <= distance_tolerance:
#             print "You got it right! It took you ", ctr, " steps to localize."
#             localized = True
#         if ctr == 1000:
#             print "Sorry, it took you too many steps to localize the target."
#         # More Visualization
#         measured_broken_robot.setheading(target_bot.heading*180/pi)
#         measured_broken_robot.goto(
#             measurement[0]*size_multiplier, measurement[1]*size_multiplier-200)
#         measured_broken_robot.stamp()
#         broken_robot.setheading(target_bot.heading*180/pi)
#         broken_robot.goto(target_bot.x*size_multiplier,
#                           target_bot.y*size_multiplier-200)
#         broken_robot.stamp()
#         prediction.setheading(target_bot.heading*180/pi)
#         prediction.goto(
#             position_guess[0]*size_multiplier, position_guess[1]*size_multiplier-200)
#         prediction.stamp()
#         # End of Visualization
#     return localized

# This is a demo for what a strategy could look like. This one isn't very good.


def naive_next_pos(measurement, OTHER=None):
    """This strategy records the first reported position of the target and
    assumes that eventually the target bot will eventually return to that
    position, so it always guesses that the first position will be the next."""
    if not OTHER:  # this is the first measurement
        OTHER = measurement
    xy_estimate = OTHER
    return xy_estimate, OTHER


# This is how we create a target bot. Check the robot.py file to understand
# How the robot class behaves.
test_target = robot(2.1, 4.3, 0.5, 2*pi / 34.0, 1.5)
measurement_noise = 0.05 * test_target.distance
test_target.set_noise(0.0, 0.0, measurement_noise)
demo_grading(estimate_next_pos, test_target)
