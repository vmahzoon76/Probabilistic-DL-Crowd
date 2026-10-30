from pathlib import Path
import numpy as np
import pysocialforce as psf
import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pickle
from datetime import datetime
from scipy.spatial import distance
from lidar import lidar


def no_overlap(new_point, current_points_list, threshold):
    for i in range(0, len(current_points_list), 2):
        this_x = current_points_list[i]
        this_y = current_points_list[i + 1]
        if distance.euclidean(new_point, [this_x, this_y]) < threshold:
            return False

    return True


start_time = datetime.now()
samples = []
# Model_based
initial_space_padding = 0.1
AgentNeighborDist = 8
agent_radius = 0.2

# set number of data samples you want to generate
num_samples =100000

num_neighbors = 10

# set number of steps you want in your simluation (this value includes the initialized step)
num_steps = 4
# set min/max number of agents in a simulation
neighbors_range = [3, 10]
# set min/max speed values for all agents
speed_range = [1.5, 3.5]
# set min/max direction values for all agents
direction_range = [-math.pi, math.pi]
# set min/max locations for goal locations
goal_x_range = [-10, 10]
goal_y_range = [-10, 10]
data=[[],[]] # (#num_samples,722)
for count_samples in range(num_samples):
    if count_samples % 1000 ==0:
        print(count_samples)
    locations = []
    locations+=[0,0]
    goal=[]
    current_velocities=[]
    X=[]
    goal=[]

    a0_speed = random.uniform(speed_range[0], speed_range[1])
    a0_angle = random.uniform(direction_range[0], direction_range[1])
    if count_samples%200==0:
        valid=-1
        while valid==-1:
            x_goal = random.uniform(-0.3, 0.3)
            y_goal = random.uniform(-0.3, 0.3)
            if math.sqrt(x_goal**2+y_goal**2)<=0.3:
                valid=1
    else:
        x_goal = random.uniform(-10, 10)
        y_goal = random.uniform(-10, 10)

    X.append([0, 0, 0.6* a0_speed * math.cos(a0_angle),  0.6*a0_speed * math.sin(a0_angle), x_goal, y_goal])
    v=[ 0.6* a0_speed * math.cos(a0_angle),  0.6*a0_speed * math.sin(a0_angle)]   #v(t) for center agent
    goal+=[x_goal,y_goal]





    for neighbor in range(1, num_neighbors + 1):
        valid = -1
        while valid == -1:
            # generate random speed, angle, and starting location
            dist_from_a0 = random.random() * AgentNeighborDist
            theta_from_a0 = random.random() * 2 * math.pi
            x = dist_from_a0 * math.cos(theta_from_a0)
            y = dist_from_a0 * math.sin(theta_from_a0)

            if no_overlap([x, y], locations, 2 * agent_radius + initial_space_padding):
                valid = 1

        speed = random.uniform(speed_range[0], speed_range[1])
        angle = random.uniform(direction_range[0], direction_range[1])
        x_goal = random.uniform(-10, 10)
        y_goal = random.uniform(-10, 10)
        X.append([x, y, 0.6*speed * math.cos(angle), 0.6*speed * math.sin(angle), x_goal, y_goal])
        locations += [x, y]
        goal += [x_goal, y_goal]
        current_velocities += [0.6 * speed * math.cos(angle), 0.6 * speed * math.sin(angle)]

    Range = 10
    Angle = 0.5
    agent_radius = 0.2

    # lidar data at time t
    locations=locations[2:]
    centers=[locations[i:i+2] for i in range(0, len(locations), 2)]     #make data in the way that lidar can work [[,],[,],...,[,]]
    l1 = lidar(Range, Angle, centers, agent_radius)   # create object
    l1.sense_obstacles()     #get data
 
    #l1.plot()

    #for lidar only at center at time t

    # p_locations=list(np.array(locations)-np.array(current_velocities)*0.1)
    # centers = [p_locations[i:i + 2] for i in range(0, len(p_locations), 2)]
    # l2 = lidar(Range, Angle, centers, agent_radius)
    # l2.sense_obstacles()
    # plt.title("s")
    #l2.plot()

    #lidar at center agent at time t-1
    p_locations = list(np.array(locations) - np.array(current_velocities) * 0.4)   # going back to time t-1
    centers=[list(np.array(p_locations[i:i+2])+0.4*np.array(v)) for i in range(0, len(p_locations), 2)]    # off-seting (put center agent at origin and offset other agents)
    l2 = lidar(Range, Angle, centers, agent_radius)
    l2.sense_obstacles()
    #l2.plot()





    data[0].append(l2.scan_array+l1.scan_array+v+goal[0:2])     #data that I want to save as x





    initial_state = np.array(X)
    tau_value, max_speed_value, multi_speed, factor_d, factor_s = 0.05, 1.4, 1.1, 1, 20
    s = psf.Simulator(
        initial_state, tau_value, max_speed_value, multi_speed, factor_d, factor_s,
        config_file="example.toml")

    num_ped = 11
    agents_locations = {}
    count = 0
    for i in range(num_steps):
        s.step()
        count += 1
        array = s.peds.ped_states

    data[1].append(list(array[-1][0][0:2]/0.4))


#
save_file_name="val"
with open(save_file_name + '.pkl', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("hi")



# data looks like [X,Y], where X is the list of inputs and Y is the list of output
