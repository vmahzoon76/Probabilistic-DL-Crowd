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

for count_samples in range(num_samples):
    print(count_samples)
    X = []
    locations = []
    current_velocities = []
    goal = []

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
    locations += [0, 0]
    current_velocities += [0.6*a0_speed * math.cos(a0_angle), 0.6*a0_speed * math.sin(a0_angle)]
    goal += [x_goal, y_goal]
    col_list = ['0_x', '0_y', '1_x', '1_y', '2_x', '2_y', '3_x', '3_y', '4_x', '4_y', '5_x', '5_y', '6_x', '6_y', '7_x',
                '7_y', '8_x', '8_y', '9_x', '9_y', '10_x', '10_y']

    this_sample_locations = pd.DataFrame(columns=col_list)
    this_sample_current_velocities = pd.DataFrame(columns=col_list)
    this_sample_goal = pd.DataFrame(columns=col_list)

    # current_velocities += list(velocity_scale*x for x in sim.getAgentVelocity(a0))
    # pref_velocities += list(velocity_scale*x for x in sim.getAgentPrefVelocity(a0))
    # pref_velocities_normalized += list(normalize_vector(sim.getAgentPrefVelocity(a0)))
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
        current_velocities += [0.6* speed * math.cos(angle), 0.6* speed * math.sin(angle)]
        goal += [x_goal, y_goal]

    this_sample_locations.loc[0] = locations
    this_sample_current_velocities.loc[0] = current_velocities
    this_sample_goal.loc[0] = goal

    initial_state = np.array(X)
    tau_value, max_speed_value, multi_speed, factor_d, factor_s=0.05, 1.4, 1.1, 1, 20
    s = psf.Simulator(
        initial_state, tau_value, max_speed_value, multi_speed, factor_d, factor_s,
        config_file="example.toml")
    # s = psf.Simulator(
    #     initial_state,
    #     config_file="example.toml")

    num_ped = 11
    agents_locations = {}
    count = 0
    for i in range(num_steps):
        s.step()
        count += 1
        array = s.peds.ped_states
        # if (i >= 1 and np.sum(array[i] == array[i - 1]) == num_ped * 7):
        #     break
    print(count)
    # for i in range(count+1):
    #     for k in range(num_ped):
    #         if k not in agents_locations.keys():
    #             agents_locations[k] = {'x': [array[i][k][0]], 'y': [array[i][k][1]]}
    #         else:
    #             agents_locations[k]['x'].append(array[i][k][0])
    #             agents_locations[k]['y'].append(array[i][k][1])
    # width = 10
    # height = 10
    # figure(figsize=(width, height))
    # plt.clf()
    #
    # array_color = ["#FF3030", "#8B1A1A", "#228B22", "#104E8B", "#FF1493", "#9400D3", "#E9967A", "#8B4513", "#20B2AA",
    #              "#000080", "#FF6103"]
    # #plot each agent's path. Plot a circle at their starting position
    # for key in agents_locations.keys():
    #    plt.plot(agents_locations[key]['x'], agents_locations[key]['y'], linestyle='--', marker='.', label=key,
    #             color=array_color[key])
    #    plt.plot(array[0][key][0], array[0][key][1], marker="x", markersize=10, color=array_color[key])
    #    circle = plt.Circle((array[0][key][4], array[0][key][5]), 0.02, color=array_color[key])
    #    plt.gca().add_patch(circle)
    #    plt.savefig("images/image_new/test{}".format(count_samples), dpi=500)

    # for step in range(1, count + 1):
    locations = []
    current_velocities = []
    goal = []
    for agent_no in range(num_neighbors + 1):
        locations += list(array[-1][agent_no][0:2])
        current_velocities += list(array[-1][agent_no][0:2]/0.4)
        goal += list(array[-1][agent_no][4:6])

    this_sample_locations.loc[1] = locations
    this_sample_current_velocities.loc[1] = current_velocities
    this_sample_goal.loc[1] = goal

    samples.append([this_sample_locations, this_sample_current_velocities, this_sample_goal])
save_file_name="sfm_god/test"
with open(save_file_name + '.pkl', 'wb') as handle:
    pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("hi")

