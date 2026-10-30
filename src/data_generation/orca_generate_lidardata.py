# necessary imports
import rvo2
import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib as mpl
import pickle
from datetime import datetime
from lidar import lidar


def normalize_vector(vector):
    norm_vec = vector / np.sqrt(sum(np.array(vector) ** 2))
    if np.isnan(norm_vec).any():
        norm_vec[np.isnan(norm_vec)] = 0.0
    return norm_vec


'''
def angle_between_vectors(v1, v2):
    v1_u = normalize_vector(v1)
    v2_u = normalize_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
'''


def within_range(value, valid_range):
    if -valid_range <= value <= valid_range:
        return True
    else:
        return False


def no_overlap(new_point, current_points_list, threshold):
    for i in range(0, len(current_points_list), 2):
        this_x = current_points_list[i]
        this_y = current_points_list[i + 1]
        if math.dist(new_point, [this_x, this_y]) < threshold:
            return False

    return True


def gaussian_noise(sigma):
    return np.random.normal(0.0, sigma)


def laplace_noise(sigma):
    return np.random.laplace(0.0, np.sqrt((sigma ** 2) / 2))


def add_masked_agents(data_list, masked_agents):
    data_list += masked_agents
    return data_list


def update_df(df, step, data_list):
    df.loc[step] = data_list
    return df


def magnitude(vector):
    return math.sqrt(sum(pow(element, 2) for element in vector))


def calculate_pref_vel(sim, pref_speed, goal_points, goal_threshold):
    for neighbor in range(0, sim.getNumAgents()):
        goal_x = goal_points[neighbor * 2]
        goal_y = goal_points[neighbor * 2 + 1]
        if math.dist(list(sim.getAgentPosition(neighbor)), [goal_x, goal_y]) < goal_threshold:
            sim.setAgentPrefVelocity(neighbor, (0.0, 0.0))
        else:
            #print("hi")
            sim.setAgentPrefVelocity(neighbor, tuple(pref_speed * normalize_vector(
                [goal_i - curr_i for goal_i, curr_i in zip([goal_x, goal_y], list(sim.getAgentPosition(neighbor)))])))


def calculate_pref_vel_one_agent(sim, agent_no, pref_speed, goal_points, goal_threshold):
    goal_x = goal_points[0]
    goal_y = goal_points[1]
    if math.dist(list(sim.getAgentPosition(agent_no)), [goal_x, goal_y]) < goal_threshold:
        sim.setAgentPrefVelocity(agent_no, (0.0, 0.0))
    else:
        sim.setAgentPrefVelocity(agent_no, tuple(pref_speed * normalize_vector(
            [goal_i - curr_i for goal_i, curr_i in zip([goal_x, goal_y], list(sim.getAgentPosition(agent_no)))])))


def all_agents_done(sim, goal_points, goal_threshold):
    for neighbor in range(0, sim.getNumAgents()):
        goal_x = goal_points[neighbor * 2]
        goal_y = goal_points[neighbor * 2 + 1]
        if not (math.dist(list(sim.getAgentPosition(neighbor)), [goal_x, goal_y]) < goal_threshold):
            return False
    return True


# set number of data samples you want to generate
num_samples = 100000

# set number of steps you want in your simluation (this value includes the initialized step)
num_steps = 2
agent_radius = 0.2

# set min/max number of agents in a simulation
neighbors_range = [3, 10]
# set min/max speed values for all agents
speed_range = [0.0, 1.8]
# set min/max direction values for all agents
direction_range = [-math.pi, math.pi]

# set sigma noise values for preferred velocity
speed_noise_range = 0.5
direction_noise_range = math.pi / 6

# type of noise distribution
noise_type = 'gaussian'

goal_dist_range = [0,10]  # 10]

pref_speed = 1.3

# set value for minimum distance between agents' starting positions from each other
initial_space_padding = 0.1

velocity_scale = 0.3

data_split = "train"

plot_samples = False

repeat_for = 7

for i_repeat_for in range(repeat_for):
    start_time = datetime.now()
    save_file_name= save_file_name = "lidar_ORCA_baseline_" + data_split + "_" + str(num_samples) + "_" + str(i_repeat_for)
    data=[[],[]]
    for i in range(num_samples):
        # set up the simulator
        '''
        Hyperparameters are set to whatever is listed in RVO2 paper
            - timeStep = 0.10 (we run 4 x 0.1 timesteps to make 0.4)
            - neighborDist = 5
            - MaxNeighbors = 10
            - AgentTimeHorizon = 5
            - AgentTimeHorizonObst = 1
            - AgentRadius = 0.2
            - AgentMaxSpeed = 1.3
        '''
        
        # 4/22/2024 baseline parameters (from gridsearch)
        pref_speed = 1.3
        dt = 0.4
        goal_threshold = 0.2*2
        sim = rvo2.PyRVOSimulator(0.10, 5, 10, 5, 1, 0.2, pref_speed+0.5)
    
    
    
        locations = []
        current_velocities = []
        goals = []
        pref_velocities = []
    
        # pick random number of neighbors
        num_neighbors = 10  # random.randint(neighbors_range[0], neighbors_range[1])
    
        # put agent 0 at the origin. give a random speed and angle for initial velocity
        a0_speed = random.uniform(speed_range[0], speed_range[1])
        a0_angle = random.uniform(direction_range[0], direction_range[1])
        a0 = sim.addAgent((0, 0))
        sim.setAgentVelocity(a0, (a0_speed * math.cos(a0_angle), a0_speed * math.sin(a0_angle)))
    
        # Uncomment to randomly initialize preferred velocity. Note that we set a preferred speed so we just need random direction
        a0_angle = random.uniform(direction_range[0], direction_range[1])
        sim.setAgentPrefVelocity(a0, (pref_speed * math.cos(a0_angle), pref_speed * math.sin(a0_angle)))
    
        # pick random distance from goal
        goal_distance = random.uniform(goal_dist_range[0], goal_dist_range[1])
    
        # find goal with random angle and distance
        goal_x = goal_distance * math.cos(a0_angle) + sim.getAgentPosition(a0)[0]
        goal_y = goal_distance * math.sin(a0_angle) + sim.getAgentPosition(a0)[1]
    
        # if the random goal distance is < agent radius, then set to preferred velocity to (0,0)
    
        calculate_pref_vel_one_agent(sim, a0, pref_speed, [goal_x, goal_y], sim.getAgentRadius(a0))
    
        locations += list(sim.getAgentPosition(a0))
        v= list(x for x in sim.getAgentVelocity(a0))
        goals += list([goal_x, goal_y])
        pref_velocities += list(x for x in sim.getAgentPrefVelocity(a0))
    
        # for each neighboring agent, calculate the same stuff as a0
        for neighbor in range(1, num_neighbors + 1):
            valid = -1
            while valid == -1:
                # generate random speed, angle, and starting location
                dist_from_a0 = random.random() * sim.getAgentNeighborDist(a0)
                theta_from_a0 = random.random() * 2 * math.pi
                x = dist_from_a0 * math.cos(theta_from_a0)
                y = dist_from_a0 * math.sin(theta_from_a0)
    
                if no_overlap([x, y], locations, 2 * sim.getAgentRadius(a0) + initial_space_padding):
                    valid = 1
    
            speed = random.uniform(speed_range[0], speed_range[1])
            angle = random.uniform(direction_range[0], direction_range[1])
            sim.addAgent((x, y))
            sim.setAgentVelocity(neighbor, (speed * math.cos(angle), speed * math.sin(angle)))
    
            angle = random.uniform(direction_range[0], direction_range[1])
            sim.setAgentPrefVelocity(neighbor, (pref_speed * math.cos(angle), pref_speed * math.sin(angle)))
    
            # pick random distance from goal
            goal_distance = random.uniform(goal_dist_range[0], goal_dist_range[1])
    
            # find goal with random angle and distance
            goal_x = goal_distance * math.cos(angle) + sim.getAgentPosition(neighbor)[0]
            goal_y = goal_distance * math.sin(angle) + sim.getAgentPosition(neighbor)[1]
    
            # if the random goal distance is < agent radius, then set to preferred velocity to (0,0)
    
            calculate_pref_vel_one_agent(sim, neighbor, pref_speed, [goal_x, goal_y], sim.getAgentRadius(neighbor))
    
            locations += list(sim.getAgentPosition(neighbor))
            current_velocities += list(x for x in sim.getAgentVelocity(neighbor))
            goals += list([goal_x, goal_y])
    
        Range = 10
        Angle = 0.5
        agent_radius = 0.2
    
        
        # get the previous location of agent_no
        a0_prev_loc = np.asarray([locations.copy()[0], locations.copy()[1]])
        # lidar data at time t
        locations = locations[2:]
        centers = [locations[i:i + 2] for i in
                   range(0, len(locations), 2)]  # make data in the way that lidar can work [[,],[,],...,[,]]
        l1 = lidar(Range, Angle, centers, agent_radius)  # create object
        l1.sense_obstacles()  # get_error data
        #l1.plot()
    
    
    
    
    
        p_locations = list(np.array(locations) - np.array(current_velocities) * 0.4)  # going back to time t-1
        centers = [list(np.array(p_locations[i:i + 2]) + 0.4 * np.array(v)) for i in
                   range(0, len(p_locations), 2)]  # off-seting (put center agent at origin and offset other agents)
        l2 = lidar(Range, Angle, centers, agent_radius)
        l2.sense_obstacles()
        #l2.plot()
    
        data[0].append(l2.scan_array + l1.scan_array + v  + goals[0:2])
    
    
        if i%1000==0:
            print('Running simulation', i)
        for step_num_steps in range(1,num_steps):
            for step in range(int(dt/0.1)):
                sim.doStep()
                calculate_pref_vel(sim, pref_speed, goals, goal_threshold)
        data[1].append(list((sim.getAgentPosition(a0) - a0_prev_loc)/dt))
    
    
    
        '''
        print('Simulation Preferred and Current Velocities')
        for agent_no in [a0] + list(range(1,num_neighbors+1)):
            for step in range(num_steps):
                print(str(agent_no) + ' preVel: (%.5f, %.5f), currVel: (%.5f, %.5f)' %( agent_pref_velocities[agent_no][step][0], agent_pref_velocities[agent_no][step][1], agent_curr_velocities[agent_no][step][0], agent_curr_velocities[agent_no][step][1]))
        '''
    
    print('done! Duration: {}'.format(datetime.now() - start_time), "for", num_samples, "samples")
    # save generated samples to pkl file
    with open(save_file_name + '.pkl', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)