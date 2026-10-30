
#TO DO 10/19/2023: REMOVE ADD MASKING [-999.0, -999.0] LOOP IN GENERATE FUNCTIONS
# necessary imports
import rvo2
import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.patches import Polygon
import matplotlib as mpl
import pickle
from datetime import datetime

# video imports
import os
import cv2
from tqdm import tqdm

def normalize_vector(vector):
    norm_vec = vector/np.sqrt(sum(np.array(vector)**2))
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
        this_y = current_points_list[i+1]
        if math.dist(new_point, [this_x, this_y]) < threshold:
            return False
    
    return True

def gaussian_noise(sigma):
    return np.random.normal(0.0, sigma)

def laplace_noise(sigma):
    return np.random.laplace(0.0, np.sqrt((sigma**2)/2))

def add_masked_agents(data_list, masked_agents):
    data_list += masked_agents
    return data_list

def update_df(df, step, data_list):
    while len(df.columns) > len(data_list):
        data_list += [-999.0]
    df.loc[step] = data_list
    return df

def plot_this_sample(i, sim, velocity_scale, this_sample_locations, this_sample_current_velocities, this_sample_pref_velocities = None, sample_type = None, circle_radius = None, line_dist = None):
    num_agents = sim.getNumAgents()
    curr_velocity_plot = this_sample_current_velocities.copy()
    curr_velocity_plot[curr_velocity_plot.select_dtypes(include=['number']).columns] *= velocity_scale
    
    if this_sample_pref_velocities != None:
        pref_velocity_plot = this_sample_pref_velocities.copy()
        pref_velocity_plot[pref_velocity_plot.select_dtypes(include=['number']).columns] *= velocity_scale
    
    width = 10
    height = 10
    figure(figsize=(width, height))
    plt.clf()
    for agent_no in range(num_agents):
        plt.plot(this_sample_locations[str(agent_no)+"_x"], this_sample_locations[str(agent_no)+"_y"], linestyle='--', marker='.', label = agent_no)
        
        if agent_no==0:
            circle = plt.Circle((this_sample_locations[str(agent_no)+"_x"][0], this_sample_locations[str(agent_no)+"_y"][0]), sim.getAgentRadius(agent_no), color = 'k', alpha = 0.1)
        else:
            circle = plt.Circle((this_sample_locations[str(agent_no)+"_x"][0], this_sample_locations[str(agent_no)+"_y"][0]), sim.getAgentRadius(agent_no), color = 'k', alpha = 0.1)
        plt.gca().add_patch(circle)
        
        if agent_no == num_agents:
            # arrow for preferred velocity
            plt.arrow(this_sample_locations[str(agent_no)+"_x"][0],
                      this_sample_locations[str(agent_no)+"_y"][0],
                      pref_velocity_plot[str(agent_no)+"_x"][0],
                      pref_velocity_plot[str(agent_no)+"_y"][0]
                      , width = 0.03, alpha = 0.2, label = "pref_vel")
            
            # arrow for current velocity
            start_x = this_sample_locations[str(agent_no)+"_x"][0] - curr_velocity_plot[str(agent_no)+"_x"][0]
            start_y = this_sample_locations[str(agent_no)+"_y"][0] - curr_velocity_plot[str(agent_no)+"_y"][0]
            
            dx = curr_velocity_plot[str(agent_no)+"_x"][0]
            dy = curr_velocity_plot[str(agent_no)+"_y"][0]
            plt.arrow(start_x, start_y, dx, dy, width = 0.03, alpha = 0.2, color = 'r', label = "cur_vel")
            
            start_x = this_sample_locations[str(agent_no)+"_x"][0]
            start_y = this_sample_locations[str(agent_no)+"_y"][0]
            
            dx = curr_velocity_plot[str(agent_no)+"_x"][1]
            dy = curr_velocity_plot[str(agent_no)+"_y"][1]
            plt.arrow(start_x, start_y, dx, dy, width = 0.03, alpha = 0.2, color = 'r')
            
        
        elif this_sample_pref_velocities != None:
            # arrow for preferred velocity
            plt.arrow(this_sample_locations[str(agent_no)+"_x"][0],
                      this_sample_locations[str(agent_no)+"_y"][0],
                      pref_velocity_plot[str(agent_no)+"_x"][0],
                      pref_velocity_plot[str(agent_no)+"_y"][0],
                      width = 0.03, alpha = 0.2)
            
            # arrow for current velocity
            start_x = (this_sample_locations[str(agent_no)+"_x"][0] - curr_velocity_plot[str(agent_no)+"_x"][0])
            start_y = (this_sample_locations[str(agent_no)+"_y"][0] - curr_velocity_plot[str(agent_no)+"_y"][0])
            
            dx = curr_velocity_plot[str(agent_no)+"_x"][0]
            dy = curr_velocity_plot[str(agent_no)+"_y"][0]
            plt.arrow(start_x, start_y, dx, dy, width = 0.03, alpha = 0.2, color = 'r')
            
            start_x = this_sample_locations[str(agent_no)+"_x"][0]
            start_y = this_sample_locations[str(agent_no)+"_y"][0]
            
            dx = curr_velocity_plot[str(agent_no)+"_x"][1]
            dy = curr_velocity_plot[str(agent_no)+"_y"][1]
            plt.arrow(start_x, start_y, dx, dy, width = 0.03, alpha = 0.2, color = 'r')
            
        
    if sample_type == 'circle':
        circle_0 = plt.Circle((0, 0), circle_radius, color = 'b', alpha = 0.05)
        plt.gca().add_patch(circle_0)
        plot_lim = circle_radius+0.5 
        plt.xlim([-plot_lim, plot_lim])
        plt.ylim([-plot_lim, plot_lim])
    elif sample_type == None:
        circle_0 = plt.Circle((0, 0), sim.getAgentNeighborDist(0), color = 'b', alpha = 0.05)
        plt.gca().add_patch(circle_0)
        
        plt.xlim([-6, 6])
        plt.ylim([-6, 6])
    elif sample_type == 'lineup':
        plt.plot([-8, 8], [-line_dist, -line_dist], linestyle='--', marker='.', color = 'k')
        plt.plot([-8, 8], [line_dist, line_dist], linestyle='--', marker='.', color = 'k')
        plt.xlim([-10, 10])
        plt.ylim([-10, 10])
    else:
        # same obsatcles as NN
        '''
        o1_vert = [(1,3), (1, 1), (4, 1), (4, 3)]
        o2_vert = [(-3.5, 3), (-3.5, 1), (-1.5, 1), (-1.5, 3)]
        '''
        
        # narrower space bewteen obstacles
        
        o1_vert = [(0.5,3), (0.5, 1), (4, 1), (4, 3)]
        o2_vert = [(-3.5, 3), (-3.5, 1), (-1, 1), (-1, 3)]
        
        
        wall1 = [(-5,10), (10, 10), (10, -6), (-5, -6)]
        plt.gca().add_patch(Polygon(o1_vert, fill = False, edgecolor = "r"))
        plt.gca().add_patch(Polygon(o2_vert, fill = False, edgecolor = "r"))
        plt.gca().add_patch(Polygon(wall1, fill = False, edgecolor = "r"))
        
        plt.ylim(-10.5, 10.5)
        plt.xlim(-10.5, 10.5)
    
    
    plt.title("sample " + str(i))
    plt.legend()
    plt.show()
    return

def magnitude(vector):
    return math.sqrt(sum(pow(element, 2) for element in vector))

def calculate_pref_vel(sim, pref_speed, goal_points, goal_threshold):
    for neighbor in range(0,sim.getNumAgents()): 
        goal_x = goal_points[neighbor*2]
        goal_y = goal_points[neighbor*2+1]
        if math.dist(list(sim.getAgentPosition(neighbor)), [goal_x, goal_y]) < goal_threshold:
            sim.setAgentPrefVelocity(neighbor, (0.0, 0.0))
        else:
            sim.setAgentPrefVelocity(neighbor, tuple(pref_speed*normalize_vector([goal_i - curr_i for goal_i, curr_i in zip([goal_x, goal_y], list(sim.getAgentPosition(neighbor)))])))

def calculate_pref_vel_one_agent(sim, agent_no, pref_speed, goal_points, goal_threshold):
    goal_x = goal_points[0]
    goal_y = goal_points[1]
    if math.dist(list(sim.getAgentPosition(agent_no)), [goal_x, goal_y]) < goal_threshold:
        sim.setAgentPrefVelocity(agent_no, (0.0, 0.0))
    else:
        sim.setAgentPrefVelocity(agent_no, tuple(pref_speed*normalize_vector([goal_i - curr_i for goal_i, curr_i in zip([goal_x, goal_y], list(sim.getAgentPosition(agent_no)))])))

def all_agents_done(sim, goal_points, goal_threshold):
    for neighbor in range(0,sim.getNumAgents()):
        goal_x = goal_points[neighbor*2]
        goal_y = goal_points[neighbor*2+1]
        if not (math.dist(list(sim.getAgentPosition(neighbor)), [goal_x, goal_y]) < goal_threshold):
            return False
    return True

def make_video(df, df_pred, simulation_idx, collisions_x, collisions_y, agent_radius, save_folder, save_name):
    
    plt.figure(figsize=(12., 8.5))
    #height = 432
    #width = 576
    width = 1000
    height = 720
    dt = 0.12
    
    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    tmp_name = this_file_dir + 'court.png'
    
    # initiate video
    
    movie_name = save_folder + '/' + save_name + '.mp4'
    video = cv2.VideoWriter(movie_name, cv2.VideoWriter_fourcc(*'mp4v'), 
                        fps = int(1./dt), frameSize=(width, height))
    '''
    movie_name = save_folder + '/' + save_name + '.avi'
    video = cv2.VideoWriter(movie_name, cv2.VideoWriter_fourcc('M','J','P','G'), 
                        fps = int(1./dt), frameSize=(width, height))
    '''
    
    # initiate plot
    plt.clf()
    
    # init scatters
    scatter = []
    # CHANGE LATER: AGENT_RADIUS LIST AS INPUT
    #agent_radius = [0.5]*4 + [0]*7 #[0.5, 1, 0.3] + [0]*8
    for i in range(len(df.columns)//2):
        x = 0
        y = 0
        
        # DELETE LATER. This plots an fake agent in the center of the plot so we can check the agent size
        '''
        #plt.scatter(0,0, s =(points_radius)**2, linewidths = 2.0, edgecolors='red')
        plt.gca().add_patch(mpl.patches.Circle((x, y), radius=radius , facecolor='black'))
        '''
        
        scatter.append(mpl.patches.Circle((x, y), radius=agent_radius, facecolor='black', edgecolor = 'red'))
        
        
        # if no prediction, then just plot the ground truth locations
        if not isinstance(df_pred, pd.DataFrame):
            plt.scatter(df['%i_x'%i], df['%i_y'%i], marker='.', color='r', alpha = 0.2)
        
        # if there are predictions, then we plot both ground truth and prediction locations
        else:
            if i ==0:
                plt.scatter(df['%i_x'%i], df['%i_y'%i], marker='.', label = 'ground truth', color='r', alpha = 0.2)
                plt.scatter(df_pred['%i_x'%i], df_pred['%i_y'%i], marker='.', label = 'prediction', color='b', alpha = 0.2)
            else:
                plt.scatter(df['%i_x'%i], df['%i_y'%i], marker='.', color='r', alpha = 0.2)
                plt.scatter(df_pred['%i_x'%i], df_pred['%i_y'%i], marker='.', color='b', alpha = 0.2)
    
    # if no collision data is given (only used when plotting the model predictions)
    if (isinstance(collisions_x, list) or isinstance(collisions_y, list)):
        plt.scatter(collisions_x,collisions_y, s=80, facecolors='none', edgecolors='k', zorder=3, alpha = 0.3, label = 'collision')
        
    
    # if no predictions, then agents will follow ground truth paths
    if not isinstance(df_pred, pd.DataFrame):
        T = len(df.index)
        for time_step in range(T):
            data = df.iloc[time_step]
            
            for i in range(len(df.columns)//2):
                x = []
                y = []
                x.append(data['%i_x'%i])
                y.append(data['%i_y'%i])
                scatter[i].center = [x, y]
                plt.gca().add_patch(scatter[i])
            if agent_radius == 0.5:
                plt.ylim(-6, 6)
                plt.xlim(-6, 6)
            else:
                plt.ylim(-10, 10)#(-3, 3)
                plt.xlim(-10, 10)#(-3, 3)
                # same obsatcles as NN
                '''
                o1_vert = [(1,3), (1, 1), (4, 1), (4, 3)]
                o2_vert = [(-3.5, 3), (-3.5, 1), (-1.5, 1), (-1.5, 3)]
                '''
                
                # narrower space bewteen obstacles
                
                o1_vert = [(0.5,3), (0.5, 1), (4, 1), (4, 3)]
                o2_vert = [(-3.5, 3), (-3.5, 1), (-1, 1), (-1, 3)]
                
                
                wall1 = [(-5,10), (10, 10), (10, -6), (-5, -6)]
                plt.gca().add_patch(Polygon(o1_vert, fill = False, edgecolor = "r"))
                plt.gca().add_patch(Polygon(o2_vert, fill = False, edgecolor = "r"))
                plt.gca().add_patch(Polygon(wall1, fill = False, edgecolor = "r"))
                plt.ylim(-10.5, 10.5)
                plt.xlim(-10.5, 10.5)
                
            
            plt.gca().set_aspect('equal')
            plt.title('Test Simulation: '+ str(time_step))
            #plt.title('Test Simulation: '+ str(simulation_idx))
            #plt.legend()
            
            #plt.show()
            #plt.pause(dt)
            plt.savefig(tmp_name, dpi=90)
            #plt.savefig(save_folder + '/' + save_name + str(time_step)  + '.png', dpi=90)
            
            frame = cv2.imread(tmp_name)
            #scale_percent = 90
            #width2 = int(frame.shape[1] * scale_percent / 100)
            #height2 = int(frame.shape[0] * scale_percent / 100)
            frame = cv2.resize(frame, (width, height))
            video.write(frame)
        video.release()
        del video
    else:
        T = len(df_pred.index)
        for time_step in range(T):
            data = df_pred.iloc[time_step]
            
            for i in range(len(df.columns)//2):
                x = []
                y = []
                x.append(data['%i_x'%i])
                y.append(data['%i_y'%i])
                scatter[i].center = [x, y]
                plt.gca().add_patch(scatter[i])
            
        
            if agent_radius == 0.2:
                plt.ylim(-6, 6)
                plt.xlim(-6, 6)
                #plt.ylim(-10, 10)
                #plt.xlim(-10, 10)
            else:
                #plt.ylim(-6, 6)
                #plt.xlim(-6, 6)
                
                plt.ylim(-10, 10)
                plt.xlim(-10, 10)
            
            # DELETE LATER. This is for testing the size of the agents
            '''
            plt.xticks(np.arange(-6.0, 6.0, 0.2), rotation = 'vertical')
            plt.yticks(np.arange(-6.0, 6.0, 0.2))
            plt.grid()
            '''
            
            
            plt.gca().set_aspect('equal')
            
            plt.title('Test Simulation: '+ str(simulation_idx))
            #plt.legend()
            
            #plt.show()
            #plt.pause(dt)
            plt.savefig(tmp_name, dpi=90)
            #plt.savefig(save_folder + '/' + save_name + str(time_step)  + '.png', dpi=90)
            
            frame = cv2.imread(tmp_name)
            #scale_percent = 90
            #width2 = int(frame.shape[1] * scale_percent / 100)
            #height2 = int(frame.shape[0] * scale_percent / 100)
            frame = cv2.resize(frame, (width, height))
            video.write(frame)
        video.release()
        del video

def generate_n_samples(num_samples, num_steps, neighbors_range, speed_range, direction_range, speed_noise_range, direction_noise_range, noise_type, goal_dist_range, pref_speed, initial_space_padding, velocity_scale, save_file_name, plot_samples):
    start_time = datetime.now()
    samples = []
    for i in range(num_samples):
        # set up the simulator
        '''
        Hyperparameters are set to whatever is listed in RVO2 paper
            - timeStep = 0.10
            - neighborDist = 3
            - MaxNeighbors = 10
            - AgentTimeHorizon = 3
            - AgentTimeHorizonObst = 1
            - AgentRadius = 0.2
            - AgentMaxSpeed = 3.5
        '''
        # AHHH model's sim
        #sim = rvo2.PyRVOSimulator(0.10, 3, 10, 3, 1, 0.2, 3.5)
        
        # finetuning sim
        #sim = rvo2.PyRVOSimulator(0.10, 5, 10, 1.0, 1, 0.5, 5.0)
        
        # DFSM finetuning sim
        #sim = rvo2.PyRVOSimulator(0.10, 3, 10, 3, 1, agent_radius, 5.0)
        
        # NEW Finetune comparison sim
        # ORCA1
        sim = rvo2.PyRVOSimulator(0.10, 3, 10, 3, 1, 0.2, pref_speed+0.5)
        
        # ORCA2
        #sim = rvo2.PyRVOSimulator(0.10, 5, 10, 1.0, 1, 0.5, pref_speed+0.5)
        
        # make a new dataframe for this sample
        col_list = ['0_x', '0_y', '1_x', '1_y', '2_x', '2_y', '3_x', '3_y', '4_x', '4_y', '5_x', '5_y', '6_x', '6_y', '7_x', '7_y', '8_x', '8_y', '9_x', '9_y', '10_x', '10_y']
        
        col_list_2 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        
        
        '''
        Each sample is a list of dataframes
            0. current location (x, y)
            1. current veloicity (x, y)
            2. preferred velocity (x, y)
            3. preferred speed (float)
            4. distance from goal (float)
            5. goal location (x, y)
        '''
        this_sample_locations = pd.DataFrame(columns=col_list)
        this_sample_current_velocities = pd.DataFrame(columns=col_list)
        this_sample_pref_velocities = pd.DataFrame(columns=col_list)
        #this_sample_pref_velocities_normalized = pd.DataFrame(columns=col_list)
        this_sample_pref_speeds = pd.DataFrame(columns=col_list_2)
        this_sample_dist_from_goal = pd.DataFrame(columns=col_list_2)
        this_sample_goals = pd.DataFrame(columns=col_list)
        
        locations = []
        current_velocities = []
        pref_velocities = []
        #pref_velocities_normalized = []
        pref_speeds = []
        dist_from_goal = []
        goals = []
        
        masked_neighbors = []
        # pick random number of neighbors
        num_neighbors = 10#random.randint(neighbors_range[0], neighbors_range[1])
        if num_neighbors<neighbors_range[1]:
            for diff in range(neighbors_range[1] - num_neighbors):
                masked_neighbors += [-999, -999]
        
        # put agent 0 at the origin. give a random speed and angle for initial velocity
        a0_speed = random.uniform(speed_range[0], speed_range[1])
        a0_angle = random.uniform(direction_range[0], direction_range[1])
        a0 = sim.addAgent((0, 0))
        sim.setAgentVelocity(a0, (a0_speed*math.cos(a0_angle), a0_speed*math.sin(a0_angle)))
        
        # Uncomment to randomly initialize preferred velocity. Note that we set a preferred speed so we just need random direction
        a0_angle = random.uniform(direction_range[0], direction_range[1])
        sim.setAgentPrefVelocity(a0, (pref_speed*math.cos(a0_angle), pref_speed*math.sin(a0_angle)))
        
        # pick random distance from goal
        goal_distance = random.uniform(goal_dist_range[0], goal_dist_range[1])
        
        # find goal with random angle and distance
        goal_x = goal_distance*math.cos(a0_angle) + sim.getAgentPosition(a0)[0]
        goal_y = goal_distance*math.sin(a0_angle) + sim.getAgentPosition(a0)[1]
        
        # if the random goal distance is < agent radius, then set to preferred velocity to (0,0)
        
        calculate_pref_vel_one_agent(sim, a0, pref_speed, [goal_x, goal_y], sim.getAgentRadius(a0))
        
        
        locations += list(sim.getAgentPosition(a0))
        current_velocities += list(x for x in sim.getAgentVelocity(a0))
        pref_velocities += list(x for x in sim.getAgentPrefVelocity(a0))
        #pref_velocities_normalized += list(normalize_vector(sim.getAgentPrefVelocity(a0)))
        pref_speeds += [pref_speed]
        dist_from_goal += [goal_distance]
        goals += list([goal_x, goal_y])
        
        
        # for each neighboring agent, calculate the same stuff as a0
        for neighbor in range(1,num_neighbors+1):
            valid = -1
            while valid == -1:
                # generate random speed, angle, and starting location
                dist_from_a0 = random.random()*sim.getAgentNeighborDist(a0)
                theta_from_a0 = random.random()*2*math.pi
                x = dist_from_a0*math.cos(theta_from_a0)
                y = dist_from_a0*math.sin(theta_from_a0)
                
                if no_overlap([x, y], locations, 2*sim.getAgentRadius(a0) + initial_space_padding):
                    valid = 1
            
            speed = random.uniform(speed_range[0], speed_range[1])
            angle = random.uniform(direction_range[0], direction_range[1])
            sim.addAgent((x, y))
            sim.setAgentVelocity(neighbor, (speed*math.cos(angle), speed*math.sin(angle)))
            
            angle = random.uniform(direction_range[0], direction_range[1])
            sim.setAgentPrefVelocity(neighbor, (pref_speed*math.cos(angle), pref_speed*math.sin(angle)))
            
            # pick random distance from goal
            goal_distance = random.uniform(goal_dist_range[0], goal_dist_range[1])
            
            # find goal with random angle and distance
            goal_x = goal_distance*math.cos(angle) + sim.getAgentPosition(neighbor)[0]
            goal_y = goal_distance*math.sin(angle) + sim.getAgentPosition(neighbor)[1]
            
            # if the random goal distance is < agent radius, then set to preferred velocity to (0,0)
            
            calculate_pref_vel_one_agent(sim, neighbor, pref_speed, [goal_x, goal_y], sim.getAgentRadius(neighbor))
            
            
            locations += list(sim.getAgentPosition(neighbor))
            current_velocities += list(x for x in sim.getAgentVelocity(neighbor))
            pref_velocities += list(x for x in sim.getAgentPrefVelocity(neighbor))
            #pref_velocities_normalized+= list(normalize_vector(sim.getAgentPrefVelocity(neighbor)))
            pref_speeds += [pref_speed]
            dist_from_goal += [goal_distance]
            goals += list([goal_x, goal_y])
             
        # record the initial location of each agent in the scene (pad any nonexistent neighbors with -999)
        list_of_data_list = [locations, current_velocities, pref_velocities, pref_speeds, dist_from_goal, goals]
        list_of_df = [this_sample_locations, this_sample_current_velocities, this_sample_pref_velocities, this_sample_pref_speeds, this_sample_dist_from_goal, this_sample_goals]
        
        # add masked_agents to the end of each list
        '''
        for data_list in list_of_data_list:
            data_list = add_masked_agents(data_list, masked_neighbors)
        '''

        # update the dataframes
        for idx in range(len(list_of_df)):  
            list_of_df[idx] = update_df(list_of_df[idx], 0, list_of_data_list[idx])
        
        # THESE ARE JUST FOR PRINTING. NOT SAVING THE DATA
        agent_pref_velocities = {key: [] for key in [a0] + list(range(1,num_neighbors+1))}
        agent_curr_velocities = {key: [] for key in [a0] + list(range(1,num_neighbors+1))}
        
        for agent_no in [a0] + list(range(1,num_neighbors+1)):
            #print(str(agent_no) + ' preVel: (%.3f, %.3f), currVel: (%.3f, %.3f)' %( sim.getAgentPrefVelocity(agent_no)[0], sim.getAgentPrefVelocity(agent_no)[1], sim.getAgentVelocity(agent_no)[0], sim.getAgentVelocity(agent_no)[1]))
            agent_pref_velocities[agent_no] += [sim.getAgentPrefVelocity(agent_no)]
            agent_curr_velocities[agent_no] += [sim.getAgentVelocity(agent_no)]
            
        
        #print('Simulation has %i agents in it.' % sim.getNumAgents())
        if i%1000==0:
            print('Running simulation', i)
        for step in range(1,num_steps):
            sim.doStep()
            locations = []
            current_velocities = []
            pref_velocities = []
            #pref_velocities_normalized = []
            pref_speeds = []
            dist_from_goal = []
            goals = []
            for agent_no in [a0] + list(range(1,num_neighbors+1)):
                
                goal_x = this_sample_goals[str(agent_no)+'_x'].iloc[0]
                goal_y = this_sample_goals[str(agent_no)+'_y'].iloc[0]
                new_goal_distance = math.dist(list(sim.getAgentPosition(agent_no)), [goal_x, goal_y])
                calculate_pref_vel_one_agent(sim, agent_no, pref_speed, [goal_x, goal_y], sim.getAgentRadius(agent_no)) #sim.setAgentPrefVelocity(agent_no, tuple(pref_speed*normalize_vector([goal_i - curr_i for goal_i, curr_i in zip([goal_x, goal_y], list(sim.getAgentPosition(agent_no)))])))
                
                locations+= list(sim.getAgentPosition(agent_no))
                current_velocities += list(x for x in sim.getAgentVelocity(agent_no))
                pref_velocities += list(x for x in sim.getAgentPrefVelocity(agent_no))
                #pref_velocities_normalized += list(normalize_vector(sim.getAgentPrefVelocity(agent_no)))
                pref_speeds += [pref_speed]
                dist_from_goal += [new_goal_distance]
                goals += list([goal_x, goal_y])
            
            # make lists for cleaner For loops
            list_of_data_list = [locations, current_velocities, pref_velocities, pref_speeds, dist_from_goal, goals]
            list_of_df = [this_sample_locations, this_sample_current_velocities, this_sample_pref_velocities, this_sample_pref_speeds, this_sample_dist_from_goal, this_sample_goals]
            
            # add masked_agents to the end of each list
            '''
            for data_list in list_of_data_list:
                data_list = add_masked_agents(data_list, masked_neighbors)
            '''

            # update the dataframes
            for idx in range(len(list_of_df)):  
                list_of_df[idx] = update_df(list_of_df[idx], step, list_of_data_list[idx])
            
            for agent_no in [a0] + list(range(1,num_neighbors+1)):
                #print(str(agent_no) + ' preVel: (%.3f, %.3f), currVel: (%.3f, %.3f)' %( sim.getAgentPrefVelocity(agent_no)[0], sim.getAgentPrefVelocity(agent_no)[1], sim.getAgentVelocity(agent_no)[0], sim.getAgentVelocity(agent_no)[1]))
                agent_pref_velocities[agent_no] += [sim.getAgentPrefVelocity(agent_no)]
                agent_curr_velocities[agent_no] += [sim.getAgentVelocity(agent_no)]
        
        samples.append([this_sample_locations, this_sample_current_velocities, this_sample_pref_velocities, this_sample_pref_speeds, this_sample_dist_from_goal, this_sample_goals])
        
        '''
        print('Simulation Preferred and Current Velocities')
        for agent_no in [a0] + list(range(1,num_neighbors+1)):
            for step in range(num_steps):
                print(str(agent_no) + ' preVel: (%.5f, %.5f), currVel: (%.5f, %.5f)' %( agent_pref_velocities[agent_no][step][0], agent_pref_velocities[agent_no][step][1], agent_curr_velocities[agent_no][step][0], agent_curr_velocities[agent_no][step][1]))
        '''
        
        # plot sample
        if plot_samples:#i==0 or (i+1)%100==0:
            plot_this_sample(i, sim, velocity_scale, this_sample_locations, this_sample_current_velocities, this_sample_pref_velocities)
            
    
    # save generated samples to pkl file
    with open(save_file_name + '.pkl', 'wb') as handle:
        pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)
    '''
    print("done! Checking file integrity...")
    with open(save_file_name + '.pkl', 'rb') as handle:
        b = pickle.load(handle)
    
    # check that saved file matches original Python object
    for sample_idx in range(len(samples)):
        for df_idx in range(len(samples[sample_idx])):
            if not all(samples[sample_idx][df_idx].all() == b[sample_idx][df_idx].all()):
                print("MISMATCH IN SAMPLE ", sample_idx, "DF", df_idx)
    '''
    print('done! Duration: {}'.format(datetime.now() - start_time), "for", num_samples, "samples")
    return

def generate_n_baseline_samples(num_samples, num_steps, neighbors_range, speed_range, direction_range, speed_noise_range, direction_noise_range, noise_type, goal_dist_range, pref_speed, initial_space_padding, velocity_scale, save_file_name, plot_samples):
    start_time = datetime.now()
    samples = []
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
        
        # make a new dataframe for this sample
        col_list = ['0_x', '0_y', '1_x', '1_y', '2_x', '2_y', '3_x', '3_y', '4_x', '4_y', '5_x', '5_y', '6_x', '6_y', '7_x', '7_y', '8_x', '8_y', '9_x', '9_y', '10_x', '10_y']
        
        col_list_2 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        
        
        '''
        Each sample is a list of dataframes
            0. current location (x, y)
            1. current veloicity (x, y)
            2. goal location (x, y)
        '''
        this_sample_locations = pd.DataFrame(columns=col_list)
        this_sample_current_velocities = pd.DataFrame(columns=col_list)
        #this_sample_pref_velocities = pd.DataFrame(columns=col_list)
        #this_sample_pref_velocities_normalized = pd.DataFrame(columns=col_list)
        #this_sample_pref_speeds = pd.DataFrame(columns=col_list_2)
        #this_sample_dist_from_goal = pd.DataFrame(columns=col_list_2)
        this_sample_goals = pd.DataFrame(columns=col_list)
        
        locations = []
        current_velocities = []
        #pref_velocities = []
        #pref_velocities_normalized = []
        #pref_speeds = []
        #dist_from_goal = []
        goals = []
        
        masked_neighbors = []
        # pick random number of neighbors
        num_neighbors = 10#random.randint(neighbors_range[0], neighbors_range[1])
        if num_neighbors<neighbors_range[1]:
            for diff in range(neighbors_range[1] - num_neighbors):
                masked_neighbors += [-999, -999]
        
        # put agent 0 at the origin. give a random speed and angle for initial velocity
        a0_speed = random.uniform(speed_range[0], speed_range[1])
        a0_angle = random.uniform(direction_range[0], direction_range[1])
        a0 = sim.addAgent((0, 0))
        sim.setAgentVelocity(a0, (a0_speed*math.cos(a0_angle), a0_speed*math.sin(a0_angle)))
        
        # Uncomment to randomly initialize preferred velocity. Note that we set a preferred speed so we just need random direction
        a0_angle = random.uniform(direction_range[0], direction_range[1])
        sim.setAgentPrefVelocity(a0, (pref_speed*math.cos(a0_angle), pref_speed*math.sin(a0_angle)))
        
        # pick random distance from goal
        goal_distance = random.uniform(goal_dist_range[0], goal_dist_range[1])
        
        # find goal with random angle and distance
        goal_x = goal_distance*math.cos(a0_angle) + sim.getAgentPosition(a0)[0]
        goal_y = goal_distance*math.sin(a0_angle) + sim.getAgentPosition(a0)[1]
        
        # if the random goal distance is < agent radius, then set to preferred velocity to (0,0)
        
        calculate_pref_vel_one_agent(sim, a0, pref_speed, [goal_x, goal_y], goal_threshold)
        
        
        locations += list(sim.getAgentPosition(a0))
        current_velocities += list(x for x in sim.getAgentVelocity(a0))
        #pref_velocities += list(x for x in sim.getAgentPrefVelocity(a0))
        #pref_velocities_normalized += list(normalize_vector(sim.getAgentPrefVelocity(a0)))
        #pref_speeds += [pref_speed]
        #dist_from_goal += [goal_distance]
        goals += list([goal_x, goal_y])
        
        
        # for each neighboring agent, calculate the same stuff as a0
        for neighbor in range(1,num_neighbors+1):
            valid = -1
            while valid == -1:
                # generate random speed, angle, and starting location
                dist_from_a0 = random.random()*sim.getAgentNeighborDist(a0)
                theta_from_a0 = random.random()*2*math.pi
                x = dist_from_a0*math.cos(theta_from_a0)
                y = dist_from_a0*math.sin(theta_from_a0)
                
                if no_overlap([x, y], locations, 2*sim.getAgentRadius(a0) + initial_space_padding):
                    valid = 1
            
            speed = random.uniform(speed_range[0], speed_range[1])
            angle = random.uniform(direction_range[0], direction_range[1])
            sim.addAgent((x, y))
            sim.setAgentVelocity(neighbor, (speed*math.cos(angle), speed*math.sin(angle)))
            
            angle = random.uniform(direction_range[0], direction_range[1])
            sim.setAgentPrefVelocity(neighbor, (pref_speed*math.cos(angle), pref_speed*math.sin(angle)))
            
            # pick random distance from goal
            goal_distance = random.uniform(goal_dist_range[0], goal_dist_range[1])
            
            # find goal with random angle and distance
            goal_x = goal_distance*math.cos(angle) + sim.getAgentPosition(neighbor)[0]
            goal_y = goal_distance*math.sin(angle) + sim.getAgentPosition(neighbor)[1]
            
            # if the random goal distance is < agent radius, then set to preferred velocity to (0,0)
            
            calculate_pref_vel_one_agent(sim, neighbor, pref_speed, [goal_x, goal_y], goal_threshold)
            
            
            locations += list(sim.getAgentPosition(neighbor))
            current_velocities += list(x for x in sim.getAgentVelocity(neighbor))
            #pref_velocities += list(x for x in sim.getAgentPrefVelocity(neighbor))
            #pref_velocities_normalized+= list(normalize_vector(sim.getAgentPrefVelocity(neighbor)))
            #pref_speeds += [pref_speed]
            #dist_from_goal += [goal_distance]
            goals += list([goal_x, goal_y])
             
        # record the initial location of each agent in the scene (pad any nonexistent neighbors with -999)
        list_of_data_list = [locations, current_velocities, goals]
        #list_of_data_list = [locations, current_velocities, pref_velocities, pref_speeds, dist_from_goal, goals]
        list_of_df = [this_sample_locations, this_sample_current_velocities, this_sample_goals]
        #list_of_df = [this_sample_locations, this_sample_current_velocities, this_sample_pref_velocities, this_sample_pref_speeds, this_sample_dist_from_goal, this_sample_goals]
        
        # add masked_agents to the end of each list
        '''
        for data_list in list_of_data_list:
            data_list = add_masked_agents(data_list, masked_neighbors)
        '''

        # update the dataframes
        for idx in range(len(list_of_df)):  
            list_of_df[idx] = update_df(list_of_df[idx], 0, list_of_data_list[idx])
        
        # THESE ARE JUST FOR PRINTING. NOT SAVING THE DATA
        agent_pref_velocities = {key: [] for key in [a0] + list(range(1,num_neighbors+1))}
        agent_curr_velocities = {key: [] for key in [a0] + list(range(1,num_neighbors+1))}
        
        for agent_no in [a0] + list(range(1,num_neighbors+1)):
            #print(str(agent_no) + ' preVel: (%.3f, %.3f), currVel: (%.3f, %.3f)' %( sim.getAgentPrefVelocity(agent_no)[0], sim.getAgentPrefVelocity(agent_no)[1], sim.getAgentVelocity(agent_no)[0], sim.getAgentVelocity(agent_no)[1]))
            agent_pref_velocities[agent_no] += [sim.getAgentPrefVelocity(agent_no)]
            agent_curr_velocities[agent_no] += [sim.getAgentVelocity(agent_no)]
            
        
        #print('Simulation has %i agents in it.' % sim.getNumAgents())
        if i%1000==0:
            print('Running simulation', i)
        for step in range(1,num_steps):
            for step in range(int(dt/0.1)):
                sim.doStep()
                calculate_pref_vel(sim, pref_speed, goals, goal_threshold)
            locations = []
            current_velocities = []
            #pref_velocities = []
            #pref_velocities_normalized = []
            #pref_speeds = []
            #dist_from_goal = []
            goals = []
            for agent_no in [a0] + list(range(1,num_neighbors+1)):
                # get the previous location of agent_no
                agent_no_prev_loc = np.asarray([this_sample_locations[str(agent_no)+'_x'].iloc[0], this_sample_locations[str(agent_no)+'_y'].iloc[0]])
                
                goal_x = this_sample_goals[str(agent_no)+'_x'].iloc[0]
                goal_y = this_sample_goals[str(agent_no)+'_y'].iloc[0]
                #new_goal_distance = math.dist(list(sim.getAgentPosition(agent_no)), [goal_x, goal_y])
                calculate_pref_vel_one_agent(sim, agent_no, pref_speed, [goal_x, goal_y], goal_threshold) #sim.setAgentPrefVelocity(agent_no, tuple(pref_speed*normalize_vector([goal_i - curr_i for goal_i, curr_i in zip([goal_x, goal_y], list(sim.getAgentPosition(agent_no)))])))
                
                locations+= list(sim.getAgentPosition(agent_no))
                current_velocities += list((sim.getAgentPosition(agent_no) - agent_no_prev_loc)/dt) #list(x for x in sim.getAgentVelocity(agent_no))
                #pref_velocities += list(x for x in sim.getAgentPrefVelocity(agent_no))
                #pref_velocities_normalized += list(normalize_vector(sim.getAgentPrefVelocity(agent_no)))
                #pref_speeds += [pref_speed]
                #dist_from_goal += [new_goal_distance]
                goals += list([goal_x, goal_y])
            
            # make lists for cleaner For loops
            list_of_data_list = [locations, current_velocities, goals]
            #list_of_data_list = [locations, current_velocities, pref_velocities, pref_speeds, dist_from_goal, goals]
            list_of_df = [this_sample_locations, this_sample_current_velocities, this_sample_goals]
            #list_of_df = [this_sample_locations, this_sample_current_velocities, this_sample_pref_velocities, this_sample_pref_speeds, this_sample_dist_from_goal, this_sample_goals]
            
            # add masked_agents to the end of each list
            '''
            for data_list in list_of_data_list:
                data_list = add_masked_agents(data_list, masked_neighbors)
            '''

            # update the dataframes
            for idx in range(len(list_of_df)):  
                list_of_df[idx] = update_df(list_of_df[idx], step, list_of_data_list[idx])
            
            for agent_no in [a0] + list(range(1,num_neighbors+1)):
                #print(str(agent_no) + ' preVel: (%.3f, %.3f), currVel: (%.3f, %.3f)' %( sim.getAgentPrefVelocity(agent_no)[0], sim.getAgentPrefVelocity(agent_no)[1], sim.getAgentVelocity(agent_no)[0], sim.getAgentVelocity(agent_no)[1]))
                agent_pref_velocities[agent_no] += [sim.getAgentPrefVelocity(agent_no)]
                agent_curr_velocities[agent_no] += [sim.getAgentVelocity(agent_no)]
        
        samples.append([this_sample_locations, this_sample_current_velocities, this_sample_goals])
        #samples.append([this_sample_locations, this_sample_current_velocities, this_sample_pref_velocities, this_sample_pref_speeds, this_sample_dist_from_goal, this_sample_goals])
        
        '''
        print('Simulation Preferred and Current Velocities')
        for agent_no in [a0] + list(range(1,num_neighbors+1)):
            for step in range(num_steps):
                print(str(agent_no) + ' preVel: (%.5f, %.5f), currVel: (%.5f, %.5f)' %( agent_pref_velocities[agent_no][step][0], agent_pref_velocities[agent_no][step][1], agent_curr_velocities[agent_no][step][0], agent_curr_velocities[agent_no][step][1]))
        '''
        
        # plot sample
        if plot_samples:#i==0 or (i+1)%100==0:
            plot_this_sample(i, sim, velocity_scale, this_sample_locations, this_sample_current_velocities)
            #plot_this_sample(i, sim, velocity_scale, this_sample_locations, this_sample_current_velocities, this_sample_pref_velocities)
            
    
    # save generated samples to pkl file
    with open(save_file_name + '.pkl', 'wb') as handle:
        pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)
    '''
    print("done! Checking file integrity...")
    with open(save_file_name + '.pkl', 'rb') as handle:
        b = pickle.load(handle)
    
    # check that saved file matches original Python object
    for sample_idx in range(len(samples)):
        for df_idx in range(len(samples[sample_idx])):
            if not all(samples[sample_idx][df_idx].all() == b[sample_idx][df_idx].all()):
                print("MISMATCH IN SAMPLE ", sample_idx, "DF", df_idx)
    '''
    print('done! Duration: {}'.format(datetime.now() - start_time), "for", num_samples, "samples")
    return

def generate_n_circle_samples(num_samples, agents_range, pref_speed, initial_space_padding, velocity_scale, max_steps, save_file_name, plot_samples, goal_threshold, agent_radius):
    start_time = datetime.now()
    samples = []
    for i in range(num_samples):
        # set up the simulator
        '''
        Hyperparameters are set to whatever is listed in RVO2 paper
            - timeStep = 0.10
            - neighborDist = 3
            - MaxNeighbors = 10
            - AgentTimeHorizon = 3
            - AgentTimeHorizonObst = 1
            - AgentRadius = 0.2
            - AgentMaxSpeed = 3.5
        '''
        # AHHH model's sim
        #sim = rvo2.PyRVOSimulator(0.10, 3, 10, 3, 1, 0.2, 3.5)
        
        # finetuning sim
        #sim = rvo2.PyRVOSimulator(0.10, 5, 10, 1.0, 1, agent_radius, 5.0)
        
        # DFSM finetuning sim
        #sim = rvo2.PyRVOSimulator(0.10, 3, 10, 3, 1, agent_radius, 5.0)
        
        # NEW Finetune comparison sim
        # ORCA1
        #sim = rvo2.PyRVOSimulator(0.10, 3, 10, 3, 1, 0.2, pref_speed+0.5)
        
        # ORCA2
        sim = rvo2.PyRVOSimulator(0.10, 5, 10, 1.0, 1, 0.5, pref_speed+0.5)
        
        # make a new dataframe for this sample
        col_list = ['0_x', '0_y', '1_x', '1_y', '2_x', '2_y', '3_x', '3_y', '4_x', '4_y', '5_x', '5_y', '6_x', '6_y', '7_x', '7_y', '8_x', '8_y', '9_x', '9_y', '10_x', '10_y']
        
        col_list_2 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        
        this_sample_locations = pd.DataFrame(columns=col_list)
        this_sample_current_velocities = pd.DataFrame(columns=col_list)
        this_sample_pref_velocities = pd.DataFrame(columns=col_list)
        #this_sample_pref_velocities_normalized = pd.DataFrame(columns=col_list)
        this_sample_pref_speeds = pd.DataFrame(columns=col_list_2)
        this_sample_dist_from_goal = pd.DataFrame(columns=col_list_2)
        this_sample_goals = pd.DataFrame(columns=col_list)
        
        locations = []
        current_velocities = []
        pref_velocities = []
        #pref_velocities_normalized = []
        pref_speeds = []
        dist_from_goal = []
        goals = []
        
        masked_agents = []
        # pick random number of agents
        num_agents = 11#random.randint(agents_range[0], agents_range[1])
        if num_agents<agents_range[1]:
            for diff in range(agents_range[1] - num_agents):
                masked_agents += [-999, -999]
        
        # choose a random radius
        circle_radius_range = [2, 7]
        circle_radius = random.uniform(circle_radius_range[0], circle_radius_range[1])
        print("circle radius:", circle_radius)
        
        # initialize  each agent
        for agent in range(0,num_agents):
            valid = -1
            while valid == -1:
                # generate random location on the circle to spawn agent
                theta = random.random()*2*math.pi
                x = circle_radius*math.cos(theta)
                y = circle_radius*math.sin(theta)
                
                # calculate goal location (should be directly across the circle)
                goal_x = -x #circle_radius*math.cos(theta + math.pi)
                goal_y = -y #circle_radius*math.sin(theta + math.pi)
                
                if no_overlap([x, y], locations, 2*agent_radius + initial_space_padding) and no_overlap([goal_x, goal_y], goals, 2*agent_radius + initial_space_padding):
                    valid = 1
            
            # set initial velocity (0,0)
            speed = 0.0
            angle = 0.0
            sim.addAgent((x, y))
            sim.setAgentVelocity(agent, (speed*math.cos(angle), speed*math.sin(angle)))
            sim.setAgentPrefVelocity(agent, tuple(pref_speed*normalize_vector([goal_i - curr_i for goal_i, curr_i in zip([goal_x, goal_y], [x, y])])))
            
            
            locations += list(sim.getAgentPosition(agent))
            current_velocities += list(x for x in sim.getAgentVelocity(agent))
            pref_velocities += list(x for x in sim.getAgentPrefVelocity(agent))
            #pref_velocities_normalized+= list(normalize_vector(sim.getAgentPrefVelocity(agent)))
            pref_speeds += [pref_speed]
            dist_from_goal += [circle_radius*2]
            goals += list([goal_x, goal_y])
            
        #print('Simulation has %i agents in it.' % sim.getNumAgents())
        if i%10==0:
            print('Running simulation', i)
        
        # record the initial location of each agent in the scene (pad any nonexistent neighbors with -999)
        list_of_data_list = [locations, current_velocities, pref_velocities, pref_speeds, dist_from_goal, goals]
        list_of_df = [this_sample_locations, this_sample_current_velocities, this_sample_pref_velocities, this_sample_pref_speeds, this_sample_dist_from_goal, this_sample_goals]
        
        # add masked_agents to the end of each list
        '''
        for data_list in list_of_data_list:
            data_list = add_masked_agents(data_list, masked_agents)
        '''
        
        # update the dataframes
        for idx in range(len(list_of_df)):  
            list_of_df[idx] = update_df(list_of_df[idx], 0, list_of_data_list[idx])
        
        # THESE ARE JUST FOR PRINTING. NOT SAVING THE DATA
        agent_pref_velocities = {key: [] for key in list(range(0,num_agents))}
        agent_curr_velocities = {key: [] for key in list(range(0,num_agents))}
        
        for agent_no in list(range(0,num_agents)):
            #print(str(agent_no) + ' preVel: (%.3f, %.3f), currVel: (%.3f, %.3f)' %( sim.getAgentPrefVelocity(agent_no)[0], sim.getAgentPrefVelocity(agent_no)[1], sim.getAgentVelocity(agent_no)[0], sim.getAgentVelocity(agent_no)[1]))
            agent_pref_velocities[agent_no] += [sim.getAgentPrefVelocity(agent_no)]
            agent_curr_velocities[agent_no] += [sim.getAgentVelocity(agent_no)]
            
        step = 1    
        while not all_agents_done(sim, goals, goal_threshold) and step <= max_steps:# for step in range(1, 200):
            sim.doStep()
            calculate_pref_vel(sim, pref_speed, goals, goal_threshold)
            locations = []
            current_velocities = []
            pref_velocities = []
            #pref_velocities_normalized = []
            pref_speeds = []
            dist_from_goal = []
            for agent_no in list(range(0,num_agents)):
                locations+= list(sim.getAgentPosition(agent_no))
                current_velocities += list(x for x in sim.getAgentVelocity(agent_no))
                pref_velocities += list(x for x in sim.getAgentPrefVelocity(agent_no))
                #pref_velocities_normalized += list(normalize_vector(sim.getAgentPrefVelocity(agent_no)))
            
            # make lists for cleaner For loops
            list_of_data_list = [locations, current_velocities, pref_velocities]
            list_of_df = [this_sample_locations, this_sample_current_velocities, this_sample_pref_velocities]
            
            # add masked_agents to the end of each list
            '''
            for data_list in list_of_data_list:
                data_list = add_masked_agents(data_list, masked_agents)
            '''

            # update the dataframes
            for idx in range(len(list_of_df)):  
                list_of_df[idx] = update_df(list_of_df[idx], step, list_of_data_list[idx])

            step += 1
            
            for agent_no in list(range(0,num_agents)):
                #print(str(agent_no) + ' preVel: (%.3f, %.3f), currVel: (%.3f, %.3f)' %( sim.getAgentPrefVelocity(agent_no)[0], sim.getAgentPrefVelocity(agent_no)[1], sim.getAgentVelocity(agent_no)[0], sim.getAgentVelocity(agent_no)[1]))
                agent_pref_velocities[agent_no] += [sim.getAgentPrefVelocity(agent_no)]
                agent_curr_velocities[agent_no] += [sim.getAgentVelocity(agent_no)]
        
        
        print("Simulation done! Number of steps:" , step)
        # plot sample
        if plot_samples:#i==0 or (i+1)%100==0:
            plot_this_sample(i, sim, velocity_scale, this_sample_locations, this_sample_current_velocities, this_sample_pref_velocities, sample_type = 'circle', circle_radius = circle_radius)
        
        samples.append([this_sample_locations, this_sample_current_velocities, this_sample_pref_velocities, this_sample_pref_speeds, this_sample_dist_from_goal, this_sample_goals])
    
    print("done! Checking file integrity...")
    
    # save generated samples to pkl file
    with open(save_file_name + '.pkl', 'wb') as handle:
        pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(save_file_name + '.pkl', 'rb') as handle:
        b = pickle.load(handle)
    
    # check that saved file matches original Python object
    for sample_idx in range(len(samples)):
        for df_idx in range(len(samples[sample_idx])):
            if not all(samples[sample_idx][df_idx].all() == b[sample_idx][df_idx].all()):
                print("MISMATCH IN SAMPLE ", sample_idx, "DF", df_idx)
    
    print('done! Duration: {}'.format(datetime.now() - start_time), "for", num_samples, "samples")
    return   

def generate_n_lineup_samples(num_samples, agents_range, pref_speed, initial_space_padding, velocity_scale, max_steps, save_file_name, plot_samples, goal_threshold, agent_radius):
    start_time = datetime.now()
    samples = []
    for i in range(num_samples):
        # set up the simulator
        '''
        Hyperparameters are set to whatever is listed in RVO2 paper
            - timeStep = 0.10
            - neighborDist = 3
            - MaxNeighbors = 10
            - AgentTimeHorizon = 3
            - AgentTimeHorizonObst = 1
            - AgentRadius = 0.2
            - AgentMaxSpeed = 3.5
        '''
        # AHHH model's sim
        #sim = rvo2.PyRVOSimulator(0.10, 3, 10, 3, 1, 0.2, 3.5)
        
        # finetuning sim
        #sim = rvo2.PyRVOSimulator(0.10, 5, 10, 1.0, 1, agent_radius, 5.0)
        
        # DFSM finetuning sim
        #sim = rvo2.PyRVOSimulator(0.10, 3, 10, 3, 1, agent_radius, 5.0)
        
        # NEW Finetune comparison sim
        # ORCA1
        sim = rvo2.PyRVOSimulator(0.10, 3, 10, 3, 1, 0.2, pref_speed+0.5)
        
        # ORCA2
        #sim = rvo2.PyRVOSimulator(0.10, 5, 10, 1.0, 1, 0.5, pref_speed+0.5)
        
        # make a new dataframe for this sample
        col_list = ['0_x', '0_y', '1_x', '1_y', '2_x', '2_y', '3_x', '3_y', '4_x', '4_y', '5_x', '5_y', '6_x', '6_y', '7_x', '7_y', '8_x', '8_y', '9_x', '9_y', '10_x', '10_y']
        
        col_list_2 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        
        this_sample_locations = pd.DataFrame(columns=col_list)
        this_sample_current_velocities = pd.DataFrame(columns=col_list)
        this_sample_pref_velocities = pd.DataFrame(columns=col_list)
        #this_sample_pref_velocities_normalized = pd.DataFrame(columns=col_list)
        this_sample_pref_speeds = pd.DataFrame(columns=col_list_2)
        this_sample_dist_from_goal = pd.DataFrame(columns=col_list_2)
        this_sample_goals = pd.DataFrame(columns=col_list)
        
        locations = []
        current_velocities = []
        pref_velocities = []
        #pref_velocities_normalized = []
        pref_speeds = []
        dist_from_goal = []
        goals = []
        
        masked_agents = []
        # pick random number of agents
        num_agents = 11#random.randint(agents_range[0], agents_range[1])
        if num_agents<agents_range[1]:
            for diff in range(agents_range[1] - num_agents):
                masked_agents += [-999, -999]
        
        # choose a random distance between the 2 lines of agents
        distance_range = [3,4]#[5,6] #[7, 9]
        distance = random.uniform(distance_range[0], distance_range[1])
        print("distance:", distance)
        
        top_is_true = [0, 1]*(num_agents//2)
        
        if num_agents%2==1:
            top_is_true += [0]
        
        random.shuffle(top_is_true)
        
        # initialize  each agent
        for agent in range(0,num_agents):
            valid = -1
            while valid == -1:
                #print("placing agent", agent)
                # generate random x location for the agent to spawn on (range is -/+)
                x = random.uniform(-distance, distance)
                # randomly choose if we put this agent on top or bottom line
                if top_is_true[agent]:
                    y = distance
                else:
                    y = -distance
                
                # calculate goal location
                goal_x = random.uniform(-distance+1, distance-1) #random.uniform(-distance/2, distance/2)
                goal_y = -y
                
                if no_overlap([x, y], locations, 2*agent_radius + initial_space_padding) and no_overlap([goal_x, goal_y], goals, 2*agent_radius + initial_space_padding):
                    valid = 1
            
            # set initial velocity (0,0)
            speed = 0.0 #random.uniform(speed_range[0], speed_range[1])
            angle = 0.0 #random.uniform(direction_range[0], direction_range[1])
            sim.addAgent((x, y))
            sim.setAgentVelocity(agent, (speed*math.cos(angle), speed*math.sin(angle)))
            sim.setAgentPrefVelocity(agent, tuple(normalize_vector([goal_i - curr_i for goal_i, curr_i in zip([goal_x, goal_y], [x, y])])))
            locations += list(sim.getAgentPosition(agent))
            current_velocities += list(x for x in sim.getAgentVelocity(agent))
            pref_velocities += list(x for x in sim.getAgentPrefVelocity(agent))
            #pref_velocities_normalized+= list(normalize_vector(sim.getAgentPrefVelocity(agent)))
            pref_speeds += [pref_speed]
            dist_from_goal += [distance*2]
            goals += list([goal_x, goal_y])
            
        #print('Simulation has %i agents in it.' % sim.getNumAgents())
        if i%1000==0:
            print('Running simulation', i)
        
        # record the initial location of each agent in the scene (pad any nonexistent neighbors with -999)
        list_of_data_list = [locations, current_velocities, pref_velocities, pref_speeds, dist_from_goal, goals]
        list_of_df = [this_sample_locations, this_sample_current_velocities, this_sample_pref_velocities, this_sample_pref_speeds, this_sample_dist_from_goal, this_sample_goals]
        
        # add masked_agents to the end of each list
        '''
        for data_list in list_of_data_list:
            data_list = add_masked_agents(data_list, masked_agents)
        '''

        # update the dataframes
        for idx in range(len(list_of_df)):  
            list_of_df[idx] = update_df(list_of_df[idx], 0, list_of_data_list[idx])
        
        # THESE ARE JUST FOR PRINTING. NOT SAVING THE DATA
        agent_pref_velocities = {key: [] for key in list(range(0,num_agents))}
        agent_curr_velocities = {key: [] for key in list(range(0,num_agents))}
        
        for agent_no in list(range(0,num_agents)):
            #print(str(agent_no) + ' preVel: (%.3f, %.3f), currVel: (%.3f, %.3f)' %( sim.getAgentPrefVelocity(agent_no)[0], sim.getAgentPrefVelocity(agent_no)[1], sim.getAgentVelocity(agent_no)[0], sim.getAgentVelocity(agent_no)[1]))
            agent_pref_velocities[agent_no] += [sim.getAgentPrefVelocity(agent_no)]
            agent_curr_velocities[agent_no] += [sim.getAgentVelocity(agent_no)]
            
        step = 1
        #for step in range(0, 5):
        while not all_agents_done(sim, goals, goal_threshold) and step <= max_steps:# for step in range(1, 200):
            sim.doStep()
            calculate_pref_vel(sim, pref_speed, goals, goal_threshold)
            locations = []
            current_velocities = []
            pref_velocities = []
            #pref_velocities_normalized = []
            pref_speeds = []
            dist_from_goal = []
            for agent_no in list(range(0,num_agents)):
                locations+= list(sim.getAgentPosition(agent_no))
                current_velocities += list(x for x in sim.getAgentVelocity(agent_no))
                pref_velocities += list(x for x in sim.getAgentPrefVelocity(agent_no))
                #pref_velocities_normalized += list(normalize_vector(sim.getAgentPrefVelocity(agent_no)))
            
            # make lists for cleaner For loops
            list_of_data_list = [locations, current_velocities, pref_velocities]
            list_of_df = [this_sample_locations, this_sample_current_velocities, this_sample_pref_velocities]
            
            # add masked_agents to the end of each list
            '''
            for data_list in list_of_data_list:
                data_list = add_masked_agents(data_list, masked_agents)
            '''

            # update the dataframes
            for idx in range(len(list_of_df)):  
                list_of_df[idx] = update_df(list_of_df[idx], step, list_of_data_list[idx])

            step += 1
            #print(step)
            
            for agent_no in list(range(0,num_agents)):
                #print(str(agent_no) + ' preVel: (%.3f, %.3f), currVel: (%.3f, %.3f)' %( sim.getAgentPrefVelocity(agent_no)[0], sim.getAgentPrefVelocity(agent_no)[1], sim.getAgentVelocity(agent_no)[0], sim.getAgentVelocity(agent_no)[1]))
                agent_pref_velocities[agent_no] += [sim.getAgentPrefVelocity(agent_no)]
                agent_curr_velocities[agent_no] += [sim.getAgentVelocity(agent_no)]
        
        
        print("Simulation done! Number of steps:" , step)
        # plot sample
        if plot_samples:#i==0 or (i+1)%100==0:
            plot_this_sample(i, sim, velocity_scale, this_sample_locations, this_sample_current_velocities, this_sample_pref_velocities, sample_type = 'lineup', circle_radius = None, line_dist = distance)
        
        samples.append([this_sample_locations, this_sample_current_velocities, this_sample_pref_velocities, this_sample_pref_speeds, this_sample_dist_from_goal, this_sample_goals])
    
    print("done! Checking file integrity...")
    
    # save generated samples to pkl file
    with open(save_file_name + '.pkl', 'wb') as handle:
        pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(save_file_name + '.pkl', 'rb') as handle:
        b = pickle.load(handle)
    
    # check that saved file matches original Python object
    for sample_idx in range(len(samples)):
        for df_idx in range(len(samples[sample_idx])):
            if not all(samples[sample_idx][df_idx].all() == b[sample_idx][df_idx].all()):
                print("MISMATCH IN SAMPLE ", sample_idx, "DF", df_idx)
    
    print('done! Duration: {}'.format(datetime.now() - start_time), "for", num_samples, "samples")
    return                

def generate_sample_from_file(load_file, agents_range, pref_speed, initial_space_padding, velocity_scale, max_steps, save_file_name, plot_samples, goal_threshold, agent_radius):
    start_time = datetime.now()
    samples = []
    # set up the simulator
    '''
    Hyperparameters are set to whatever is listed in RVO2 paper
        - timeStep = 0.10
        - neighborDist = 3
        - MaxNeighbors = 10
        - AgentTimeHorizon = 3
        - AgentTimeHorizonObst = 1
        - AgentRadius = 0.2
        - AgentMaxSpeed = 3.5
    '''
    # AHHH model's sim
    #sim = rvo2.PyRVOSimulator(0.10, 3, 10, 3, 1, 0.2, 3.5)
    
    # finetuning sim
    #sim = rvo2.PyRVOSimulator(0.10, 5, 10, 1.0, 1, agent_radius, 5.0)
    
    # DFSM finetuning sim
    #sim = rvo2.PyRVOSimulator(0.10, 3, 10, 3, 1, agent_radius, 5.0)
    
    # NEW Finetune comparison sim
    # ORCA1
    #sim = rvo2.PyRVOSimulator(0.10, 3, 10, 3, 1, 0.2, pref_speed+0.5)
    
    # ORCA2
    sim = rvo2.PyRVOSimulator(0.10, 5, 10, 1.0, 1, 0.5, pref_speed+0.5)
    
    
    # Load pkl file and get the first timestep
    with open(load_file, 'rb') as handle:
        sample_list = pickle.load(handle)
        
    for sample_idx in range(len(sample_list)):
        this_sample = sample_list[sample_idx]
        '''
        The sample_list should be a list where len(sample_list) = number of generated simulations.
        Each element of sample_list should be another list of 4 elements:
            0. current location dataframe (x, y)
            1. current velocity dataframe (x, y)
            2. preferred velocity dataframe (x, y)
            3. preferred speed dataframe (float)
            4. distance from goal dataframe (float)
            5. goal location dataframe (x, y)
        '''
    
        # make a new dataframe for this sample
        col_list = ['0_x', '0_y', '1_x', '1_y', '2_x', '2_y', '3_x', '3_y', '4_x', '4_y', '5_x', '5_y', '6_x', '6_y', '7_x', '7_y', '8_x', '8_y', '9_x', '9_y', '10_x', '10_y']
        
        col_list_2 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        
        this_sample_locations = pd.DataFrame(columns=col_list)
        this_sample_current_velocities = pd.DataFrame(columns=col_list)
        this_sample_pref_velocities = pd.DataFrame(columns=col_list)
        #this_sample_pref_velocities_normalized = pd.DataFrame(columns=col_list)
        this_sample_pref_speeds = pd.DataFrame(columns=col_list_2)
        this_sample_dist_from_goal = pd.DataFrame(columns=col_list_2)
        this_sample_goals = pd.DataFrame(columns=col_list)
        
        locations = list(this_sample[0].iloc[0])
        current_velocities = list(this_sample[1].iloc[0])
        pref_velocities = list(this_sample[2].iloc[0])
        #pref_velocities_normalized = []
        pref_speeds = list(this_sample[3].iloc[0])
        dist_from_goal = list(this_sample[4].iloc[0])
        goals = list(this_sample[5].iloc[0])
        
        distance = dist_from_goal[0]/2
        
        masked_agents = []
        # pick random number of agents
        num_agents = 11#random.randint(agents_range[0], agents_range[1])
        if num_agents<agents_range[1]:
            for diff in range(agents_range[1] - num_agents):
                masked_agents += [-999, -999]
        
        # initialize  each agent
        for agent in range(0,num_agents):
            
            x = locations[agent*2]
            y = locations[agent*2+1]
            
            # calculate goal location (should be directly across the circle)
            goal_x = goals[agent*2]
            goal_y = goals[agent*2+1]
                
            
            # set initial velocity (0,0)
            sim.addAgent((x, y))
            sim.setAgentVelocity(agent, (current_velocities[agent*2], current_velocities[agent*2+1]))
            sim.setAgentPrefVelocity(agent, tuple(pref_speed*normalize_vector([goal_i - curr_i for goal_i, curr_i in zip([goal_x, goal_y], [x, y])])))
            
            '''
            locations += list(sim.getAgentPosition(agent))
            current_velocities += list(x for x in sim.getAgentVelocity(agent))
            pref_velocities += list(x for x in sim.getAgentPrefVelocity(agent))
            #pref_velocities_normalized+= list(normalize_vector(sim.getAgentPrefVelocity(agent)))
            goals += list([goal_x, goal_y])
            '''
            
        print('Running simulation')
        
        # record the initial location of each agent in the scene (pad any nonexistent neighbors with -999)
        list_of_data_list = [locations, current_velocities, pref_velocities, pref_speeds, dist_from_goal, goals]
        list_of_df = [this_sample_locations, this_sample_current_velocities, this_sample_pref_velocities, this_sample_pref_speeds, this_sample_dist_from_goal, this_sample_goals]
        
        # add masked_agents to the end of each list
        '''
        for data_list in list_of_data_list:
            data_list = add_masked_agents(data_list, masked_agents)
        '''

        # update the dataframes
        for idx in range(len(list_of_df)):  
            list_of_df[idx] = update_df(list_of_df[idx], 0, list_of_data_list[idx])
        
        # THESE ARE JUST FOR PRINTING. NOT SAVING THE DATA
        agent_pref_velocities = {key: [] for key in list(range(0,num_agents))}
        agent_curr_velocities = {key: [] for key in list(range(0,num_agents))}
        
        for agent_no in list(range(0,num_agents)):
            #print(str(agent_no) + ' preVel: (%.3f, %.3f), currVel: (%.3f, %.3f)' %( sim.getAgentPrefVelocity(agent_no)[0], sim.getAgentPrefVelocity(agent_no)[1], sim.getAgentVelocity(agent_no)[0], sim.getAgentVelocity(agent_no)[1]))
            agent_pref_velocities[agent_no] += [sim.getAgentPrefVelocity(agent_no)]
            agent_curr_velocities[agent_no] += [sim.getAgentVelocity(agent_no)]
            
        step = 1    
        while not all_agents_done(sim, goals, goal_threshold) and step <= max_steps:# for step in range(1, 200):
            sim.doStep()
            calculate_pref_vel(sim, pref_speed, goals, goal_threshold)
            locations = []
            current_velocities = []
            pref_velocities = []
            #pref_velocities_normalized = []
            pref_speeds = []
            dist_from_goal = []
            for agent_no in list(range(0,num_agents)):
                locations+= list(sim.getAgentPosition(agent_no))
                current_velocities += list(x for x in sim.getAgentVelocity(agent_no))
                pref_velocities += list(x for x in sim.getAgentPrefVelocity(agent_no))
                #pref_velocities_normalized += list(normalize_vector(sim.getAgentPrefVelocity(agent_no)))
            
            # make lists for cleaner For loops
            list_of_data_list = [locations, current_velocities, pref_velocities]
            list_of_df = [this_sample_locations, this_sample_current_velocities, this_sample_pref_velocities]
            
            # add masked_agents to the end of each list
            '''
            for data_list in list_of_data_list:
                data_list = add_masked_agents(data_list, masked_agents)
            '''
            
            # update the dataframes
            for idx in range(len(list_of_df)):  
                list_of_df[idx] = update_df(list_of_df[idx], step, list_of_data_list[idx])

            step += 1
            
            for agent_no in list(range(0,num_agents)):
                #print(str(agent_no) + ' preVel: (%.3f, %.3f), currVel: (%.3f, %.3f)' %( sim.getAgentPrefVelocity(agent_no)[0], sim.getAgentPrefVelocity(agent_no)[1], sim.getAgentVelocity(agent_no)[0], sim.getAgentVelocity(agent_no)[1]))
                agent_pref_velocities[agent_no] += [sim.getAgentPrefVelocity(agent_no)]
                agent_curr_velocities[agent_no] += [sim.getAgentVelocity(agent_no)]
        
        
        print("Simulation done! Number of steps:" , step)
        # plot sample
        if plot_samples:#i==0 or (i+1)%100==0:
            if 'circle' in load_file:
                plot_this_sample(sample_idx, sim, velocity_scale, this_sample_locations, this_sample_current_velocities, this_sample_pref_velocities, sample_type = 'circle', circle_radius = distance, line_dist = None)
            elif 'lineup' in load_file:
                plot_this_sample(sample_idx, sim, velocity_scale, this_sample_locations, this_sample_current_velocities, this_sample_pref_velocities, sample_type = 'lineup', circle_radius = None, line_dist = distance)
        
        samples.append([this_sample_locations, this_sample_current_velocities, this_sample_pref_velocities, this_sample_pref_speeds, this_sample_dist_from_goal, this_sample_goals])
    
    print("done! Checking file integrity...")
    
    # save generated samples to pkl file
    with open(save_file_name + '.pkl', 'wb') as handle:
        pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(save_file_name + '.pkl', 'rb') as handle:
        b = pickle.load(handle)
    
    # check that saved file matches original Python object
    for sample_idx in range(len(samples)):
        for df_idx in range(len(samples[sample_idx])):
            if not all(samples[sample_idx][df_idx].all() == b[sample_idx][df_idx].all()):
                print("MISMATCH IN SAMPLE ", sample_idx, "DF", df_idx)
    
    print('done! Duration: {}'.format(datetime.now() - start_time), "for", len(samples), "samples")
    return

# DON'T USE FOR DATA GENERATION (this was for making a very specific proof of concept)
def generate_n_agents_samples(num_samples, agents_range, pref_speed, initial_space_padding, velocity_scale, max_steps, save_file_name, plot_samples, goal_threshold, agent_radius):
    start_time = datetime.now()
    samples = []
    for i in range(num_samples):
        # set up the simulator
        '''
        Hyperparameters are set to whatever is listed in RVO2 paper
            - timeStep = 0.10
            - neighborDist = 3
            - MaxNeighbors = 10
            - AgentTimeHorizon = 3
            - AgentTimeHorizonObst = 1
            - AgentRadius = 0.2
            - AgentMaxSpeed = 3.5
        '''
        
        # default settings for agents (counterclockwise vertices makes filled in obstacles)
        sim = rvo2.PyRVOSimulator(0.10, 3, 10, 1, 1, 0.2, pref_speed+0.5)
        
        # add obstacles to the scene
        # same obsatcles as NN
        '''
        o1_vert = [(1,3), (1, 1), (4, 1), (4, 3)]
        o2_vert = [(-3.5, 3), (-3.5, 1), (-1.5, 1), (-1.5, 3)]
        '''
        
        # narrower space bewteen obstacles
        
        o1_vert = [(0.5,3), (0.5, 1), (4, 1), (4, 3)]
        o2_vert = [(-3.5, 3), (-3.5, 1), (-1, 1), (-1, 3)]
        
        
        # walls (clockwise vertices creates negative obstable, where polygon is perimeter of environment)
        wall1 = [(-5,10), (10, 10), (10, -6), (-5, -6)]
        
        
        sim.addObstacle(o1_vert)
        sim.addObstacle(o2_vert)
        sim.addObstacle(wall1)
        sim.processObstacles()
        
        # make a new dataframe for this sample
        col_list = ['0_x', '0_y', '1_x', '1_y', '2_x', '2_y', '3_x', '3_y', '4_x', '4_y', '5_x', '5_y', '6_x', '6_y', '7_x', '7_y', '8_x', '8_y', '9_x', '9_y', '10_x', '10_y']
        
        col_list_2 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        
        this_sample_locations = pd.DataFrame(columns=col_list)
        this_sample_current_velocities = pd.DataFrame(columns=col_list)
        this_sample_pref_velocities = pd.DataFrame(columns=col_list)
        #this_sample_pref_velocities_normalized = pd.DataFrame(columns=col_list)
        this_sample_pref_speeds = pd.DataFrame(columns=col_list_2)
        this_sample_dist_from_goal = pd.DataFrame(columns=col_list_2)
        this_sample_goals = pd.DataFrame(columns=col_list)
        
        locations = []
        current_velocities = []
        pref_velocities = []
        #pref_velocities_normalized = []
        pref_speeds = []
        dist_from_goal = []
        goals = []
        
        masked_agents = []
        # pick random number of agents
        num_agents = 9#random.randint(agents_range[0], agents_range[1])
        
        # place agents (They will be indexed 0-8)
        count = 0
        location_robot=[[-1,-2],[0,-2],[1,-2],
                        [-1,-3],[0,-3],[1,-3],
                        [-1,-4],[0,-4],[1,-4]] 
        # final goals
        '''
        goal_main=[[-1.5,7],[0,7],[1.5,7],
                   [-1.5,6],[0,6],[1.5,6],
                   [-1.5,5],[0,5],[1.5,5]]
        '''
        
        # sub goals (for NN obstacles)
        '''
        goal_main=[[-1.5,7],[0,7],[8,3.5],
                   [-1.5,6],[0,6],[8,2.5],
                   [-1.5,5],[0,5],[8,1.5]]
        '''
        
        # sub goals (for smaller space)
        goal_main=[[-4.5,3],[0,7],[8,3.5],
                   [-4.5,2],[0,6],[8,2.5],
                   [-4.5,1],[0,5],[8,1.5]]
        
        for i in range(3):
            for j in range(1, 4):
                speed = 0.0
                angle = 0.0
                x = location_robot[count][0]
                y = location_robot[count][1]
                goal_x = goal_main[count][0]
                goal_y = goal_main[count][1]
                sim.addAgent((x, y))
                sim.setAgentVelocity(count, (speed*math.cos(angle), speed*math.sin(angle)))
                sim.setAgentPrefVelocity(count, tuple(pref_speed*normalize_vector([goal_i - curr_i for goal_i, curr_i in zip([goal_x, goal_y], [x, y])])))
                locations += list(sim.getAgentPosition(count))
                current_velocities += list(h for h in sim.getAgentVelocity(count))
                pref_velocities += list(h for h in sim.getAgentPrefVelocity(count))
                #pref_velocities_normalized+= list(normalize_vector(sim.getAgentPrefVelocity(agent)))
                pref_speeds += [pref_speed]
                dist_from_goal += [4]
                goals += list([goal_x, goal_y])
                count +=1
        
        '''
        # choose a random radius
        circle_radius_range = [2, 7]
        circle_radius = 4 #random.uniform(circle_radius_range[0], circle_radius_range[1])
        print("circle radius:", circle_radius)

        # Add 2 default agents
        theta = 0.5
        x = circle_radius*math.cos(theta + math.pi)
        y = circle_radius*math.sin(theta + math.pi)
        
        # calculate goal location (should be directly across the circle)
        goal_x = -x #circle_radius*math.cos(theta + math.pi)
        goal_y = -y #circle_radius*math.sin(theta + math.pi)
        
        speed = 0.0
        angle = 0.0
        sim.addAgent((x, y))
        sim.setAgentVelocity(0, (speed*math.cos(angle), speed*math.sin(angle)))
        sim.setAgentPrefVelocity(0, tuple(pref_speed*normalize_vector([goal_i - curr_i for goal_i, curr_i in zip([goal_x, goal_y], [x, y])])))
        
        locations += list(sim.getAgentPosition(0))
        current_velocities += list(x for x in sim.getAgentVelocity(0))
        pref_velocities += list(x for x in sim.getAgentPrefVelocity(0))
        #pref_velocities_normalized+= list(normalize_vector(sim.getAgentPrefVelocity(agent)))
        pref_speeds += [pref_speed]
        dist_from_goal += [circle_radius*2]
        goals += list([goal_x, goal_y])
        
        x = -x
        y = -y
        
        # calculate goal location (should be directly across the circle)
        goal_x = -x #circle_radius*math.cos(theta + math.pi)
        goal_y = -y #circle_radius*math.sin(theta + math.pi)
        
        speed = 0.0
        angle = 0.0
        #sim.addAgent((x, y))
        sim.addAgent((x, y))
        sim.setAgentVelocity(1, (speed*math.cos(angle), speed*math.sin(angle)))
        sim.setAgentPrefVelocity(1, tuple(pref_speed*normalize_vector([goal_i - curr_i for goal_i, curr_i in zip([goal_x, goal_y], [x, y])])))
        
        locations += list(sim.getAgentPosition(1))
        current_velocities += list(x for x in sim.getAgentVelocity(1))
        pref_velocities += list(x for x in sim.getAgentPrefVelocity(1))
        #pref_velocities_normalized+= list(normalize_vector(sim.getAgentPrefVelocity(agent)))
        pref_speeds += [pref_speed]
        dist_from_goal += [circle_radius*2]
        goals += list([goal_x, goal_y])
        
        # Add 2 other agents
        theta = 1.5
        x = circle_radius*math.cos(theta + math.pi)
        y = circle_radius*math.sin(theta + math.pi)
        
        # calculate goal location (should be directly across the circle)
        goal_x = -x #circle_radius*math.cos(theta + math.pi)
        goal_y = -y #circle_radius*math.sin(theta + math.pi)
        
        speed = 0.0
        angle = 0.0
        sim.addAgent((x, y))
        #sim.addAgent((x, y), 10, 10, 0.5, 2, 0.5 ,pref_speed*0.5+0.5, (0,0))
        sim.setAgentVelocity(2, (speed*math.cos(angle), speed*math.sin(angle)))
        sim.setAgentPrefVelocity(2, tuple(pref_speed*normalize_vector([goal_i - curr_i for goal_i, curr_i in zip([goal_x, goal_y], [x, y])])))
        
        locations += list(sim.getAgentPosition(2))
        current_velocities += list(x for x in sim.getAgentVelocity(2))
        pref_velocities += list(x for x in sim.getAgentPrefVelocity(2))
        #pref_velocities_normalized+= list(normalize_vector(sim.getAgentPrefVelocity(agent)))
        pref_speeds += [pref_speed]
        dist_from_goal += [circle_radius*2]
        goals += list([goal_x, goal_y])     
        
        x = -x
        y = -y
        
        # calculate goal location (should be directly across the circle)
        goal_x = -x #circle_radius*math.cos(theta + math.pi)
        goal_y = -y #circle_radius*math.sin(theta + math.pi)
        
        speed = 0.0
        angle = 0.0
        sim.addAgent((x, y))
        sim.setAgentVelocity(3, (speed*math.cos(angle), speed*math.sin(angle)))
        sim.setAgentPrefVelocity(3, tuple(pref_speed*normalize_vector([goal_i - curr_i for goal_i, curr_i in zip([goal_x, goal_y], [x, y])])))
        
        locations += list(sim.getAgentPosition(3))
        current_velocities += list(x for x in sim.getAgentVelocity(3))
        pref_velocities += list(x for x in sim.getAgentPrefVelocity(3))
        #pref_velocities_normalized+= list(normalize_vector(sim.getAgentPrefVelocity(agent)))
        pref_speeds += [pref_speed]
        dist_from_goal += [circle_radius*2]
        goals += list([goal_x, goal_y])
        '''
        
        #print('Simulation has %i agents in it.' % sim.getNumAgents())
        if i%1000==0:
            print('Running simulation', i)
        
        # record the initial location of each agent in the scene (pad any nonexistent neighbors with -999)
        list_of_data_list = [locations, current_velocities, pref_velocities, pref_speeds, dist_from_goal, goals]
        list_of_df = [this_sample_locations, this_sample_current_velocities, this_sample_pref_velocities, this_sample_pref_speeds, this_sample_dist_from_goal, this_sample_goals]
        
        # add masked_agents to the end of each list
        '''
        for data_list in list_of_data_list:
            data_list = add_masked_agents(data_list, masked_agents)
        '''

        # update the dataframes
        for idx in range(len(list_of_df)):  
            list_of_df[idx] = update_df(list_of_df[idx], 0, list_of_data_list[idx])
        
        # THESE ARE JUST FOR PRINTING. NOT SAVING THE DATA
        agent_pref_velocities = {key: [] for key in list(range(0,num_agents))}
        agent_curr_velocities = {key: [] for key in list(range(0,num_agents))}
        
        for agent_no in list(range(0,num_agents)):
            #print(str(agent_no) + ' preVel: (%.3f, %.3f), currVel: (%.3f, %.3f)' %( sim.getAgentPrefVelocity(agent_no)[0], sim.getAgentPrefVelocity(agent_no)[1], sim.getAgentVelocity(agent_no)[0], sim.getAgentVelocity(agent_no)[1]))
            agent_pref_velocities[agent_no] += [sim.getAgentPrefVelocity(agent_no)]
            agent_curr_velocities[agent_no] += [sim.getAgentVelocity(agent_no)]
            
        step = 1    
        while not all_agents_done(sim, goals, goal_threshold) and step <= max_steps:# for step in range(1, 200):
            if step ==30:
                main_goal = [[-1.5,7],[0,7],[8,3.5],
                             [-1.5,6],[0,6],[8,2.5],
                             [-1.5,5],[0,5],[8,1.5]]
                goals = [item for sublist in main_goal for item in sublist]
            if step ==40:
                main_goal = [[-1.5,7],[0,7],[1.5,7],
                             [-1.5,6],[0,6],[1.5,6],
                             [-1.5,5],[0,5],[1.5,5]]
                goals = [item for sublist in main_goal for item in sublist]
            sim.doStep()
            calculate_pref_vel(sim, pref_speed, goals, goal_threshold)
            locations = []
            current_velocities = []
            pref_velocities = []
            #pref_velocities_normalized = []
            pref_speeds = []
            dist_from_goal = []
            for agent_no in list(range(0,num_agents)):
                locations+= list(sim.getAgentPosition(agent_no))
                current_velocities += list(x for x in sim.getAgentVelocity(agent_no))
                pref_velocities += list(x for x in sim.getAgentPrefVelocity(agent_no))
                #pref_velocities_normalized += list(normalize_vector(sim.getAgentPrefVelocity(agent_no)))
            
            # make lists for cleaner For loops
            list_of_data_list = [locations, current_velocities, pref_velocities]
            list_of_df = [this_sample_locations, this_sample_current_velocities, this_sample_pref_velocities]
            
            # add masked_agents to the end of each list
            '''
            for data_list in list_of_data_list:
                data_list = add_masked_agents(data_list, masked_agents)
            '''

            # update the dataframes
            for idx in range(len(list_of_df)):  
                list_of_df[idx] = update_df(list_of_df[idx], step, list_of_data_list[idx])

            step += 1
            
            for agent_no in list(range(0,num_agents)):
                #print(str(agent_no) + ' preVel: (%.3f, %.3f), currVel: (%.3f, %.3f)' %( sim.getAgentPrefVelocity(agent_no)[0], sim.getAgentPrefVelocity(agent_no)[1], sim.getAgentVelocity(agent_no)[0], sim.getAgentVelocity(agent_no)[1]))
                agent_pref_velocities[agent_no] += [sim.getAgentPrefVelocity(agent_no)]
                agent_curr_velocities[agent_no] += [sim.getAgentVelocity(agent_no)]
        
        
        print("Simulation done! Number of steps:" , step)
        # plot sample
        if plot_samples:#i==0 or (i+1)%100==0:
            plot_this_sample(i, sim, velocity_scale, this_sample_locations, this_sample_current_velocities, this_sample_pref_velocities, sample_type = 'n_agents')
        
        samples.append([this_sample_locations, this_sample_current_velocities, this_sample_pref_velocities, this_sample_pref_speeds, this_sample_dist_from_goal, this_sample_goals])
    
    print("done! Checking file integrity...")
    
    # save generated samples to pkl file
    with open(save_file_name + '.pkl', 'wb') as handle:
        pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(save_file_name + '.pkl', 'rb') as handle:
        b = pickle.load(handle)
    
    # check that saved file matches original Python object
    for sample_idx in range(len(samples)):
        for df_idx in range(len(samples[sample_idx])):
            if not all(samples[sample_idx][df_idx].all() == b[sample_idx][df_idx].all()):
                print("MISMATCH IN SAMPLE ", sample_idx, "DF", df_idx)
    
    print('done! Duration: {}'.format(datetime.now() - start_time), "for", num_samples, "samples")
    return

def generate_traj_clustering_circle_samples(num_samples, agents_range, pref_speed, initial_space_padding, velocity_scale, max_steps, save_file_name, plot_samples, goal_threshold, agent_radius):
    start_time = datetime.now()
    samples = []
    for i in range(num_samples):
        # set up the simulator
        '''
        Hyperparameters are set to whatever is listed in RVO2 paper
            - timeStep = 0.10
            - neighborDist = 3
            - MaxNeighbors = 10
            - AgentTimeHorizon = 3
            - AgentTimeHorizonObst = 1
            - AgentRadius = 0.2
            - AgentMaxSpeed = 3.5
        '''
        
        # default settings for agents (counterclockwise vertices makes filled in obstacles)
        sim = rvo2.PyRVOSimulator(0.10, 4, 10, 4, 1, 0.2, pref_speed + 0.5)
        
        
        # make a new dataframe for this sample
        col_list = ['0_x', '0_y', '1_x', '1_y', '2_x', '2_y', '3_x', '3_y', '4_x', '4_y', '5_x', '5_y', '6_x', '6_y', '7_x', '7_y', '8_x', '8_y', '9_x', '9_y', '10_x', '10_y']
        
        col_list_2 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        
        this_sample_locations = pd.DataFrame(columns=col_list)
        this_sample_current_velocities = pd.DataFrame(columns=col_list)
        this_sample_pref_velocities = pd.DataFrame(columns=col_list)
        #this_sample_pref_velocities_normalized = pd.DataFrame(columns=col_list)
        this_sample_pref_speeds = pd.DataFrame(columns=col_list_2)
        this_sample_dist_from_goal = pd.DataFrame(columns=col_list_2)
        this_sample_goals = pd.DataFrame(columns=col_list)
        
        locations = []
        current_velocities = []
        pref_velocities = []
        #pref_velocities_normalized = []
        pref_speeds = []
        dist_from_goal = []
        goals = []
        
        masked_agents = []
        # pick random number of agents
        num_agents = 4#random.randint(agents_range[0], agents_range[1])
        
        
        # choose a random radius
        circle_radius_range = [2, 7]
        circle_radius = 2.5 #random.uniform(circle_radius_range[0], circle_radius_range[1])
        print("circle radius:", circle_radius)

        # Add 2 default agents
        theta = 0.5
        x = circle_radius*math.cos(theta + math.pi)
        y = circle_radius*math.sin(theta + math.pi)
        
        # calculate goal location (should be directly across the circle)
        goal_x = -x #circle_radius*math.cos(theta + math.pi)
        goal_y = -y #circle_radius*math.sin(theta + math.pi)
        
        speed = 0.0
        angle = 0.0
        sim.addAgent((x, y))
        sim.setAgentVelocity(0, (speed*math.cos(angle), speed*math.sin(angle)))
        sim.setAgentPrefVelocity(0, tuple(pref_speed*normalize_vector([goal_i - curr_i for goal_i, curr_i in zip([goal_x, goal_y], [x, y])])))
        
        locations += list(sim.getAgentPosition(0))
        current_velocities += list(x for x in sim.getAgentVelocity(0))
        pref_velocities += list(x for x in sim.getAgentPrefVelocity(0))
        #pref_velocities_normalized+= list(normalize_vector(sim.getAgentPrefVelocity(agent)))
        pref_speeds += [pref_speed]
        dist_from_goal += [circle_radius*2]
        goals += list([goal_x, goal_y])
        
        x = -x
        y = -y
        
        # calculate goal location (should be directly across the circle)
        goal_x = -x #circle_radius*math.cos(theta + math.pi)
        goal_y = -y #circle_radius*math.sin(theta + math.pi)
        
        speed = 0.0
        angle = 0.0
        #sim.addAgent((x, y))
        sim.addAgent((x,y), 1, 10, 1, 1, 0.2, pref_speed + 0.5, (0, 0))
        sim.setAgentVelocity(1, (speed*math.cos(angle), speed*math.sin(angle)))
        sim.setAgentPrefVelocity(1, tuple(pref_speed*normalize_vector([goal_i - curr_i for goal_i, curr_i in zip([goal_x, goal_y], [x, y])])))
        
        locations += list(sim.getAgentPosition(1))
        current_velocities += list(x for x in sim.getAgentVelocity(1))
        pref_velocities += list(x for x in sim.getAgentPrefVelocity(1))
        #pref_velocities_normalized+= list(normalize_vector(sim.getAgentPrefVelocity(agent)))
        pref_speeds += [pref_speed]
        dist_from_goal += [circle_radius*2]
        goals += list([goal_x, goal_y])
        
        # Add 2 other agents
        theta = 1.5
        x = circle_radius*math.cos(theta + math.pi)
        y = circle_radius*math.sin(theta + math.pi)
        
        # calculate goal location (should be directly across the circle)
        goal_x = -x #circle_radius*math.cos(theta + math.pi)
        goal_y = -y #circle_radius*math.sin(theta + math.pi)
        
        speed = 0.0
        angle = 0.0
        sim.addAgent((x, y))
        #sim.addAgent((x, y), 10, 10, 0.5, 2, 0.5 ,pref_speed*0.5+0.5, (0,0))
        sim.setAgentVelocity(2, (speed*math.cos(angle), speed*math.sin(angle)))
        sim.setAgentPrefVelocity(2, tuple(pref_speed*normalize_vector([goal_i - curr_i for goal_i, curr_i in zip([goal_x, goal_y], [x, y])])))
        
        locations += list(sim.getAgentPosition(2))
        current_velocities += list(x for x in sim.getAgentVelocity(2))
        pref_velocities += list(x for x in sim.getAgentPrefVelocity(2))
        #pref_velocities_normalized+= list(normalize_vector(sim.getAgentPrefVelocity(agent)))
        pref_speeds += [pref_speed]
        dist_from_goal += [circle_radius*2]
        goals += list([goal_x, goal_y])     
        
        x = -x
        y = -y
        
        # calculate goal location (should be directly across the circle)
        goal_x = -x #circle_radius*math.cos(theta + math.pi)
        goal_y = -y #circle_radius*math.sin(theta + math.pi)
        
        speed = 0.0
        angle = 0.0
        sim.addAgent((x,y), 1, 10, 1, 1, 0.2, pref_speed + 0.5, (0, 0))
        sim.setAgentVelocity(3, (speed*math.cos(angle), speed*math.sin(angle)))
        sim.setAgentPrefVelocity(3, tuple(pref_speed*normalize_vector([goal_i - curr_i for goal_i, curr_i in zip([goal_x, goal_y], [x, y])])))
        
        locations += list(sim.getAgentPosition(3))
        current_velocities += list(x for x in sim.getAgentVelocity(3))
        pref_velocities += list(x for x in sim.getAgentPrefVelocity(3))
        #pref_velocities_normalized+= list(normalize_vector(sim.getAgentPrefVelocity(agent)))
        pref_speeds += [pref_speed]
        dist_from_goal += [circle_radius*2]
        goals += list([goal_x, goal_y])
        
        
        #print('Simulation has %i agents in it.' % sim.getNumAgents())
        if i%1000==0:
            print('Running simulation', i)
        
        # record the initial location of each agent in the scene (pad any nonexistent neighbors with -999)
        list_of_data_list = [locations, current_velocities, pref_velocities, pref_speeds, dist_from_goal, goals]
        list_of_df = [this_sample_locations, this_sample_current_velocities, this_sample_pref_velocities, this_sample_pref_speeds, this_sample_dist_from_goal, this_sample_goals]
        
        # add masked_agents to the end of each list
        '''
        for data_list in list_of_data_list:
            data_list = add_masked_agents(data_list, masked_agents)
        '''

        # update the dataframes
        for idx in range(len(list_of_df)):  
            list_of_df[idx] = update_df(list_of_df[idx], 0, list_of_data_list[idx])
        
        # THESE ARE JUST FOR PRINTING. NOT SAVING THE DATA
        agent_pref_velocities = {key: [] for key in list(range(0,num_agents))}
        agent_curr_velocities = {key: [] for key in list(range(0,num_agents))}
        
        for agent_no in list(range(0,num_agents)):
            #print(str(agent_no) + ' preVel: (%.3f, %.3f), currVel: (%.3f, %.3f)' %( sim.getAgentPrefVelocity(agent_no)[0], sim.getAgentPrefVelocity(agent_no)[1], sim.getAgentVelocity(agent_no)[0], sim.getAgentVelocity(agent_no)[1]))
            agent_pref_velocities[agent_no] += [sim.getAgentPrefVelocity(agent_no)]
            agent_curr_velocities[agent_no] += [sim.getAgentVelocity(agent_no)]
            
        step = 1    
        while not all_agents_done(sim, goals, goal_threshold) and step <= max_steps:# for step in range(1, 200):
            sim.doStep()
            calculate_pref_vel(sim, pref_speed, goals, goal_threshold)
            locations = []
            current_velocities = []
            pref_velocities = []
            #pref_velocities_normalized = []
            pref_speeds = []
            dist_from_goal = []
            for agent_no in list(range(0,num_agents)):
                locations+= list(sim.getAgentPosition(agent_no))
                current_velocities += list(x for x in sim.getAgentVelocity(agent_no))
                pref_velocities += list(x for x in sim.getAgentPrefVelocity(agent_no))
                #pref_velocities_normalized += list(normalize_vector(sim.getAgentPrefVelocity(agent_no)))
            
            # make lists for cleaner For loops
            list_of_data_list = [locations, current_velocities, pref_velocities]
            list_of_df = [this_sample_locations, this_sample_current_velocities, this_sample_pref_velocities]
            
            # add masked_agents to the end of each list
            '''
            for data_list in list_of_data_list:
                data_list = add_masked_agents(data_list, masked_agents)
            '''

            # update the dataframes
            for idx in range(len(list_of_df)):  
                list_of_df[idx] = update_df(list_of_df[idx], step, list_of_data_list[idx])

            step += 1
            
            for agent_no in list(range(0,num_agents)):
                #print(str(agent_no) + ' preVel: (%.3f, %.3f), currVel: (%.3f, %.3f)' %( sim.getAgentPrefVelocity(agent_no)[0], sim.getAgentPrefVelocity(agent_no)[1], sim.getAgentVelocity(agent_no)[0], sim.getAgentVelocity(agent_no)[1]))
                agent_pref_velocities[agent_no] += [sim.getAgentPrefVelocity(agent_no)]
                agent_curr_velocities[agent_no] += [sim.getAgentVelocity(agent_no)]
        
        
        print("Simulation done! Number of steps:" , step)
        # plot sample
        if plot_samples:#i==0 or (i+1)%100==0:
            plot_this_sample(i, sim, velocity_scale, this_sample_locations, this_sample_current_velocities, this_sample_pref_velocities, sample_type = 'circle', circle_radius = circle_radius)
        
        samples.append([this_sample_locations, this_sample_current_velocities, this_sample_pref_velocities, this_sample_pref_speeds, this_sample_dist_from_goal, this_sample_goals])
    
    print("done! Checking file integrity...")
    
    # save generated samples to pkl file
    
    with open(save_file_name + '.pkl', 'wb') as handle:
        pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(save_file_name + '.pkl', 'rb') as handle:
        b = pickle.load(handle)
    
    # check that saved file matches original Python object
    for sample_idx in range(len(samples)):
        for df_idx in range(len(samples[sample_idx])):
            if not all(samples[sample_idx][df_idx].all() == b[sample_idx][df_idx].all()):
                print("MISMATCH IN SAMPLE ", sample_idx, "DF", df_idx)
    
    print('done! Duration: {}'.format(datetime.now() - start_time), "for", num_samples, "samples")
    return

def file_to_df(load_file, save_file_name):
    start_time = datetime.now()
    # make a list to save the input/output samples (sorted by agent)
    agent_list = []
    
    # open the pkl file
    with open(load_file, 'rb') as handle:
        OG_sample_list = pickle.load(handle)
    
    OG_sample = OG_sample_list[0]
    
    for ped in range(4):
        input_col_list = ['pedestrian_ID', 'curr_loc_x', 'curr_loc_y', 'curr_vel_x', 'curr_vel_y', 'goal_loc_x', 'goal_loc_y']
        output_col_list = ['pedestrian_ID','pred_vel_x', 'pred_vel_y']
        input_df = pd.DataFrame(columns = input_col_list)
        output_df = pd.DataFrame(columns = output_col_list)
        for idx in range(len(OG_sample[0])-1):
            '''
            The sample_list should be a list where len(sample_list) = number of generated simulations.
            
            We want to re-organize the data so that it is by agent, rather than by timestep.
            
            Each element of sample_list is an agent_list for that sample. In each agent_list there are 2 elements:
                    0. input dataframe, the columns are the (x, y) values for
                        - current locations, 
                        - current velocities, and 
                        - goal locations 
                        of all the agents in a scene.
                    1. output dataframe, the columns are the (x,y) values for the 
                        next-timestep velocity of the first agent listed in input 
                        dataframe.
                        - (Since it is only 2 values, we could make this into
                        an array or list. But I kept it as a dataframe to keep the 
                        consistency between the generated and real life data)
            '''
            
            x_frame = idx
            y_frame = x_frame+1
            
            # get agent current location, current velocity, and goal location of x_frame
            x_frame_curr_loc_x = OG_sample[0][str(ped) +'_x'].iloc[x_frame]
            x_frame_curr_loc_y = OG_sample[0][str(ped) +'_y'].iloc[x_frame]
            x_frame_curr_vel_x = OG_sample[1][str(ped) +'_x'].iloc[x_frame]
            x_frame_curr_vel_y = OG_sample[1][str(ped) +'_y'].iloc[x_frame]
            x_frame_goal_loc_x = OG_sample[5][str(ped) +'_x'].iloc[0]
            x_frame_goal_loc_y = OG_sample[5][str(ped) +'_y'].iloc[0]
            
            # get agent current velocity of y_frame
            y_frame_curr_vel_x = OG_sample[1][str(ped) +'_x'].iloc[y_frame]
            y_frame_curr_vel_y = OG_sample[1][str(ped) +'_y'].iloc[y_frame]
            
            # concat to input and output df
            input_temp = [int(ped), x_frame_curr_loc_x, x_frame_curr_loc_y, x_frame_curr_vel_x, x_frame_curr_vel_y, x_frame_goal_loc_x, x_frame_goal_loc_y]
            input_df.loc[len(input_df)] = input_temp
            output_temp = [ped, x_frame_curr_vel_x, x_frame_curr_vel_y]
            output_df.loc[len(output_df)] = output_temp
            
        # Once we've gone through all frames for this agent, save the input and output df as element in 
        agent_list.append([input_df, output_df])
    
    # save generated samples to pkl file
    with open(save_file_name + '.pkl', 'wb') as handle:
        pickle.dump(agent_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
     
    print('done! Duration: {}'.format(datetime.now() - start_time), "for", len(agent_list), "agents")
    return agent_list
    

if __name__ == '__main__':
    #file_to_df("RVO2_data/4_agents.pkl", "RVO2_data/4_agents_by_agent")
    
    
    # for generating train/validation/test samples and rollout situations (circle/lineup)
    make_videos = True
    sample_type =  '3_agents'
    
    if sample_type == None:
        # code for generating n samples of data for training the model. (note that these samples are randomized, and not supposed to be any specific situation)
        
        # set number of data samples you want to generate
        num_samples = 100000
    
        # set number of steps you want in your simluation (this value includes the initialized step)
        num_steps = 2
        agent_radius = 0.2
    
        # set min/max number of agents in a simulation
        neighbors_range = [3, 10]
        # set min/max speed values for all agents
        speed_range = [0.0, 2.5]
        # set min/max direction values for all agents
        direction_range = [-math.pi, math.pi]
    
        # set sigma noise values for preferred velocity
        speed_noise_range = 0.5
        direction_noise_range = math.pi/6
    
        # type of noise distribution
        noise_type = 'gaussian'
        
        goal_dist_range = [0, 10]
        
        pref_speed = 2.0
    
        # set value for minimum distance between agents' starting positions from each other
        initial_space_padding = 0.1
    
        velocity_scale = 0.3
        
        data_split = "train"
    
        plot_samples = False
        
        repeat_for = 1
        
        for i in range(repeat_for):
            save_file_name = "GOAL_REACHED_ORCA1_" + data_split + "_" + str(num_samples) + "_" + str(i)
        
            generate_n_samples(num_samples, num_steps, neighbors_range, speed_range, direction_range, speed_noise_range, direction_noise_range, noise_type, goal_dist_range, pref_speed, initial_space_padding, velocity_scale, save_file_name, plot_samples)
    
    elif sample_type == 'baseline':
        # code for generating n samples of data for training the model. (note that these samples are randomized, and not supposed to be any specific situation)
        
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
        direction_noise_range = math.pi/6
    
        # type of noise distribution
        noise_type = 'gaussian'
        
        goal_dist_range = [0, 10]
        
        pref_speed = 1.3
    
        # set value for minimum distance between agents' starting positions from each other
        initial_space_padding = 0.1
    
        velocity_scale = 0.3
        
        data_split = "train"
    
        plot_samples = False
        
        repeat_for = 7
        
        for i in range(repeat_for):
            save_file_name = "ORCA_baseline_" + data_split + "_" + str(num_samples) + "_" + str(i)
        
            generate_n_baseline_samples(num_samples, num_steps, neighbors_range, speed_range, direction_range, speed_noise_range, direction_noise_range, noise_type, goal_dist_range, pref_speed, initial_space_padding, velocity_scale, save_file_name, plot_samples)
    
    elif sample_type== 'circle':
        # code for generating n samples of circle situations, where some number of agents are placed in a circle. Their goal location is the opposite side of the circle.
        
        # set number of data samples you want to generate
        num_samples = 1
    
        # set min/max number of agents in a simulation
        agents_range = [3, 10]
        # set preferred speed values for all agents
        pref_speed = 2.0 #1.0
    
        # set value for minimum distance between agents' starting positions from each other
        initial_space_padding = 0.2
    
        velocity_scale = 0.3
        
        max_steps = 300
    
        plot_samples = True
        
        agent_radius = 0.5
        goal_threshold = agent_radius
        
        save_file_name = 'RVO2_data/rollout/finetune/ORCA2/' + "circle_" + str("DUMMY_04") #str(num_samples)
        
        generate_n_circle_samples(num_samples, agents_range, pref_speed, initial_space_padding, velocity_scale, max_steps, save_file_name, plot_samples, goal_threshold, agent_radius)
        
    elif sample_type == 'lineup':
        # code for generating n samples of lineup situations, where the agents are put into 2 lines some distance away from each other. Each agent's goal location is on the other line. Agents will need to pass each other to get to their goal positions
        
        # set number of data samples you want to generate
        num_samples = 1
    
        # set min/max number of agents in a simulation
        agents_range = [3, 10]
        # set preferred speed values for all agents
        pref_speed =  2.0 #1.0
    
        # set value for minimum distance between agents' starting positions from each other
        initial_space_padding = 0.1
    
        velocity_scale = 0.3
        
        max_steps = 300
    
        save_file_name = 'RVO2_data/rollout/finetune/ORCA1/' + "lineup_" + str(7)
    
        plot_samples = True
        
        agent_radius = 0.2
        goal_threshold = agent_radius
        
        generate_n_lineup_samples(num_samples, agents_range, pref_speed, initial_space_padding, velocity_scale, max_steps, save_file_name, plot_samples, goal_threshold, agent_radius)
    
    elif sample_type == 'from_file':
        # code for generating samples from an already existing file. This is if you want to run the exact sme scenario but with different ORCA parameters
    
        # set min/max number of agents in a simulation
        agents_range = [3, 10]
        # set preferred speed values for all agents
        pref_speed =  2.0 #1.0
    
        # set value for minimum distance between agents' starting positions from each other
        initial_space_padding = 0.1
    
        velocity_scale = 0.3
        
        max_steps = 300
    
        plot_samples = True
        
        agent_radius = 0.2
        goal_threshold = agent_radius
        
        data_folder = 'RVO2_data/rollout/finetune/ORCA2/'
        file_list = os.listdir(data_folder)
        count = 0
        for file in tqdm(file_list):
            if file.endswith(".pkl"):
                save_file_name = 'RVO2_data/rollout/finetune/ORCA2/check_1'#'RVO2_data/rollout/finetune/ORCA1/' + "lineup_" + str(count)
                count += 1
                generate_sample_from_file(data_folder + file, agents_range, pref_speed, initial_space_padding, velocity_scale, max_steps, save_file_name, plot_samples, goal_threshold, agent_radius)
    elif sample_type== '3_agents':
        # code for generating n samples of circle situations, where some number of agents are placed in a circle. Their goal location is the opposite side of the circle.
        
        # set number of data samples you want to generate
        num_samples = 1
    
        # set min/max number of agents in a simulation
        agents_range = [3, 10]
        # set preferred speed values for all agents
        pref_speed = 2.0 #1.0
    
        # set value for minimum distance between agents' starting positions from each other
        initial_space_padding = 0.2
    
        velocity_scale = 0.3
        
        max_steps = 100
    
        plot_samples = False
        
        agent_radius = 0.5
        goal_threshold = agent_radius
        
        save_file_name = 'RVO2_data/temp/' + "temp"
        
        generate_n_agents_samples(num_samples, agents_range, pref_speed, initial_space_padding, velocity_scale, max_steps, save_file_name, plot_samples, goal_threshold, agent_radius)
    elif sample_type== 'traj_clustering':
        # code for generating n samples of circle situations, where some number of agents are placed in a circle. Their goal location is the opposite side of the circle.
        
        # set number of data samples you want to generate
        num_samples = 1
    
        # set min/max number of agents in a simulation
        agents_range = [3, 10]
        # set preferred speed values for all agents
        pref_speed = 2.0 #1.0
    
        # set value for minimum distance between agents' starting positions from each other
        initial_space_padding = 0.2
    
        velocity_scale = 0.3
        
        max_steps = 100
    
        plot_samples = True
        
        agent_radius = 0.2
        goal_threshold = agent_radius
        
        save_file_name = 'RVO2_data/' + "4_agents"
        
        generate_traj_clustering_circle_samples(num_samples, agents_range, pref_speed, initial_space_padding, velocity_scale, max_steps, save_file_name, plot_samples, goal_threshold, agent_radius)
    else:
        print('invalid sample type. check input')
    
    if make_videos == True and sample_type != None:
        # for making videos of situation .pkl files from a specific directory
        
        data_folder = 'RVO2_data/temp/'#'RVO2_data/'#'RVO2_data/rollout/finetune/ORCA1/'
        save_folder = data_folder 
        file_list = os.listdir(data_folder)
        for file in tqdm(file_list):
            
            # check if file is .pkl
            if file.endswith(".pkl"):        
                
                # open .pkl file as sample_list
                '''
                The sample_list should be a list where len(sample_list) = number of generated simulations.
                Each element of sample_list should be another list of 4 elements:
                    - current location dataframe
                    - current velocity dataframe
                    - preferred velocity dataframe
                    - preferred velocity dataframe (normalized)
                '''
                file_idx = os.path.splitext(file)[0]
                load_file = data_folder + "/" + file
                with open(load_file, 'rb') as handle:
                    sample_list = pickle.load(handle)
                
                # for each sample in sample_list, take what you need for x(input) and y (output)
                '''
                For this video we only need the locations
                '''
                for sample_idx in range(len(sample_list)):
                    this_sample = sample_list[sample_idx]
                    
                    curr_loc_df = this_sample[0]
                    agent_radius = 0.2
                    #make_video(curr_loc_df, None, sample_idx, None, None, agent_radius, save_folder, file_idx + '_' + str(sample_idx))
                    make_video(curr_loc_df, None, sample_idx, None, None, agent_radius, save_folder, file_idx + '_' + str(sample_idx))
        
        
        
        
        
        
        
        
