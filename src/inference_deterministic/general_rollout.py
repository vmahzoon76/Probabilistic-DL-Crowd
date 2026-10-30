import torch
import numpy as np
import pandas as pd
import matplotlib as mpl
from shapely.geometry import Point, Polygon
from cnnmodel import CNN
import cv2
import matplotlib.pyplot as plt
from Transformer import TransformerModel
from lidar_obs import lidar
import pickle
import math
import rvo2

device = 'cuda:0'

def normalize_vector(vector):
    norm_vec = vector/np.sqrt(sum(np.array(vector)**2))
    if np.isnan(norm_vec).any():
        norm_vec[np.isnan(norm_vec)] = 0.0
    return norm_vec

def calculate_pref_vel(sim, pref_speed, goal_points, goal_threshold):
    for neighbor in range(0,sim.getNumAgents()): 
        goal_x = goal_points[neighbor*2]
        goal_y = goal_points[neighbor*2+1]
        if math.dist(list(sim.getAgentPosition(neighbor)), [goal_x, goal_y]) < goal_threshold:
            sim.setAgentPrefVelocity(neighbor, (0.0, 0.0))
        else:
            sim.setAgentPrefVelocity(neighbor, tuple(pref_speed*normalize_vector([goal_i - curr_i for goal_i, curr_i in zip([goal_x, goal_y], list(sim.getAgentPosition(neighbor)))])))

def video_generater(df, arr, name, dataset):
    if dataset == "ETH":
        limits_x = [-10, 17]
        limits_y = [-10, 17]

        obs_orig = [[[-3, -3], [-1, 0], [5, 2], [13.5, 2.5], [13.499, -2.99]],
                    [[-3, 12.75], [13.5, 12.7499], [13.499, 9.5], [2, 9.499]]]

    elif dataset == "ETH_rotate":
        limits_x = [-10, 17]
        limits_y = [-17, 10]

        obs_orig = [[[-3, 3], [0, 1], [2, -5], [2.5, -13.5], [-2.99, -13.499]],         [[12.75, 3], [12.7499, -13.5], [9.5, -13.499], [9.499, -2]]] 
    
    elif dataset == "Zara1":
        limits_y = [2, 23]
        limits_x = [-10, 11]

        obs_orig = [[[-0.7989999999999999, 17.198999999999998], [-6.3, 17.2], [-6.2989999999999995, 4.7], [-3.3, 4.699], [-3.299, 6.7], [-1.8, 6.699], [-1.7990000000000002, 14.7], [-0.8, 14.699]],[[7.250, 12.000], [5.500, 11.750], [5.499, 6.500], [7.249, 6.250]]]


    elif dataset == "Zara2":
        limits_y = [2, 23]
        limits_x = [-10, 11]

        obs_orig = [[[-0.7989999999999999, 17.198999999999998], [-6.3, 17.2], [-6.2989999999999995, 4.7], [-3.3, 4.699], [-3.299, 6.7], [-1.8, 6.699], [-1.7990000000000002, 14.7], [-0.8, 14.699]]]



    elif dataset=="Hotel":
        limits_y = [-13, 7]
        limits_x = [-13, 7]

        obs_orig = [[[-0.827, -5.126], [-0.89199, -5.0134], [-1.022, -5.01343], [-1.087, -5.126], [-1.022, -5.23858],
                        [-0.89199, -5.23859]],
                       [[-0.689, -1.76], [-0.754, -1.6474], [-0.88399, -1.6475], [-0.949, -1.76], [-0.88399, -1.87258],
                        [-0.754, -1.872583]],
                       [[-0.727, 1.917], [-0.792, 2.02958], [-0.92199, 2.02957], [-0.987, 1.917], [-0.92199, 1.8044],
                        [-0.792, 1.80441]], [[-0.517, -10.065], [-0.613, -7.755], [-1.199, -7.737], [-1.2, -10.015]]]
    # df is the original dataset and arr is the output of roll-out

    # determing the min and max for the red boundary (rectangle)
    x_range = [df.loc[df["pos_x"].idxmin()]["pos_x"] - 1, df.loc[df["pos_x"].idxmax()]["pos_x"] + 1]
    y_range = [df.loc[df["pos_y"].idxmin()]["pos_y"] - 1, df.loc[df["pos_y"].idxmax()]["pos_y"] + 1]

    width = 1000
    height = 720
    dt = 0.2

    simulation_idx = 0

    tmp_name = 'court.png'

    # initiate video
    movie_name = name + '.avi'
    video = cv2.VideoWriter(movie_name, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                            fps=int(1. / dt), frameSize=(width, height))
    fig = plt.figure(figsize=(12., 8.5))
    ax = fig.add_subplot()

    polygon1 = obs_orig[0]

    polygon1.append(polygon1[0])  # repeat the first point to create a 'closed loop'

    xs1, ys1 = zip(*polygon1)  # create lists of x and y values
    if dataset!="Zara2":
        polygon2 = obs_orig[1]
        polygon2.append(polygon2[0])
        xs2, ys2 = zip(*polygon2)

    if dataset=="Hotel":
        polygon3 = obs_orig[2]
        polygon4 = obs_orig[3]
        polygon3.append(polygon3[0])  # repeat the first point to create a 'closed loop'
        polygon4.append(polygon4[0])
        xs3, ys3 = zip(*polygon3)  # create lists of x and y values
        xs4, ys4 = zip(*polygon4)
    # going over each frame in the roll-out output
    for cells in arr:
        # creating red boundary
        rect = mpl.patches.Rectangle((x_range[0], y_range[0]), x_range[1] - x_range[0], y_range[1] - y_range[0],
                                     edgecolor='red',
                                     facecolor='none', linewidth=2)
        ax.add_patch(rect)

        ax.plot(xs1, ys1, color="black")
        if dataset!="Zara2":
            ax.plot(xs2, ys2, color="black")
        if dataset=="Hotel":
            ax.plot(xs3, ys3, color="black")
            ax.plot(xs4, ys4, color="black")
        # finding the pedestrain
        peds = list(cells.keys())

        # going over each pedestrian and scattering groundtruth
        for ped in peds:
            df_new = df[df["pedestrian_ID"] == ped]
            plt.scatter(df_new['pos_x'], df_new['pos_y'], marker='.', label='prediction', color='r',
                        alpha=0.4)

        # creating an array for objects of circles ( circle which represents each pedestrian)
        scatter = []

        # now processing each frame
        for i in range(len(cells)):
            # adding circle of each pedestrian
            scatter.append(mpl.patches.Circle((0, 0), radius=0.2, color='g'))
        i = 0
        for ped in cells:
            x = []
            y = []
            # determining the center of the circle
            x.append(cells[ped][0])
            y.append(cells[ped][1])
            scatter[i].center = [x, y]
            ax.add_patch(scatter[i])
            # determining the number of pedestrian
            ax.annotate(int(ped), (cells[ped][0] + 0.15, cells[ped][1] + 0.15))
            i = i + 1
        plt.ylim(limits_y[0], limits_y[1])
        plt.xlim(limits_x[0], limits_x[1])
        plt.gca().set_aspect('equal')
        plt.title('Test Simulation: ' + str(simulation_idx))
        # plt.legend()

        # plt.show()
        # plt.pause(dt)
        plt.savefig(tmp_name, dpi=90)
        # plt.savefig(save_folder + '/' + save_name + str(time_step)  + '.png', dpi=90)

        frame = cv2.imread(tmp_name)
        # scale_percent = 90
        # width2 = int(frame.shape[1] * scale_percent / 100)
        # height2 = int(frame.shape[0] * scale_percent / 100)
        frame = cv2.resize(frame, (width, height))
        video.write(frame)
        ax.clear()
    video.release()
    del video


def process_data(data, max_ped):
    curr_loc_df = data[["curr_loc_x", "curr_loc_y"]]
    curr_vel_df = data[["curr_vel_x", "curr_vel_y"]]
    pref_vel_df = data[["goal_loc_x", "goal_loc_y"]]

    pref_vel_df = pref_vel_df - np.array(curr_loc_df.iloc[0])
    curr_loc_df = curr_loc_df - curr_loc_df.iloc[0]

    # reshape the arrays into (#sample, #agents, #components)
    (s, c) = curr_loc_df.shape
    x_curr_loc = np.concatenate(
        (np.array(curr_loc_df).reshape(1, s, c), np.ones((1, max_ped - s, 2)) * 999), axis=1)
    x_curr_vel = np.concatenate(
        (np.array(curr_vel_df).reshape(1, s, c), np.ones((1, max_ped - s, 2)) * 999), axis=1)
    x_pref_vel = np.concatenate(
        (np.array(pref_vel_df).reshape(1, s, c), np.ones((1, max_ped - s, 2)) * 999), axis=1)

    x_combined = np.concatenate((x_curr_loc, x_curr_vel, x_pref_vel), axis=-1)

    return x_combined


def process_data_lidar(data, obs_orig):
    curr_loc_df = data[["curr_loc_x", "curr_loc_y"]]
    curr_vel_df = data[["curr_vel_x", "curr_vel_y"]]
    pref_vel_df = data[["goal_loc_x", "goal_loc_y"]]
    pos = np.array(curr_loc_df.iloc[0])
    pref_vel_df = pref_vel_df - np.array(curr_loc_df.iloc[0])
    curr_loc_df = curr_loc_df - curr_loc_df.iloc[0]

    #
    # pref_vel_df = pref_vel_df - np.array(curr_loc_df.iloc[0])
    # curr_loc_df = curr_loc_df - curr_loc_df.iloc[0]

    curr = np.array(curr_loc_df).tolist()
    curr_lidar = curr[1:]
    prev = np.array(curr_loc_df) - 0.4 * np.array(curr_vel_df) + 0.4 * np.array(curr_vel_df)[0]
    prev_lidar = prev.tolist()[1:]

    goal = np.array(pref_vel_df)[0].tolist()
    curr_vel = np.array(curr_vel_df)[0].tolist()
    # obs_orig = [[[-3, -3], [-1, 0], [5, 2], [13.5, 2.5], [13.499, -2.99]],
    #             [[-3, 12.75], [13.5, 12.7499], [13.499, 9.5], [2, 9.499]]]
    if len(obs_orig)==1:
        obs = [(np.array(obs_orig[0]) - pos).tolist()]
    elif len(obs_orig)==2:
        obs = [(np.array(obs_orig[0]) - pos).tolist()] + [(np.array(obs_orig[1]) - pos).tolist()]
    else:
        obs = [(np.array(obs_orig[0]) - pos).tolist()] + [(np.array(obs_orig[1]) - pos).tolist()] + [(np.array(obs_orig[2]) - pos).tolist()] + [(np.array(obs_orig[3]) - pos).tolist()]
    c = lidar(10, 0.5, obs, curr_lidar)
    c.sense_obstacles()
    # c.plot()
    curr_lidar = c.scan_array
    pos = pos - 0.4 * np.array(curr_vel_df.iloc[0])
    if len(obs_orig) == 1:
        obs = [(np.array(obs_orig[0]) - pos).tolist()]
    elif len(obs_orig) == 2:
        obs = [(np.array(obs_orig[0]) - pos).tolist()] + [(np.array(obs_orig[1]) - pos).tolist()]
    else:
        obs = [(np.array(obs_orig[0]) - pos).tolist()] + [(np.array(obs_orig[1]) - pos).tolist()] + [
            (np.array(obs_orig[2]) - pos).tolist()] + [(np.array(obs_orig[3]) - pos).tolist()]
    c = lidar(10, 0.5, obs, prev_lidar)
    c.sense_obstacles()
    prev_lidar = c.scan_array
    l = prev_lidar + curr_lidar + curr_vel + goal

    z = [l[0:720], l[720:1440]]
    h = [l[1440:1444]]

    return z, h


# model_type= "transformer", "lidar", "CVM"!
# environment= arrays of polygons without the final connection.
# dataset_name="ETH" or "Zara1"
# type random, finetuning, pretrained, None

def Rollout(df, goal_df, model_type, enviornment, dataset_name, exclude_agents, start_time, end_time, training_type="None", model_trained_on_data_part = 1):
    model_name = training_type + '_' + dataset_name.lower() + '_' + str(model_trained_on_data_part) + ".pth"
    # choose the model we are rolling out with
    if model_type == "transformer":
        model = TransformerModel(inputs=6, pred_type="trajectory", out_dim=2, n_objects=27, nhead=8, nhid=128,
                                 nlayers=4).to(device)
        if training_type in ["random", "finetuning"]:
            model.load_state_dict(torch.load(
                "Weights/" + dataset_name + "/" + model_type + "/" + training_type + "/" + model_name))
            print("loaded", model_name)
        else:
            model = (
                torch.load("Weights/" + training_type + "/" + model_type + "/" + "trajctory_predictor_pedApril23,2024,1630.pth"))
            print("loaded", "trajctory_predictor_pedApril23,2024,1630.pth")

    elif model_type == "lidar":
        model = CNN().to(device)
        if training_type in ["random", "finetuning"]:
            model.load_state_dict(torch.load(
                "Weights/" + dataset_name + "/" + model_type + "/" + training_type + "/" + model_name))
            print("loaded", model_name)
        else:
            print("hi")
            model = (torch.load("Weights/" + training_type + "/" + model_type + "/" + "CNN_sfm_april26.pth"))
            print("loaded", "CNN_sfm_april26.pth")

    max_ped = 27
    # take vertice lise obs_orig and make a list of Polygons obs_check. Initialize obs_agents, which keeps track of agents that go through obstacles
    obs_orig = [list(item) for item in enviornment]
    obs_check = [list(item) for item in obs_orig]
    for item in obs_check:
        item.append(item[0])
    obs_check = [Polygon(item) for item in obs_check]
    obs_agents = []

    dt = 0.4

    vel_lin_dict = {}

    final_peds_locations = []
    df_new = df[df["frame_number"] == start_time]
    peds = list(df_new["pedestrian_ID"])
    dict_ped = {}
    for ped in peds:
        dict_ped[ped] = np.array(df_new[df_new["pedestrian_ID"] == ped][["pos_x", "pos_y", "v_x", "v_y"]])[0]
        if model_type == "CVM":
            v = np.array(goal_df[goal_df["pedestrian_ID"] == ped][["g_x", "g_y"]])[0] - dict_ped[ped][0:2]
            if np.linalg.norm(v)==0:
                vel_lin_dict[ped]=v
            else:
                vel_lin_dict[ped] = np.linalg.norm(dict_ped[ped][2:4]) * v / np.linalg.norm(v)
    final_peds_locations.append(dict_ped)
    count = 0
    for time in range(start_time + 1, end_time + 1):
        print(time)
        prev_dict = final_peds_locations[count]
        dict_ped = {}
        df_new = df[df["frame_number"] == time]
        peds = list(df_new["pedestrian_ID"])
        for ped in peds:
            if ped in exclude_agents: # check if agent is excluded. If yes then we do not do rollout on this pedestrian
                dict_ped[ped] = np.array(df_new[df_new["pedestrian_ID"] == ped][["pos_x", "pos_y", "v_x", "v_y"]])[0]
                continue
            elif ped not in list(prev_dict.keys()): # check if this is a pedestrian's first timestep. If yes, then we get their first timestep from the ground truth data
                print("hi")
                dict_ped[ped] = np.array(df_new[df_new["pedestrian_ID"] == ped][["pos_x", "pos_y", "v_x", "v_y"]])[0]
                if model_type == "CVM":
                    v = np.array(goal_df[goal_df["pedestrian_ID"] == ped][["g_x", "g_y"]])[0] - dict_ped[ped][0:2]
                    if np.linalg.norm(v) == 0:
                        vel_lin_dict[ped] = v
                    else:
                        vel_lin_dict[ped] = np.linalg.norm(dict_ped[ped][2:4]) * v / np.linalg.norm(v)
                continue
            elif ped in obs_agents: # check if pedestrian has gone through an obstacle. If yes then they do not move (so we set their velocity to (0,0))
                outs_curr_vel = np.array([0, 0])
            elif model_type == "CVM": # if we are using the CVM, then just reuse the current velocity
                outs_curr_vel = vel_lin_dict[ped]
            else: # if we are using a NN, then prepare the model inputs for the center agent
                input_col_list = ['pedestrian_ID', 'curr_loc_x', 'curr_loc_y', 'curr_vel_x', 'curr_vel_y', 'goal_loc_x',
                                  'goal_loc_y']
                input_df = pd.DataFrame(columns=input_col_list)
                arr_goal = np.array(goal_df[goal_df["pedestrian_ID"] == ped][["g_x", "g_y"]])[0]
                arr_posvel = prev_dict[ped]
                concatenated_array = np.concatenate((np.array([ped]), arr_posvel, arr_goal))
                new_row = pd.DataFrame([concatenated_array], columns=input_df.columns)
                input_df = pd.concat([input_df, new_row], ignore_index=True)

                for i in list(prev_dict.keys()): # prepare the model inputs for the neighboring agents
                    if i != ped:
                        arr_goal = np.array(goal_df[goal_df["pedestrian_ID"] == i][["g_x", "g_y"]])[0]
                        arr_posvel = prev_dict[i]
                        concatenated_array = np.concatenate((np.array([i]), arr_posvel, arr_goal))
                        new_row = pd.DataFrame([concatenated_array], columns=input_df.columns)
                        input_df = pd.concat([input_df, new_row], ignore_index=True)

                if model_type == "transformer": # if we are using a transformer based model
                    X = process_data(input_df, max_ped)
                    X = np.array(X, dtype=np.float64)
                    outs_curr_vel = model(torch.from_numpy(X).to(device).float()).cpu().detach().numpy()[
                        0, 0]
                elif model_type == "lidar": # if we are using a CNN (lidar) based model
                    x, z = process_data_lidar(input_df, obs_orig)
                    outs_curr_vel = model(torch.from_numpy(np.array(x).reshape(1, 2, 720)).to(device).float(),
                                          torch.from_numpy(np.array(z).reshape(1, 4)).to(
                                              device).float()).cpu().detach().numpy()[0]
                elif model_type == "ORCA": # if we are using ORCA simulator as the model
                    # ORCA input is very similar to the transformer inputs
                    # First initialize the simulator object. Below are the names of the variables and their assigned values
                    '''
                    The following values are from slide 4 ORCA1 from this slidedeck (last edit was June 5, 2023)
              
                    Timestep: 0.4s (changed from 0.1 to 0.4 to match other models. Note we are doing 4 x 0.1s to get to 0.4s)
                    NeighborDist: 5m
                    MaxNeighbors: 30 agents
                    TimeHorizon: 5s
                    TimeHorizonObst: 1s
                    Radius: 0.2m
                    MaxSpeed: pref_speed + 0.5m/s
                    '''
                    pref_speed = 1.3
                    goal_threshold = 0.4 # I just set this to 2 x agent radius
                    goals = []
                    sim = rvo2.PyRVOSimulator(0.1, 5, 30, 5, 1, 0.2, pref_speed+0.5)
                    
                    # initialize the obstacles in the simulator (Vertices must be in counter-clockwise solid objects. Negative obstacles, like bounding polygon around an environment, must be in clockwise order) Comment out if you want ORCA to be blind to obstacles
                    '''
                    for obstacle in enviornment:
                        sim.addObstacle(obstacle)
                    sim.processObstacles() # Documentation is kind of unclear about what this does. But apparently we need it(?)
                    '''
                        
                    # go through the rows in the input_df. each row is an agent, so we can add agents to the sim one by one
                    for input_df_idx in range(len(input_df)):
                        # get the agent current position, current velocity, and preferred velocity
                        this_agent_curr_loc = (input_df.iloc[input_df_idx]['curr_loc_x'], input_df.iloc[input_df_idx]['curr_loc_y'])
                        this_agent_curr_vel = (input_df.iloc[input_df_idx]['curr_vel_x'], input_df.iloc[input_df_idx]['curr_vel_y'])
                        goals += [input_df.iloc[input_df_idx]['goal_loc_x'], input_df.iloc[input_df_idx]['goal_loc_y']]
                        
                        # NOTE: each agent added is assigned a number so the sim can keep track of it. First agent is assigned number 0. (I don't think I can set it manually to match the pedestrian_ID in input_df)
                        sim.addAgent(this_agent_curr_loc)
                        sim.setAgentVelocity(input_df_idx, this_agent_curr_vel)
                    
                    # After every agent has been added to the sim, use this function to get their first preferred velocities
                    calculate_pref_vel(sim, pref_speed, goals, goal_threshold)
                    
                    # run 1 step of the sim, then recalculate the pref velocities (since we are using 0.1s in the sim() but want 0.4s as the timestep, we do a FOR loop and keep the last step's information. This is to match the SFM baseline)
                    
                    # save the first location
                    agent_0_prev_loc = np.asarray(sim.getAgentPosition(0))
                    for step in range(int(dt/0.1)):
                        sim.doStep()
                        calculate_pref_vel(sim, pref_speed, goals, goal_threshold)
                    
                    # the outputs we want are the velocity and location of the agent 0. Just get them directly from the sim
                    #outs_curr_vel = np.asarray(sim.getAgentVelocity(0))
                    outs_loc = np.asarray(sim.getAgentPosition(0))
                    outs_curr_vel = (outs_loc - agent_0_prev_loc)/dt
                    

            if model_type != "ORCA" or ped in obs_agents: # if we are using ORCA, then we already got them directly from the sim
                outs_loc = prev_dict[ped][0:2] + dt * outs_curr_vel # use the predicted velocity to calculate the next location
                
            dict_ped[ped] = np.concatenate((outs_loc, outs_curr_vel))

            point = Point(list(outs_loc)) # check if the center agent entered an obstacle after this prediction. If yes, then add it to the obs_agents list so we can freeze it for the remaining timesteps
            for obs in obs_check:
                if point.within(obs):
                    obs_agents.append(ped)

        print(dict_ped)
        final_peds_locations.append(dict_ped)
        count = count + 1
    print(f"number of obstacle: {len(set(obs_agents))}")
    return final_peds_locations, list(set(obs_agents)) # return the locations of agents after the entire rollout is completed


def _calculate_SADE_SFDE(rollout_list, ped_table, gt_list, first_timestep, last_timestep, nonlinear_agents, exclude_agents):
    print(first_timestep)

    # splice gt_list to match rollout_list indexing
    gt_list = gt_list[first_timestep: last_timestep + 1]  # the +1 is to include the last timestep

    # get the first ped ID and the last ped ID
    min_ped_ID = min(rollout_list[0].keys())
    max_ped_ID = max(rollout_list[-1].keys())

    ped_list = list(ped_table['pedestrian_ID'])

    min_ped_ID_idx = ped_list.index(min_ped_ID)
    max_ped_ID_idx = ped_list.index(max_ped_ID)

    ped_list = ped_list[min_ped_ID_idx:max_ped_ID_idx + 1]  # the +1 is to account for python indexing

    # remove exclude_peds from ped_list
    ped_list = list(set(ped_list) - set(exclude_agents))
    # remove peds that are in the nonlinear list but not in the updated ped list
    nonlinear_agents = list(set(nonlinear_agents) - (set(nonlinear_agents) - set(ped_list)))

    ADE_dict = {}
    NL_ADE_dict = {}
    FDE_dict = {}
    MDE_dict = {}
    skipped_peds = 0
    NL_skipped_peds = 0
    # loop through ped table
    for ped in ped_list:

        # get first and last timesteps
        first = ped_table[ped_table['pedestrian_ID'] == ped]['First_Frame'].values[0]
        last = ped_table[ped_table['pedestrian_ID'] == ped]['Last_Frame'].values[0]

        # first_timestep and last_timestep are the first/last timesteps of trai/validation/test disjoint split. We need to boud the variables first and last within these values
        # if first is smaller than first_timestep, set first to first_timestep
        if first < first_timestep:
            first = first_timestep
        # if last is larger than last_timestep, set last to last_timestep
        if last > last_timestep:
            last = last_timestep

        # if first == last, then this pedestrian only appears in 1 timestep. Since first timestep is gt, we don't calculate ADE or FDE on
        if first != last and len(list(range(first - first_timestep + 1,
                                            last - first_timestep + 1))) > 0:

            # add ped to the flag dict
            ADE_dict[ped] = 0
            FDE_dict[ped] = 0
            MDE_dict[ped] = 0

            displacement_sum = 0
            min_dist = 999
            # print("ped", ped)
            count = 0
            # get all locations of this ped
            for idx in range(first - first_timestep + 1,
                             last - first_timestep + 1):  # include the last timesteps's idx in the loop, and we skip the first timestep because that is gt
                # DELETE LATER (just check that the first idx is the the same for gt and rollout)
                '''
                rollout_loc = [rollout_list[idx-1][ped][0], rollout_list[idx-1][ped][1]]
                gt_loc = [gt_list[idx-1][ped][0], gt_list[idx-1][ped][1]]
                test = math.dist([rollout_loc[0], rollout_loc[1]], [gt_loc[0], gt_loc[1]])
                if count == 0 and test !=0:
                    print('wtf')
                count +=1
                '''

                # get the loc_x, loc_y for this timestep and the next for this ped
                rollout_loc = [rollout_list[idx][ped][0], rollout_list[idx][ped][1]]
                gt_loc = [gt_list[idx][ped][0], gt_list[idx][ped][1]]

                # calculate the distance between the rollout loc and the gt loc
                this_dist = math.dist([rollout_loc[0], rollout_loc[1]], [gt_loc[0], gt_loc[1]])
                displacement_sum += this_dist
                if min_dist > this_dist:
                    min_dist = this_dist
                # if ped == 357:
                #    print(math.dist([rollout_loc[0], rollout_loc[1]], [gt_loc[0], gt_loc[1]]))

                # if this is the last timestep fo rthis agent, update FDE
                if idx == last - first_timestep:
                    FDE_dict[ped] = this_dist

            ADE_dict[ped] = displacement_sum / (last - first)
            MDE_dict[ped] = min_dist

            if ped in nonlinear_agents:
                NL_ADE_dict[ped] = displacement_sum / (last - first)
        else:
            skipped_peds += 1
            if ped in nonlinear_agents:
                NL_skipped_peds += 1

    # calculate the average distance traveled across all ped
    if (len(ped_list) - skipped_peds) == 0:
        ADE = -999
        NL_ADE = -999
        FDE = -999
        MDE = -999
    elif (len(nonlinear_agents) - NL_skipped_peds) == 0:
        ADE = sum(list(ADE_dict.values())) / (len(ped_list) - skipped_peds)
        NL_ADE = -999
        FDE = sum(list(FDE_dict.values())) / (len(ped_list) - skipped_peds)
        MDE = sum(list(MDE_dict.values())) / (len(ped_list) - skipped_peds)
    else:
        ADE = sum(list(ADE_dict.values())) / (len(ped_list) - skipped_peds)
        NL_ADE = sum(list(NL_ADE_dict.values())) / (len(nonlinear_agents) - NL_skipped_peds)
        FDE = sum(list(FDE_dict.values())) / (len(ped_list) - skipped_peds)
        MDE = sum(list(MDE_dict.values())) / (len(ped_list) - skipped_peds)

    return ADE_dict, ADE, NL_ADE_dict, NL_ADE, FDE_dict, FDE, MDE_dict, MDE

def _calculate_collision_severity(rollout_list, ped_table, obs_agents, first_timestep, last_timestep,
                                  collision_threshold,
                                  timestep_threshold):
    collisions = []
    collision_time_steps_count = 0
    collision_severity = []
    found = False
    timestep_too_close = 0
    total_pairs_checked = 0

    # sum the number of timesteps for each agent in the roillout_list
    sum_timesteps_for_each_agent = 0

    # loop through every timestep
    # NOTE: if you want to measure collision severity for a specific subset of timesteps, please use list slicing on rollout_list input
    for idx in range(first_timestep - first_timestep + 1,
                     last_timestep - first_timestep + 1):  # the +1 is to include the last timesteps's idx in the loop
        # print(idx)
        this_timestep = rollout_list[idx]
        # get the list of pedestrians in this timestep
        this_ped_list = list(this_timestep.keys())
        # sort the list (just in case it isn't already sorted. otherwise the next nested for loops will not work)
        this_ped_list.sort()
        num_peds = len(this_ped_list)
        sum_timesteps_for_each_agent += num_peds

        # only check for collisions if there are multiple peds in the timestep (if there is only 1 ped, there cannot be any collisions)
        if len(this_ped_list) > 1:
            # loop through every pedestrian in the timestep
            for ped_1_idx in range(0, num_peds - 1):
                ped_1 = this_ped_list[ped_1_idx]
                x1 = this_timestep[ped_1][0]
                y1 = this_timestep[ped_1][1]
                for ped_2_idx in range(ped_1_idx + 1, num_peds):  # k -- agent index
                    ped_2 = this_ped_list[ped_2_idx]
                    x2 = this_timestep[ped_2][0]
                    y2 = this_timestep[ped_2][1]

                    # if this is ped_2's first timestep, we do not count the collision (this is like ped_2 is spawning out of nowhere, giving ped_1 no time to react)
                    # NOTE: this is assuming the this_ped_list has been sorted! So ped_1 should appear in the scene before ped_2!
                    '''
                    print('ped_1', ped_1,'ped_2', ped_2)
                    if ped_1 == 157 and ped_2 == 158:
                        print("hi")
                    '''

                    ped_2_first_timestep = ped_table[ped_table['pedestrian_ID'] == ped_2]['First_Frame'].values[0]

                    total_pairs_checked += 1
                    if (first_timestep + idx - ped_2_first_timestep) <= timestep_threshold:
                        timestep_too_close += 1

                    # we only check collisions if the current timestep is some threshold away from ped_2's first timestep. So we will check anything larger than that threshold
                    if (ped_1 not in obs_agents or ped_2 not in obs_agents) and (
                            first_timestep + idx - ped_2_first_timestep) > timestep_threshold:

                        # check if a collision occurred
                        if math.dist([x1, y1], [x2, y2]) <= collision_threshold:
                            x = (x1 + x2) * .5
                            y = (y1 + y2) * .5

                            collisions.append([x, y, ped_1, ped_2, idx])

                            overlap_value = collision_threshold - math.dist([x1, y1], [x2, y2])

                            # scale by severity of the overlap

                            # this is piecewise function metric
                            '''
                            if overlap_value > threshold/2: # large overlap
                                scale = 10.0
                            elif overlap_value <= threshold/2 and overlap_value > threshold/10: # medium overlap
                                scale = 5.0
                            else: # small overlap (this should be all overlap_value <= threshold/100. so the agents are barely touching each other)
                                scale = 1.0
                            '''

                            # continuous metric
                            scale = 2.5

                            collision_severity.append(scale * overlap_value)
                            if found == False:
                                found = True

        # if this timestep had at least 1 collision, increment the counter
        if found:
            collision_time_steps_count += 1
        # reset found_flag for the next timestep
        found = False

    '''
    After finding all the collisions, we will have 3 variables:
        -- collisions: this is a list of every collsiion. each element has
            * loc_x of collision
            * loc_y of the collision
            * ped_1 ID involved in the collision
            * ped_2 ID involved in the collision
            * timestep ID when collision happened (this should match the ID of the original rollout_list)
        -- collision_time_steps_count: total number of timesteps that had at least 1 collision happen
        -- collision_severity: list of collision severity values (should be same length as collisions list)
    '''

    # use _collisions_plot() to get the (x,y) locations of the collisions for plotting in videos
    # collisions_x, collisions_y = _collisions_plot(collisions)

    if len(collision_severity) > 0:
        sum_collision_severity = sum(collision_severity)
    else:
        sum_collision_severity = 0.0

    # print("total_pairs_checked", total_pairs_checked)
    # print("timestep_too_close", timestep_too_close)
    return collisions, collision_time_steps_count, collision_severity, sum_collision_severity, sum_timesteps_for_each_agent


def _collisions_plot(collisions):
    x = []
    y = []
    for i in collisions:
        x.append(i[0])
        y.append(i[1])

    return x, y


def calculate_metrics(model_type, training_type, model_trained_on_data_part, h, ped_table, gt_list, first_timestep,
                      last_timestep, window, nonlinear_agents, exclude_agents):
    metrics = {}
    # for loop through each h
    for h_idx in range(len(h)):
        this_h = h[h_idx]
        this_window = window[h_idx]
        this_ADE = []
        this_NL_ADE = []
        this_FDE = []
        this_MDE = []
        this_collision_time_steps_count = []
        this_sum_collision_severity = []
        this_collision_list = []
        total_collisions = []
        total_sum_timesteps_for_each_agent = []

        print('calculating metrics for h=', this_h, "with window=", this_window)
        # plot_print = 'calculating metrics for \nh= ' + str(this_h) + " with window= " + str(this_window)
        # plt.clf()
        # plt.text(-9, 0, plot_print, fontsize=22)
        # plt.ylim(-10, 10)
        # plt.xlim(-10, 10)
        # plt.show()

        # +1 to include the last timestep
        for i in range(first_timestep, last_timestep - this_h + 1, this_window):
            # do rollout with model from timestep i to i+this_h
            output, obs_agents = Rollout(df, ped_table, model_type, enviornment, dataset, exclude_agents, i, i + this_h,
                                         training_type, model_trained_on_data_part)
            # calculate the SADE and SFDE
            ADE_dict, ADE, NL_ADE_dict, NL_ADE, FDE_dict, FDE, MDE_dict, MDE = _calculate_SADE_SFDE(output, ped_table,
                                                                                                    gt_list, i,
                                                                                                    i + this_h,
                                                                                                    nonlinear_agents,
                                                                                                    exclude_agents)
            if len(ADE_dict) > 0:
                this_ADE.append(ADE)
                this_FDE.append(FDE)
                this_MDE.append(MDE)
            if len(NL_ADE_dict) > 0:
                this_NL_ADE.append(NL_ADE)

            # calculate collision and collision severity
            collisions, collision_time_steps_count, collision_severity, sum_collision_severity, sum_timesteps_for_each_agent = _calculate_collision_severity(
                output, ped_table, obs_agents, i, i + this_h, collision_threshold=0.4, timestep_threshold=3)
            if collision_time_steps_count >= 0:  # added an = sign so that it counts all timesteps. Remove it if you only want to count timesteps where collision occured (this distiction only applies to total_sum_timesteps_for_each_agent)
                this_collision_time_steps_count.append(collision_time_steps_count)
                this_sum_collision_severity.append(sum_collision_severity)
                this_collision_list.append(collisions)
                total_collisions += collisions
                total_sum_timesteps_for_each_agent.append(sum_timesteps_for_each_agent)

        # print("denom", sum(total_sum_timesteps_for_each_agent))
        if len(this_collision_time_steps_count) > 0:
            metrics[this_h] = {'SADE': sum(this_ADE) / len(this_ADE),
                               'NL_SADE': sum(this_NL_ADE) / len(this_NL_ADE),
                               'SFDE': sum(this_FDE) / len(this_FDE),
                               'SMDE': sum(this_MDE) / len(this_MDE),
                               'collision_timesteps': 100 * 2 * sum(this_collision_time_steps_count) / sum(
                                   total_sum_timesteps_for_each_agent),
                               'hard_collision': 100 * 2 * len(total_collisions) / sum(
                                   total_sum_timesteps_for_each_agent),
                               'soft_collision': 100 * 2 * sum(this_sum_collision_severity) / sum(
                                   total_sum_timesteps_for_each_agent),
                               'collision_list': this_collision_list
                               }
        else:
            metrics[this_h] = {'SADE': sum(this_ADE) / len(this_ADE),
                               'NL_SADE': sum(this_NL_ADE) / len(this_NL_ADE),
                               'SFDE': sum(this_FDE) / len(this_FDE),
                               'SMDE': sum(this_MDE) / len(this_MDE),
                               'collision_timesteps': 0.0,
                               'hard_collision_severity': 0,
                               'soft_collision_severity': 0.0
                               }
    return metrics

def gridsearch(model_type, training_type, model_trained_on_data_part, h, ped_table, gt_list, first_timestep, last_timestep, window, nonlinear_agents, exclude_agents):
    
    # If model is not ORCA or SFM, then kill this function
    if model_type not in ["ORCA", "SFM"]:
        return "Invalid model type. Please check inputs and try again."
    
    best_SADE = 999999
    best_SADE_params = []
    best_SADE_metrics = {}
    count = 0
    
    # So at this point in the code, the model type is valid. Now use IF statements to determine which parameters we are doing grid search on
    if model_type == "ORCA":
        NeighborDist_range = list(range(1,6)) # this is equal to [1, 2, 3, 4, 5]
        MaxNeighbors_range = [30] # we decided to hardcode this to 30, since max number of pedestrians in a timestep is 27
        TimeHorizon_range = list(range(1,6)) # this is equal to [1, 2, 3, 4, 5]
        TimeHorizonObst_range = list(range(1,6)) # this is equal to [1, 2, 3, 4, 5]
        Radius_range = [0.2] # we decided to hardcode this to 0.2 since avg shoulder width of human is 0.4
        pref_speed_range = list(np.linspace(1.0, 1.3, 4)) # this is equal to [1.0, 1.1, 1.2, 1.3]
        
        # with this current set of parameters, it will be 5 x 1 x 5 x 5 x 1 x 4 = 500 different parameter combinations
        for this_NeighborDist in NeighborDist_range:
            for this_MaxNeighbors in MaxNeighbors_range:
                for this_TimeHorizon in TimeHorizon_range:
                    for this_TimeHorizonObst in TimeHorizonObst_range:
                        for this_Radius in Radius_range:
                            for this_pref_speed in pref_speed_range:
                                
                                count+=1
                                ORCA_params = [this_NeighborDist, this_MaxNeighbors, this_TimeHorizon, this_TimeHorizonObst, this_Radius, this_pref_speed]
                                
                                this_h = h[0]
                                this_window = window[0]
                                this_ADE = []
                                this_NL_ADE = []
                                this_FDE = []
                                this_MDE = []
                                
                                '''
                                print('calculating metrics for h=', this_h, "with window=", this_window)
                                plot_print = 'calculating metrics for \nh= ' + str(this_h) +" with window= " + str(this_window)
                                plt.clf()
                                plt.text(-9, 0, plot_print, fontsize = 22)
                                plt.ylim(-10,10)
                                plt.xlim(-10,10)
                                plt.show()
                                '''
                                
                                # +1 to include the last timestep
                                for i in range(first_timestep, last_timestep - this_h + 1, this_window):
                                    # do rollout with model from timestep i to i+this_h
                                    output = _ORCA_Rollout(ORCA_params, df, ped_table, model_type, enviornment, dataset, exclude_agents, i, i + this_h, training_type, model_trained_on_data_part)
                                    # calculate the SADE and SFDE
                                    ADE_dict, ADE, NL_ADE_dict, NL_ADE, FDE_dict, FDE, MDE_dict, MDE = _calculate_SADE_SFDE(output, ped_table, gt_list, i, i + this_h, nonlinear_agents, exclude_agents)
                                    if len(ADE_dict) > 0:
                                        this_ADE.append(ADE)
                                        this_FDE.append(FDE)
                                        this_MDE.append(MDE)
                                    if len(NL_ADE_dict) > 0:
                                        this_NL_ADE.append(NL_ADE)
                        
                                if len(this_NL_ADE) > 0:
                                    metrics = {'SADE': sum(this_ADE) / len(this_ADE),
                                                'NL_SADE': sum(this_NL_ADE) / len(this_NL_ADE),
                                                'SFDE': sum(this_FDE) / len(this_FDE),
                                                'SMDE': sum(this_MDE) / len(this_MDE)
                                                }
                                else:
                                    metrics = {'SADE': sum(this_ADE) / len(this_ADE),
                                                'SFDE': sum(this_FDE) / len(this_FDE),
                                                'SMDE': sum(this_MDE) / len(this_MDE)
                                                }
                                if best_SADE > metrics['SADE']:
                                    best_SADE = metrics['SADE']
                                    best_SADE_params = ORCA_params.copy()
                                    best_SADE_metrics = metrics.copy()
       
    print("count", count)
    return best_SADE, best_SADE_params, best_SADE_metrics

def _ORCA_Rollout(ORCA_params, df, goal_df, model_type, enviornment, dataset_name, exclude_agents, start_time, end_time, training_type="None", model_trained_on_data_part = 1):
    
    max_ped = 27
    # take vertice lise obs_orig and make a list of Polygons obs_check. Initialize obs_agents, which keeps track of agents that go through obstacles
    obs_orig = [list(item) for item in enviornment]
    obs_check = [list(item) for item in obs_orig]
    for item in obs_check:
        item.append(item[0])
    obs_check = [Polygon(item) for item in obs_check]
    obs_agents = []

    dt = 0.4

    vel_lin_dict = {}

    final_peds_locations = []
    df_new = df[df["frame_number"] == start_time]
    peds = list(df_new["pedestrian_ID"])
    dict_ped = {}
    for ped in peds:
        dict_ped[ped] = np.array(df_new[df_new["pedestrian_ID"] == ped][["pos_x", "pos_y", "v_x", "v_y"]])[0]
    final_peds_locations.append(dict_ped)
    count = 0
    for time in range(start_time + 1, end_time + 1):
        print(time)
        prev_dict = final_peds_locations[count]
        dict_ped = {}
        df_new = df[df["frame_number"] == time]
        peds = list(df_new["pedestrian_ID"])
        for ped in peds:
            if ped in exclude_agents: # check if agent is excluded. If yes then we do not do rollout on this pedestrian
                dict_ped[ped] = np.array(df_new[df_new["pedestrian_ID"] == ped][["pos_x", "pos_y", "v_x", "v_y"]])[0]
                continue
            elif ped not in list(prev_dict.keys()): # check if this is a pedestrian's first timestep. If yes, then we get their first timestep from the ground truth data
                print("hi")
                dict_ped[ped] = np.array(df_new[df_new["pedestrian_ID"] == ped][["pos_x", "pos_y", "v_x", "v_y"]])[0]        
                continue
            elif ped in obs_agents: # check if pedestrian has gone through an obstacle. If yes then they do not move (so we set their velocity to (0,0))
                outs_curr_vel = np.array([0, 0])
            else: # if we are using a NN, then prepare the model inputs for the center agent
                input_col_list = ['pedestrian_ID', 'curr_loc_x', 'curr_loc_y', 'curr_vel_x', 'curr_vel_y', 'goal_loc_x',
                                  'goal_loc_y']
                input_df = pd.DataFrame(columns=input_col_list)
                arr_goal = np.array(goal_df[goal_df["pedestrian_ID"] == ped][["g_x", "g_y"]])[0]
                arr_posvel = prev_dict[ped]
                concatenated_array = np.concatenate((np.array([ped]), arr_posvel, arr_goal))
                new_row = pd.DataFrame([concatenated_array], columns=input_df.columns)
                input_df = pd.concat([input_df, new_row], ignore_index=True)

                for i in list(prev_dict.keys()): # prepare the model inputs for the neighboring agents
                    if i != ped:
                        arr_goal = np.array(goal_df[goal_df["pedestrian_ID"] == i][["g_x", "g_y"]])[0]
                        arr_posvel = prev_dict[i]
                        concatenated_array = np.concatenate((np.array([i]), arr_posvel, arr_goal))
                        new_row = pd.DataFrame([concatenated_array], columns=input_df.columns)
                        input_df = pd.concat([input_df, new_row], ignore_index=True)

                if model_type == "ORCA": # if we are using ORCA simulator as the model
                    # ORCA input is very similar to the transformer inputs
                    # First initialize the simulator object. Below are the names of the variables and their assigned values
                    '''
                    The following values are from slide 4 ORCA1 from this slidedeck (last edit was June 5, 2023)
                    
                    Timestep: 0.4s (we are doing 0.1 in the actual simulator for 4 steps. Then just keep the last step's information')
                    NeighborDist: 3m
                    MaxNeighbors: 10 agents
                    TimeHorizon: 3s
                    TimeHorizonObst: 3s (we didn't have obstacles so we had the value at 1 in the past, changed to 3 to match AgentTimeHorizon)
                    Radius: 0.2m
                    MaxSpeed: pref_speed + 0.5m/s
                    '''
                    maxSpeed = ORCA_params[5] + 0.5
                    goal_threshold = ORCA_params[4]*2 # I just set this to 2 x agent radius
                    goals = []
                    sim = rvo2.PyRVOSimulator(0.1, ORCA_params[0], ORCA_params[1], ORCA_params[2], ORCA_params[3], ORCA_params[4], maxSpeed)
                    
                    # initialize the obstacles in the simulator (Vertices must be in counter-clockwise solid objects. Negative obstacles, like bounding polygon around an environment, must be in clockwise order) Comment out if you want ORCA to be blind to obstacles
                    
                    for obstacle in enviornment:
                        sim.addObstacle(obstacle)
                    sim.processObstacles() # Documentation is kind of unclear about what this does. But apparently we need it(?)
                    
                    
                    # go through the rows in the input_df. each row is an agent, so we can add agents to the sim one by one
                    for input_df_idx in range(len(input_df)):
                        # get the agent current position, current velocity, and preferred velocity
                        this_agent_curr_loc = (input_df.iloc[input_df_idx]['curr_loc_x'], input_df.iloc[input_df_idx]['curr_loc_y'])
                        this_agent_curr_vel = (input_df.iloc[input_df_idx]['curr_vel_x'], input_df.iloc[input_df_idx]['curr_vel_y'])
                        goals += [input_df.iloc[input_df_idx]['goal_loc_x'], input_df.iloc[input_df_idx]['goal_loc_y']]
                        
                        # NOTE: each agent added is assigned a number so the sim can keep track of it. First agent is assigned number 0. (I don't think I can set it manually to match the pedestrian_ID in input_df)
                        sim.addAgent(this_agent_curr_loc)
                        sim.setAgentVelocity(input_df_idx, this_agent_curr_vel)
                    
                    # After every agent has been added to the sim, use this function to get their first preferred velocities
                    calculate_pref_vel(sim, ORCA_params[5], goals, goal_threshold)
                    
                    # run 1 step of the sim, then recalculate the pref velocities (since we are using 0.1s in the sim() but want 0.4s as the timestep, we do a FOR loop and keep the last step's information. This is to match the SFM baseline)
                    # save the first location
                    agent_0_prev_loc = np.asarray(sim.getAgentPosition(0))
                    for step in range(int(dt/0.1)):
                        sim.doStep()
                        calculate_pref_vel(sim, ORCA_params[5], goals, goal_threshold)
                    
                    # the outputs we want are the velocity and location of the agent 0. Just get them directly from the sim
                    #outs_curr_vel = np.asarray(sim.getAgentVelocity(0))
                    outs_loc = np.asarray(sim.getAgentPosition(0))
                    outs_curr_vel = (outs_loc - agent_0_prev_loc)/dt
                    

            if model_type != "ORCA" or ped in obs_agents: # if we are using ORCA, then we already got them directly from the sim
                outs_loc = prev_dict[ped][0:2] + dt * outs_curr_vel # use the predicted velocity to calculate the next location
                
            dict_ped[ped] = np.concatenate((outs_loc, outs_curr_vel))

            point = Point(list(outs_loc)) # check if the center agent entered an obstacle after this prediction. If yes, then add it to the obs_agents list so we can freeze it for the remaining timesteps
            for obs in obs_check:
                if point.within(obs):
                    obs_agents.append(ped)

        print(dict_ped)
        final_peds_locations.append(dict_ped)
        count = count + 1
    return final_peds_locations # return the locations of agents after the entire rollout is completed


if __name__ == '__main__':
    datasets_dict={"Zara1":[470,865],"Zara2":[610,1051],"Hotel":[610,1167],"ETH":[950,1447]}
    dataset ="Zara2"  # Zara1
    start_time,end_time =datasets_dict[dataset]  # ETH data part 1 = 0, ETH data part 2 = 950, hotel_split,zara2_split:610, zara1_split: 470!

    model_type = "lidar" # CVM transformer lidar ORCA
    training_type = "finetuning" # random finetuning pretrained
    model_trained_on_data_part = "1"    #leave_one_out
    dt = 0.4
    L = end_time - start_time
    h = [ L] # ETH data part 1 = 950 ETH data part 2 = 497  [1, 2, 5, 10, 20, 50, 100, L]
    #h = [1, 2, 5, 10, 20, 50, 100, L]
    window = [math.ceil(x/2) for x in h]

    col_list = ['frame_number', 'pedestrian_ID', 'pos_x', 'pos_z', 'pos_y', 'v_x', 'v_z', 'v_y']
    # obsmat.txt file contains the data for dataset. Specifically the pedestrian location, and velocity at each timestesps
    df = pd.read_csv("Datasets/" + dataset + "/" + "obsmat.txt", sep=r"\s+", header=None)
    df.columns = col_list

    drop_list = ['pos_z', 'v_z']
    df = df.drop(drop_list, axis=1)

    # Make frame_number and pedestrian_ID type int
    df["frame_number"] = df["frame_number"].astype("int")
    df["pedestrian_ID"] = df["pedestrian_ID"].astype("int")

    # Get unique list of the frame numbers. Map them integers starting at 0 for easier
    frames_unique = list(df.frame_number.unique())
    new_frames_unique = list(range(0, len(frames_unique)))
    frame_to_num = {}
    count = 0
    for number in frames_unique:
        frame_to_num[number] = new_frames_unique[count]
        count = count + 1
    df["frame_number"] = df["frame_number"].map(frame_to_num)

    # Open the Ped Table to get the goal location of each pedestrian in the dataset
    with open("Datasets/" + dataset + "/" + "Ped_Table.pkl", 'rb') as handle:
        goal_df = pickle.load(handle)

    if dataset == "ETH":
        enviornment = [[[-3, 3], [-2.99, -13.499], [2.5, -13.5], [2, -5], [0, 1]],
                       [[12.75, 3], [9.499, -2], [9.5, -13.499], [12.7499, -13.5]]]

        load_file = "Datasets/" + dataset + "/" + "ETH_for_Metrics.pkl"
        with open(load_file, 'rb') as handle:
            gt_list = pickle.load(handle)

        exclude_agents = [171, 216]
        # 62 nonlinear agents
        # old nonlinear_agents = [238, 264, 263, 345, 354, 189, 342, 276, 232, 255, 319, 222, 2, 86, 265, 248, 353, 331, 117, 140, 209, 250, 195, 174, 259, 359, 280, 236, 243, 352, 256, 197, 131, 163, 221, 235, 130, 268, 148, 355, 333, 201, 114, 323, 170, 87, 17, 281, 351, 141, 253, 348, 307, 40, 6, 128, 332, 196, 135, 322, 136, 118]

        nonlinear_agents = [2, 17, 40, 86, 87, 117, 118, 128, 130, 131, 135, 136, 140, 141, 148, 163, 170, 174, 189,
                            195, 196, 197, 201, 209, 212, 221, 222, 232, 235, 236, 238, 243, 250, 255, 256, 263, 264,
                            276, 280, 307, 319, 322, 323, 328, 331, 332, 333, 342, 345, 348, 351, 352, 353, 354, 355,
                            359]
    elif dataset=="ETH_rotate":
        # environment with vertices in clockwise order (for ORCA sim we need them in counter-clockwise)
        '''
        enviornment = [[[-3, 3], [0, 1], [2, -5], [2.5, -13.5], [-2.99, -13.499]],
         [[12.75, 3], [12.7499, -13.5], [9.5, -13.499], [9.499, -2]]]
        '''
        # environment with vertices in counter-clockwise order
        enviornment = [[[-3, 3], [-2.99, -13.499], [2.5, -13.5], [2, -5], [0, 1]],
         [[12.75, 3], [9.499, -2], [9.5, -13.499], [12.7499, -13.5]]]

        load_file = "Datasets/" + dataset + "/" + "ETH_rotate_for_Metrics.pkl"
        with open(load_file, 'rb') as handle:
            gt_list = pickle.load(handle)
            
        exclude_agents = [171, 216]
        # 62 nonlinear agents
        # old nonlinear_agents = [238, 264, 263, 345, 354, 189, 342, 276, 232, 255, 319, 222, 2, 86, 265, 248, 353, 331, 117, 140, 209, 250, 195, 174, 259, 359, 280, 236, 243, 352, 256, 197, 131, 163, 221, 235, 130, 268, 148, 355, 333, 201, 114, 323, 170, 87, 17, 281, 351, 141, 253, 348, 307, 40, 6, 128, 332, 196, 135, 322, 136, 118]

        nonlinear_agents = [2, 17, 40, 86, 87, 117, 118, 128, 130, 131, 135, 136, 140, 141, 148, 163, 170, 174, 189, 195, 196, 197, 201, 209, 212, 221, 222, 232, 235, 236, 238, 243, 250, 255, 256, 263, 264, 276, 280, 307, 319, 322, 323, 328, 331, 332, 333, 342, 345, 348, 351, 352, 353, 354, 355, 359]
    elif dataset=="Zara1":
        enviornment = [[[-0.7989999999999999, 17.198999999999998], [-6.3, 17.2], [-6.2989999999999995, 4.7], [-3.3, 4.699], [-3.299, 6.7], [-1.8, 6.699], [-1.7990000000000002, 14.7], [-0.8, 14.699]],[[7.250, 12.000], [5.500, 11.750], [5.499, 6.500], [7.249, 6.250]]]

        load_file = "Datasets/" + dataset + "/" + "zara1_for_Metrics.pkl"
        with open(load_file, 'rb') as handle:
            gt_list = pickle.load(handle)
            
        exclude_agents = [8]
        # 61 nonlinear agents
        # old nonlinear_agents = [90, 66, 65, 21, 130, 6, 89, 16, 37, 140, 27, 133, 17, 22, 127, 78, 129, 126, 77, 68, 20, 26, 57, 83, 128, 19, 15, 148, 95, 54, 134, 56, 99, 58, 100, 138, 82, 81, 91, 40, 93, 70, 123, 59, 4, 92, 139, 122, 84, 79, 137, 39, 120, 55, 63, 14, 75, 124, 10, 119]

        nonlinear_agents = [6, 9, 16, 17, 19, 20, 21, 22, 26, 27, 32, 33, 34, 35, 37, 40, 54, 56, 57, 58, 59, 65, 66, 68, 70, 77, 78, 81, 82, 83, 89, 90,91, 93, 95, 99, 100, 123, 126, 127, 128, 129, 130, 133, 134, 138, 140, 148]
    elif dataset=="Zara2":

        enviornment=[[[-0.7989999999999999, 17.198999999999998], [-6.3, 17.2], [-6.2989999999999995, 4.7], [-3.3, 4.699], [-3.299, 6.7], [-1.8, 6.699], [-1.7990000000000002, 14.7], [-0.8, 14.699]]]
        load_file = "Datasets/" + dataset + "/" + "zara2_for_Metrics.pkl"
        with open(load_file, 'rb') as handle:
            gt_list = pickle.load(handle)

        exclude_agents = [38, 39, 40, 43, 44, 69, 70, 71, 76, 77, 111, 112, 130, 131, 169, 174, 177, 178]
        
        # old nonlinear_agents = [169, 76, 77, 43, 41, 139, 20, 140, 78, 54, 21, 130, 17, 7, 117, 84, 115, 18, 153, 122, 144, 108, 66, 158, 97, 143, 156, 157, 171, 168, 56, 82, 145, 164, 83, 165, 91, 107, 137, 23, 154, 40, 151, 128, 166, 147, 39, 34, 55, 85, 127, 124, 98, 99]
        
        nonlinear_agents = [7, 15, 17, 18, 20, 21, 23, 28, 34, 41, 54, 55, 56, 66, 78, 82, 83, 84, 85, 91, 92, 97, 98, 99, 106, 107, 108, 115, 117, 118, 120, 121, 122, 124, 127, 128, 137, 139, 140, 143, 144, 145, 147, 151, 153, 154, 156, 157, 158, 164, 165, 166, 167, 168, 171]


    elif dataset=="Hotel":

        enviornment =[[[-0.827, -5.126], [-0.89199, -5.0134], [-1.022, -5.01343], [-1.087, -5.126], [-1.022, -5.23858], [-0.89199, -5.23859]], [[-0.689, -1.76], [-0.754, -1.6474], [-0.88399, -1.6475], [-0.949, -1.76], [-0.88399, -1.87258], [-0.754, -1.872583]], [[-0.727, 1.917], [-0.792, 2.02958], [-0.92199, 2.02957], [-0.987, 1.917], [-0.92199, 1.8044], [-0.792, 1.80441]], [[-0.517, -10.065], [-0.613, -7.755], [-1.199, -7.737], [-1.2, -10.015]]]
        # old nonlinear_agents = [288, 299, 206, 317, 315, 190, 97, 36, 119, 230, 225, 137, 261, 115]
        nonlinear_agents=[288, 299, 206, 317]
        load_file = "Datasets/" + dataset + "/" + "Hotel_for_Metrics.pkl"
        with open(load_file, 'rb') as handle:
            gt_list = pickle.load(handle)
        exclude_agents = []

    # Generate the video of rollout    
    '''
    arr=Rollout(df, goal_df, model_type, enviornment, dataset, exclude_agents, start_time,end_time,training_type, model_trained_on_data_part)
    movie_name = dataset+"_data_"+ str(model_trained_on_data_part) + "_"+training_type + '_' + model_type
    video_generater(df,arr,movie_name,dataset)
    '''
    
    
    # preparing data and save to file
    '''
    #with open('line.pkl', 'wb') as f:  # open a text file
     #   pickle.dump(arr, f)  # serialize the list
    # with open('rollout_Hotel_To_ETH.pkl', 'rb') as f:
    #     arr = pickle.load(f)
    '''

    # Calculate metrics
    
    metrics = calculate_metrics(model_type, training_type, model_trained_on_data_part, h, goal_df, gt_list, start_time, end_time, window, nonlinear_agents, exclude_agents)
    print(model_type)
    print(training_type + '_' + dataset.lower() + '_' + str(model_trained_on_data_part) + ".pth")
    for key in metrics.keys():
        print("metrics for h=", key)
        this_h = metrics[key]
        for h_key in this_h.keys():
            if h_key not in ["collision_list", "collision_timesteps","SADE","SFDE","NL_SADE"]:
                print(h_key, this_h[h_key])
        print()
    
    
        
    '''
    best_SADE, best_SADE_params, best_SADE_metrics = gridsearch(model_type, training_type, model_trained_on_data_part, h, goal_df, gt_list, start_time, end_time, window, nonlinear_agents, exclude_agents)
    print(model_type)
    print(dataset.lower())
    for key in best_SADE_metrics.keys():
        if key not in ["SFDE"]:
            print(key, best_SADE_metrics[key])
    
    print()
    
    print("Parameters used:")
    print("NeighborDiist", best_SADE_params[0])
    print("MaxNeighbors", best_SADE_params[1])
    print("TimeHorizon", best_SADE_params[2])
    print("TimeHorizonObst", best_SADE_params[3])
    print("Radius", best_SADE_params[4])
    print("pref_speed", best_SADE_params[5])
    '''


