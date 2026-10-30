import torch
import torch.nn as nn
import numpy as np

from torch.autograd import Variable

import numpy as np
import torch
import itertools
from torch.autograd import Variable

import pickle

device="cuda:0"


def getGridMask(frame, dimensions, num_person, neighborhood_size, grid_size, is_occupancy=False):
    '''
    This function computes the binary mask that represents the
    occupancy of each ped in the other's grid
    params:
    frame : This will be a MNP x 3 matrix with each row being [pedID, x, y]
    dimensions : This will be a list [width, height]
    neighborhood_size : Scalar value representing the size of neighborhood considered
    grid_size : Scalar value representing the size of the grid discretization
    num_person : number of people exist in given frame
    is_occupancy: A flag using for calculation of accupancy map

    '''
    mnp = num_person

    width, height = dimensions[0], dimensions[1]
    if is_occupancy:
        frame_mask = np.zeros((mnp, grid_size ** 2))
    else:
        frame_mask = np.zeros((mnp, mnp, grid_size ** 2))
    frame_np = frame.data.numpy()

    # width_bound, height_bound = (neighborhood_size/(width*1.0)), (neighborhood_size/(height*1.0))
    width_bound, height_bound = (neighborhood_size / (width * 1.0)) * 2, (neighborhood_size / (height * 1.0)) * 2
    # print("weight_bound: ", width_bound, "height_bound: ", height_bound)

    # instead of 2 inner loop, we check all possible 2-permutations which is 2 times faster.
    list_indices = list(range(0, mnp))
    for real_frame_index, other_real_frame_index in itertools.permutations(list_indices, 2):
        current_x, current_y = frame_np[real_frame_index, 0], frame_np[real_frame_index, 1]

        width_low, width_high = current_x - width_bound / 2, current_x + width_bound / 2
        height_low, height_high = current_y - height_bound / 2, current_y + height_bound / 2

        other_x, other_y = frame_np[other_real_frame_index, 0], frame_np[other_real_frame_index, 1]

        # if (other_x >= width_high).all() or (other_x < width_low).all() or (other_y >= height_high).all() or (other_y < height_low).all():
        if (other_x >= width_high) or (other_x < width_low) or (other_y >= height_high) or (other_y < height_low):
            # Ped not in surrounding, so binary mask should be zero
            # print("not surrounding")
            continue
        # If in surrounding, calculate the grid cell
        cell_x = int(np.floor(((other_x - width_low) / width_bound) * grid_size))
        cell_y = int(np.floor(((other_y - height_low) / height_bound) * grid_size))

        if cell_x >= grid_size or cell_x < 0 or cell_y >= grid_size or cell_y < 0:
            continue

        if is_occupancy:
            frame_mask[real_frame_index, cell_x + cell_y * grid_size] = 1
        else:
            # Other ped is in the corresponding grid cell of current ped
            frame_mask[real_frame_index, other_real_frame_index, cell_x + cell_y * grid_size] = 1

    return frame_mask


def getSequenceGridMask(sequence, dimensions, neighborhood_size, grid_size, using_cuda,
                        is_occupancy=False):
    '''
    Get the grid masks for all the frames in the sequence
    params:
    sequence : A numpy matrix of shape SL x MNP x 3
    dimensions : This will be a list [width, height]
    neighborhood_size : Scalar value representing the size of neighborhood considered
    grid_size : Scalar value representing the size of the grid discretization
    using_cuda: Boolean value denoting if using GPU or not
    is_occupancy: A flag using for calculation of accupancy map
    '''
    sl = len(sequence)
    sequence_mask = []

    for i in range(sl):
        #print(sequence[i].shape)
        mask = Variable(torch.from_numpy(
            getGridMask(sequence[i], dimensions, sequence[i].shape[0], neighborhood_size, grid_size,
                        is_occupancy)).float())
        if using_cuda:
            mask = mask.cuda()
        sequence_mask.append(mask)

    return sequence_mask


class SocialModel(nn.Module):

    def __init__(self):
        '''
        Initializer function
        params:
        args: Training arguments
        infer: Training or test time (true if test time)
        '''
        super(SocialModel, self).__init__()


        self.use_cuda = True


        self.seq_length = 1


        # Store required sizes
        self.rnn_size = 128
        self.grid_size = 4
        self.embedding_size = 64
        self.input_size = 2
        self.output_size = 62
        # self.maxNumPeds = args.maxNumPeds
        # self.seq_length = args.seq_length

        # The LSTM cell
        self.cell = nn.LSTMCell(2 * self.embedding_size, self.rnn_size)



        self.dropout=0.0001

        # Linear layer to embed the input position
        self.input_embedding_layer = nn.Linear(self.input_size, self.embedding_size)
        # Linear layer to embed the social tensor
        self.tensor_embedding_layer = nn.Linear(self.grid_size * self.grid_size * self.rnn_size, self.embedding_size)

        # Linear layer to map the hidden state of LSTM to output
        self.output_layer = nn.Linear(self.rnn_size, self.output_size)

        # ReLU and dropout unit
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout)

        self.final1=nn.Linear(self.output_size+2,32)
        self.final2 = nn.Linear(32, 2)


    def getSocialTensor(self, grid, hidden_states):
        '''
        Computes the social tensor for a given grid mask and hidden states of all peds
        params:
        grid : Grid masks
        hidden_states : Hidden states of all peds
        '''
        # Number of peds
        numNodes = grid.size()[0]

        # Construct the variable
        social_tensor = Variable(torch.zeros(numNodes, self.grid_size * self.grid_size, self.rnn_size))
        if self.use_cuda:
            social_tensor = social_tensor.cuda()

        # For each ped
        for node in range(numNodes):
            # Compute the social tensor
            social_tensor[node] = torch.mm(torch.t(grid[node]), hidden_states)

        # Reshape the social tensor
        social_tensor = social_tensor.view(numNodes, self.grid_size * self.grid_size * self.rnn_size)
        return social_tensor

    # def forward(self, input_data, grids, hidden_states, cell_states ,PedsList, num_pedlist,dataloader, look_up):
    def forward(self, input_data,grids,goal):

        hidden_states = Variable(torch.zeros(input_data.shape[1], self.rnn_size)).to(device)
        cell_states = Variable(torch.zeros(input_data.shape[1], self.rnn_size)).to(device)






        numNodes = input_data.shape[1]
        # outputs = Variable(torch.zeros(self.seq_length * numNodes, self.output_size))
        # if self.use_cuda:
        #     outputs = outputs.cuda()

        # For each frame in the sequence
        for framenum, frame in enumerate(input_data):

            nodes_current = frame.to(device)
            # Get the corresponding grid masks
            grid_current = grids[framenum]

            # Get the corresponding hidden and cell states
            hidden_states_current = hidden_states


            cell_states_current = cell_states


            social_tensor = self.getSocialTensor(grid_current, hidden_states_current)

            # Embed inputs

            input_embedded = self.dropout(self.relu(self.input_embedding_layer(nodes_current)))
            # Embed the social tensor
            tensor_embedded = self.dropout(self.relu(self.tensor_embedding_layer(social_tensor)))

            # Concat input
            concat_embedded = torch.cat((input_embedded, tensor_embedded), 1)


            h_nodes, c_nodes = self.cell(concat_embedded, (hidden_states_current, cell_states_current))


            # Compute the output
            #outputs[framenum * numNodes + corr_index.data] = self.output_layer(h_nodes)

            # Update hidden and cell states
            hidden_states = h_nodes
            cell_states= c_nodes

        # Reshape outputs
        # outputs_return = Variable(torch.zeros(self.seq_length, numNodes, self.output_size))
        # if self.use_cuda:
        #     outputs_return = outputs_return.cuda()
        # for framenum in range(self.seq_length):
        #     for node in range(numNodes):
        #         outputs_return[framenum, node, :] = outputs[framenum * numNodes + node, :]

        outputs_return=self.output_layer(h_nodes)

        out=outputs_return[0]
        outputs_return=self.final2(self.relu(self.final1(torch.cat((out, goal), dim=0) )))


        return outputs_return, hidden_states, cell_states



# epoch=0/
# x_seq=torch.ones(2,10,2)
# goal=torch.ones(2,).to(device)
# dataset_data=[720,576]
# rnn_size=128
# use_cuda=True
# grid_size=4
# neighborhood_size=32
# if(epoch is 0):
#     grid_seq = getSequenceGridMask(x_seq, dataset_data,neighborhood_size, grid_size, use_cuda)
# net = SocialModel().to(device)
# output,_,_=net(x_seq,grid_seq,goal)
# print(output.shape)


def get_data(data_folder):
    x = []
    y = []

    with open(data_folder, 'rb') as handle:
        sample_list = pickle.load(handle)

    for sample_idx in range(len(sample_list)):
        this_sample = sample_list[sample_idx]

        curr_loc_df = this_sample[0][["curr_loc_x", "curr_loc_y"]]
        curr_vel_df = this_sample[0][["curr_vel_x", "curr_vel_y"]]
        pref_vel_df = this_sample[0][["goal_loc_x", "goal_loc_y"]]
        prev_loc_df=np.array(curr_loc_df)-0.4*np.array(curr_vel_df)
        loc_df=np.stack((prev_loc_df, np.array(curr_loc_df)), axis=0)
        goal=np.array(pref_vel_df.iloc[0])




        y_curr_vel = np.array(this_sample[1][["pred_vel_x", "pred_vel_y"]]).reshape(2,)

        # append the sample to the x and y lists
        x.append((loc_df,goal))
        y.append(y_curr_vel)
    #     print(x.shape)
    #     # turn the lists into arrays. Then return arrays
    # x = np.vstack(x)  # (#numsamples, 11, 6)
    # y = np.vstack(y)  # (#numsamples, 11, 2)

    return x, y


def update_df(df, step, data_list):
    df.loc[step] = data_list
    return df


def normalize_vector(vector):
    norm_vec = vector / np.sqrt(sum(np.array(vector) ** 2))
    if np.isnan(norm_vec).any():
        norm_vec[np.isnan(norm_vec)] = 0.0
    return norm_vec


def train_model():
    train_loss = []
    valid_loss = []
    device = 'cuda:0'
    dt = 0.10

    print('extract training trajectories', flush=True)
    print()
    x_train, y_train = get_data("Datasets/Zara1/data_for_model/transformer/Zara1_train_part_1.pkl")

    # percent_train = int(len(x_train) * 0.9)
    # x_val, y_val = x_train[percent_train:], y_train[percent_train:]
    # x_train, y_train = x_train[:percent_train], y_train[:percent_train]

    from sklearn.model_selection import train_test_split


    # Split the lists into training and test sets
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1,
                                                                                random_state=42)

    x_test, y_test = get_data("Datasets/Zara1/data_for_model/transformer/Zara1_train_part_2.pkl")

    print(f'Training X shape: {len(x_train)}')
    print(f'Training Y shape: {len(y_train)}')
    print('extract validation trajectories', flush=True)
    print(f'Validation X shape: {len(x_val)}')
    print(f'Validation Y shape: {len(y_val)}')
    print()
    print(f'test X shape: {len(x_test)}')
    print(f'test Y shape: {len(y_test)}')
    print()

    # uses the transformer yaml file to get some variables
    num_epochs = 100
    patience = 50
    # if train_type == 'early_stopping':
    #     print("Training with early stopping. Max number of epochs is ", num_epochs, ".")
    # else:
    #     print("Training for set number of ", num_epochs, " epochs.")
    learning_rate = float(0.0001)
    best_loss = 1e9
    best_epoch = 0
    batch_size = 1  # 1024

    # initialize model
    dataset_data = [720, 576]
    rnn_size = 128
    use_cuda = True
    grid_size = 4
    neighborhood_size = 32

    #grid_seq = getSequenceGridMask(x_seq, dataset_data, neighborhood_size, grid_size, use_cuda)
    model = SocialModel().to(device)
    #output, _, _ = model(x_seq, grid_seq, goal)
    #model = net().to(device)

    # if train_from_scratch != True:
    #     model.load_state_dict(torch.load("Weights/pretrained/transformer/RVO_DSFM_500000.pth"))
    # initialize loss function and optimizer (this is what is used to update the model in back propegation)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)





    train_loss_lst = []

    # Train the model
    for epoch in range(num_epochs):
        # X is a torch Variable
        n_train =len(y_train)
        permutation = torch.randperm(n_train)

        loss_avg = 0
        model.train()
        for i in range(0, n_train, batch_size):
            indices = permutation[i]  # size is 1024 except the last batch (which may be < 1024 )

            batch_x, batch_y = x_train[indices], y_train[indices]
            batch_x,batch_goal=batch_x[0],batch_x[1]
            # convert to tensors
            batch_x = torch.from_numpy(batch_x)
            grid_seq = getSequenceGridMask(batch_x, dataset_data, neighborhood_size, grid_size, use_cuda)
            batch_x=batch_x.to(device)
            batch_y = torch.from_numpy(batch_y).to(device)
            batch_goal=torch.from_numpy(batch_goal).to(device)

            n_batch = batch_x.shape[0]  # 1024
            batch_y = batch_y.float()

            # add augmentation later

            outs,_,_ = model(batch_x.float(),grid_seq,batch_goal.float())

            # obtain the loss function
            loss = criterion(outs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_avg += loss.item() * n_batch / n_train


        with torch.no_grad():
            model.eval()
            # X is a torch Variable
            n_val = len(y_val) # 1793
            permutation = torch.randperm(n_val)

            val_loss_avg = 0
            for i in range(0, n_val, batch_size):
                indices = permutation[i]  # size is 1024 except the last batch (which may be < 1024 )

                batch_x, batch_y = x_val[indices], y_val[indices]
                batch_x, batch_goal = batch_x[0], batch_x[1]
                # convert to tensors
                batch_x = torch.from_numpy(batch_x)
                grid_seq = getSequenceGridMask(batch_x, dataset_data, neighborhood_size, grid_size, use_cuda)
                batch_x = batch_x.to(device)
                batch_y = torch.from_numpy(batch_y).to(device)
                batch_goal = torch.from_numpy(batch_goal).to(device)

                n_batch = batch_x.shape[0]  # 1024
                batch_y = batch_y.float()

                # add augmentation later

                outs, _, _ = model(batch_x.float(), grid_seq, batch_goal.float())

                # obtain the loss function
                # obtain the loss function
                loss = criterion(outs, batch_y)

                val_loss_avg += loss.item() * n_batch / n_val
            if val_loss_avg < best_loss:
                print('best so far (saving):')
                best_loss = val_loss_avg
                best_epoch = epoch
                print("hi")
                torch.save(model.state_dict(), "baseline-zara1.pth")
                print("bye")

                print(f'New best epoch {epoch}: train loss {loss_avg:.8f}, val loss {val_loss_avg:.8f}')

            else:
                print(f'Epoch {epoch}: train loss {loss_avg:.8f}')
            if epoch - best_epoch > patience:
                break





    #model.load_state_dict(torch.load(save_folder + name + ".pth"))
    with torch.no_grad():
        model.eval()
        # X is a torch Variable
        n_val = len(x_test) # 1793
        permutation = torch.randperm(n_val)

        val_loss_avg = 0
        for i in range(0, n_val, batch_size):
            indices = permutation[i]
            batch_x, batch_y = x_test[indices], y_test[indices]
            batch_x, batch_goal = batch_x[0], batch_x[1]
            # convert to tensors
            batch_x = torch.from_numpy(batch_x)
            grid_seq = getSequenceGridMask(batch_x, dataset_data, neighborhood_size, grid_size, use_cuda)
            batch_x = batch_x.to(device)
            batch_y = torch.from_numpy(batch_y).to(device)
            batch_goal = torch.from_numpy(batch_goal).to(device)

            n_batch = batch_x.shape[0]  # 1024
            batch_y = batch_y.float()

            # add augmentation later

            outs, _, _ = model(batch_x.float(), grid_seq, batch_goal.float())

            # obtain the loss function
            # obtain the loss function
            loss = criterion(outs, batch_y)

            val_loss_avg += loss.item() * n_batch / n_val
    print(f"test loss: {val_loss_avg}")
    return train_loss, valid_loss


if __name__ == '__main__':
    # train
    train_loss, valid_loss = train_model(
                                         )



