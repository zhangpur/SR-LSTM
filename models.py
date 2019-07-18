'''
Main Models
Author: Pu Zhang
Date: 2019/7/1
'''

from utils import *
from basemodel import *

class LSTM(nn.Module):
    def __init__(self,args):
        super(LSTM,self).__init__()
        self.args=args
        self.ifdropout=args.ifdropout
        self.using_cuda=args.using_cuda
        self.inputLayer=nn.Linear(args.input_size,args.input_embed_size)
        self.cell = LSTMCell(args.input_embed_size, args.rnn_size)
        self.outputLayer=nn.Linear(args.rnn_size,args.output_size)
        self.dropout=nn.Dropout(args.dropratio)

        nn.init.constant(self.inputLayer.bias, 0.0)
        nn.init.normal(self.inputLayer.weight,std=args.std_in)
        nn.init.xavier_uniform(self.cell.weight_ih)
        nn.init.orthogonal(self.cell.weight_hh,gain=0.001)

        nn.init.constant(self.cell.bias_ih, 0.0)
        nn.init.constant(self.cell.bias_hh, 0.0)
        n = self.cell.bias_ih.size(0)
        nn.init.constant(self.cell.bias_ih[n // 4:n // 2], 1.0)
        nn.init.constant(self.outputLayer.bias, 0.0)
        nn.init.normal(self.outputLayer.weight,std=args.std_out)
        self.input_Ac=nn.ReLU()

    def forward(self, inputs,iftest=False):
        nodes_abs, nodes_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum = inputs

        num_Ped = nodes_norm.shape[1]

        outputs = torch.zeros(nodes_norm.shape[0], num_Ped, self.args.output_size)
        hidden_states = torch.zeros(num_Ped, self.args.rnn_size)
        cell_states = torch.zeros(num_Ped, self.args.rnn_size)

        if self.args.using_cuda:
            outputs=outputs.cuda()
            hidden_states = hidden_states.cuda()
            cell_states = cell_states.cuda()
        # For each frame in the sequence
        for framenum in range(self.args.seq_length-1):
            if framenum >= self.args.obs_length and iftest:
                node_index = seq_list[self.args.obs_length - 1] > 0
                nodes_current = outputs[framenum - 1, node_index]
            else:
                node_index=seq_list[framenum]>0
                nodes_current = nodes_norm[framenum,node_index]

            hidden_states_current=hidden_states[node_index]
            cell_states_current=cell_states[node_index]
            input_embedded = self.dropout(self.input_Ac(self.inputLayer(nodes_current)))
            _,hidden_states_current, cell_states_current = self.cell.forward(input_embedded, (hidden_states_current,cell_states_current))
            if self.using_cuda:
                outputs_current = self.outputLayer(hidden_states_current).cuda()
            else:
                outputs_current = self.outputLayer(hidden_states_current)
            outputs[framenum,node_index]=outputs_current
            hidden_states[node_index]=hidden_states_current
            cell_states[node_index] = cell_states_current
        return outputs, hidden_states, cell_states,(0,0,0)
class SRLSTM(nn.Module):
    def __init__(self, args):
        super(SRLSTM, self).__init__()
        self.args = args
        self.ifdropout = args.ifdropout
        self.using_cuda = args.using_cuda
        self.inputLayer = nn.Linear(args.input_size, args.input_embed_size)
        self.cell = LSTMCell(args.input_embed_size, args.rnn_size)

        self.gcn = GCN(args,self.args.rela_embed_size, args.rnn_size)

        if self.args.passing_time>1:
            self.gcn1 = GCN(args, self.args.rela_embed_size, args.rnn_size)
            if self.args.passing_time==3:
                self.gcn2 = GCN(args, self.args.rela_embed_size, args.rnn_size)

        self.outputLayer = nn.Linear(args.rnn_size, args.output_size)
        self.dropout = nn.Dropout(args.dropratio)

        self.input_Ac = nn.ReLU()

        if args.using_cuda:
            self = self.cuda(device=args.gpu)
        self.init_parameters()

    def init_parameters(self):
        nn.init.constant(self.inputLayer.bias, 0.0)
        nn.init.normal(self.inputLayer.weight, std=self.args.std_in)

        nn.init.xavier_uniform(self.cell.weight_ih)
        nn.init.orthogonal(self.cell.weight_hh, gain=0.001)

        nn.init.constant(self.cell.bias_ih, 0.0)
        nn.init.constant(self.cell.bias_hh, 0.0)
        n = self.cell.bias_ih.size(0)
        nn.init.constant(self.cell.bias_ih[n // 4:n // 2], 1.0)

        nn.init.constant(self.outputLayer.bias, 0.0)
        nn.init.normal(self.outputLayer.weight, std=self.args.std_out)

    def forward(self, inputs,iftest=False):

        nodes_abs, nodes_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum=inputs
        num_Ped = nodes_norm.shape[1]
        
        outputs=torch.zeros(nodes_norm.shape[0],num_Ped, self.args.output_size)
        hidden_states = torch.zeros(num_Ped, self.args.rnn_size)
        cell_states = torch.zeros(num_Ped, self.args.rnn_size)

        value1_sum=0
        value2_sum=0
        value3_sum=0

        if self.using_cuda:
            outputs=outputs.cuda()
            hidden_states = hidden_states.cuda()
            cell_states = cell_states.cuda()
            
        # For each frame in the sequence
        for framenum in range(self.args.seq_length-1):
            if framenum >= self.args.obs_length and iftest:
                node_index = seq_list[self.args.obs_length - 1] > 0
                nodes_current = outputs[framenum - 1, node_index].clone()

                nodes_abs=shift_value[framenum,node_index]+nodes_current

                nodes_abs=nodes_abs.repeat(nodes_abs.shape[0], 1, 1)
                corr_index=nodes_abs.transpose(0,1)-nodes_abs
            else:
                node_index=seq_list[framenum]>0
                nodes_current = nodes_norm[framenum,node_index]
                corr = nodes_abs[framenum, node_index].repeat(nodes_current.shape[0], 1, 1)
                nei_index = nei_list[framenum, node_index]
                nei_index = nei_index[:, node_index]
                # relative coords
                corr_index = corr.transpose(0,1)-corr
                nei_num_index=nei_num[framenum,node_index]

            hidden_states_current=hidden_states[node_index]
            cell_states_current=cell_states[node_index]

            input_embedded = self.dropout(self.input_Ac(self.inputLayer(nodes_current)))

            lstm_state = self.cell.forward(input_embedded, (hidden_states_current,cell_states_current))

            for p in range(self.args.passing_time ):
                if p==0:
                    lstm_state, look = self.gcn.forward(corr_index, nei_index, nei_num_index, lstm_state,self.gcn.W_nei)
                    value1, value2, value3 = look
                if p==1:
                    lstm_state, look = self.gcn1.forward(corr_index, nei_index, nei_num_index, lstm_state,self.gcn1.W_nei)


            _, hidden_states_current, cell_states_current = lstm_state

            value1_sum+=value1
            value2_sum+=value2
            value3_sum+=value3

            outputs_current = self.outputLayer(hidden_states_current)
            outputs[framenum,node_index]=outputs_current
            hidden_states[node_index]=hidden_states_current
            cell_states[node_index] = cell_states_current

        return outputs, hidden_states, cell_states,(value1_sum/self.args.seq_length,value2_sum/self.args.seq_length,value3_sum/self.args.seq_length)
class SocialLSTM(nn.Module):
    def __init__(self,args):
        super(SocialLSTM,self).__init__()
        self.grid_size=args.grid_size
        self.args=args
        self.ifdropout=args.ifdropout
        self.using_cuda=args.using_cuda
        self.inputLayer=nn.Linear(self.args.input_size,self.args.input_embed_size)
        self.TensorEmbedLayer = nn.Linear(self.grid_size*self.grid_size*self.args.rnn_size, self.args.input_embed_size)

        self.cell = nn.LSTMCell(args.input_embed_size*2, args.rnn_size)
        self.outputLayer=nn.Linear(args.rnn_size,args.output_size)
        self.dropout=nn.Dropout(args.dropratio)
        self.relu=nn.ReLU()

        nn.init.constant(self.TensorEmbedLayer.bias, 0.0)

        nn.init.xavier_uniform(self.TensorEmbedLayer.weight)

        nn.init.constant(self.inputLayer.bias, 0.0)
        nn.init.normal(self.inputLayer.weight,std=args.std_in)

        nn.init.xavier_uniform(self.cell.weight_ih)
        nn.init.orthogonal(self.cell.weight_hh,gain=0.001)

        nn.init.constant(self.cell.bias_ih, 0.0)
        nn.init.constant(self.cell.bias_hh, 0.0)
        n = self.cell.bias_ih.size(0)
        nn.init.constant(self.cell.bias_ih[n // 4:n // 2], 1.0)
        nn.init.constant(self.outputLayer.bias, 0.0)
        nn.init.normal(self.outputLayer.weight,std=args.std_out)
        if args.using_cuda:
            self=self.cuda(device=args.gpu)

    def forward(self, inputs,iftest=False):
        nodes_abs, nodes_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum=inputs
        num_Ped = nodes_norm.shape[1]
        outputs=torch.zeros(nodes_norm.shape[0],num_Ped, self.args.output_size)
        hidden_states = torch.zeros(num_Ped, self.args.rnn_size)
        cell_states = torch.zeros(num_Ped, self.args.rnn_size)
        scale_sum=0
        mean_sum=0
        std_sum=0

        if self.using_cuda:
            outputs=outputs.cuda()
            hidden_states = hidden_states.cuda()
            cell_states = cell_states.cuda()
        # For each frame in the sequence
        for framenum in range(self.args.seq_length-1):
            if framenum >= self.args.obs_length and iftest:
                node_index = seq_list[self.args.obs_length - 1] > 0
                nodes_current = outputs[framenum - 1, node_index].clone()
                nodes_abs=shift_value[framenum,node_index]+nodes_current

                nodes_abs=nodes_abs.repeat(nodes_abs.shape[0], 1, 1)
                corr_index=nodes_abs.transpose(0,1)-nodes_abs
            else:
                node_index=seq_list[framenum]>0
                nodes_current = nodes_norm[framenum,node_index]
                corr = nodes_abs[framenum, node_index].repeat(nodes_current.shape[0], 1, 1)
                nei_index = nei_list[framenum, node_index]
                nei_index = nei_index[:, node_index]
                # relative coords
                corr_index = corr.transpose(0,1)-corr

            hidden_states_current=hidden_states[node_index]
            cell_states_current=cell_states[node_index]

            grid_current = self.getGridMask(nodes_current.shape[0], corr_index,nei_index)
            social_tensor = self.getSocialTensor(grid_current, hidden_states_current)
            tensor_embedded = self.dropout(self.relu(self.TensorEmbedLayer(social_tensor)))
            input_embedded = self.dropout(self.relu(self.inputLayer(nodes_current)))
            concat_embedded = torch.cat((input_embedded, tensor_embedded), 1)

            lstm_state = self.cell.forward(concat_embedded, (hidden_states_current,cell_states_current))
            hidden_states_current, cell_states_current = lstm_state

            outputs_current = self.outputLayer(hidden_states_current)
            outputs[framenum,node_index]=outputs_current

            hidden_states[node_index]=hidden_states_current
            cell_states[node_index] = cell_states_current

        return outputs, hidden_states, cell_states,(scale_sum/self.args.seq_length,mean_sum/self.args.seq_length,std_sum/self.args.seq_length)

    def getGridMask(self,numPeds, corr_index,nei_index):
        '''
        Comput the binary mask of the pairwise occupancy grid
        Params:
            numPeds: number of all Pedestrians
            corr_index: their relative locations
            nei_index: neighbor exsistence flag
        Return:
            grid_size : Scalar value representing the size of the grid discretization
        '''
        grid = torch.full((numPeds,numPeds,self.args.grid_size,self.args.grid_size), 0, device=torch.device("cuda")) #cubic tensor H
        corr_index_true=corr_index*torch.cat((nei_index.view(numPeds,numPeds,1),nei_index.view(numPeds,numPeds,1)),2)

        # relative locations between all pedestrians
        corr_index_true[corr_index_true==0]=np.Inf

        # valid relative locations considering neighbors standing in neighbor_thred * neighbor_thred
        corr_index_true[abs(corr_index_true)>(self.args.nei_thred_slstm-1e-5)]=np.Inf

        # relative location validation mask
        gridmask=corr_index_true!=np.Inf
        gridmask[:,:,0]=gridmask[:,:,0] * gridmask[:,:,1]
        gridmask[:, :, 1]=gridmask[:,:,0]
        if torch.sum(gridmask)>0:
            # bin side size of the cubic tensor H
            bin_size = self.args.nei_thred_slstm * 2 / self.args.grid_size

            # convert the distance meter into bin index,
            # -1e-10 is to make sure the result bin index is smaller than grid_size
            corr_index_true = corr_index_true // bin_size
	    # may have bug using //
            # corr_index_true = (corr_index_true / bin_size).floor()
            # make the bin index from [-grid_size/2, grid_size/2-1] to [0,grid_size-1]
            corr_index_true += self.args.grid_size / 2
            grid_index = corr_index_true[gridmask].view(-1, 2).long()
            # Get the cubic tensor
            grid[gridmask[:, :, 0], grid_index[:, 0], grid_index[:, 1]] = 1
            grid=grid.view(numPeds,numPeds,-1)

        return grid
    def getSocialTensor(self, grid, hidden_states):
        '''
        Computes the social tensor.
            grid : Grid masks
            hidden_states : Hidden states of all peds
        Return: social_tensor, N*D
        Note:
            Imagining that there is only one grid, it is equal to sum the all valid hidden values,
        we can respectively sum features along hidden dimension.
        '''
        # Number of peds
        numPeds = grid.size()[0]

        #exchange the dimension,
        grid1 = grid.transpose(1,0).contiguous().view(numPeds, -1)
        social_tensor=torch.mm(torch.t(grid1),hidden_states).view(numPeds, -1)
        return social_tensor


