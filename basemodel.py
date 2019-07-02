'''
Basic Models
Author: Pu Zhang
Date: 2019/7/1
'''
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.rnn import RNNCellBase
from torch.nn.parameter import Parameter
class MakeMLP(nn.Module):
    def __init__(self,args,layer_num,layer_name,input_num,hunit_num,output_num,active_fun,drop_ratio,ifbias,iflastac=True):
        super(MakeMLP, self).__init__()
        layers=[]
        self.args=args
        if iflastac:
            lastac=active_fun
            lastdrop=drop_ratio
        else:
            lastac=''
            lastdrop=0
        if layer_num>1:
            self.addLayer(layers, input_num, hunit_num, ifbias, active_fun, drop_ratio)
            for i in range(layer_num-2):
                self.addLayer(layers, hunit_num, hunit_num, ifbias, active_fun, drop_ratio)
            self.addLayer(layers, hunit_num, output_num, ifbias, lastac, lastdrop)
        else:
            self.addLayer(layers, input_num, output_num, ifbias, lastac, lastdrop)
        self.MLP=nn.Sequential(*layers)
        if layer_name=='rel':
            self.MLP.apply(self.init_weights_rel)
        elif layer_name=='nei':
            self.MLP.apply(self.init_weights_nei)
        elif layer_name=='attR':
            self.MLP.apply(self.init_weights_attr)
        elif layer_name=='ngate':
            self.MLP.apply(self.init_weights_ngate)

    def addLayer(self,layers,input_num,output_num,ifbias,active_fun,drop_ratio):
        layers.append(nn.Linear(input_num, output_num, bias=ifbias))
        if active_fun == 'sig':
            Active_fun = nn.Sigmoid
            layers.append(Active_fun())
        elif active_fun== 'relu':
            Active_fun = nn.ReLU
            layers.append(Active_fun())
        elif active_fun == 'lrelu':
            Active_fun = nn.LeakyReLU
            layers.append(Active_fun(0.1))
        elif active_fun == 'tanh':
            Active_fun = nn.Tanh
            layers.append(Active_fun())
        if drop_ratio!=0:
            layers.append(nn.Dropout(drop_ratio))
        return layers
    def init_weights(self,m):
        if type(m)==nn.Linear:
            nn.init.xavier_uniform(m.weight)
            try:
                nn.init.constant(m.bias, 0)
            except:
                pass

    def init_weights_ngate(self,m):
        if type(m)==nn.Linear:
            nn.init.normal(m.weight, std=0.005)

            if self.args.ifbias_gate:
                nn.init.constant(m.bias,0)
    def init_weights_nei(self,m):
        if type(m)==nn.Linear:

            nn.init.orthogonal(m.weight,gain=self.args.nei_std)
            if self.args.ifbias_nei:
                nn.init.constant(m.bias,0)

    def init_weights_attr(self,m):
        if type(m)==nn.Linear:
            #nn.init.normal(m.weight,mean=0,std=self.args.WAq_std)
            nn.init.xavier_uniform(m.weight)
            try:
                nn.init.constant(m.bias,0)
            except:
                pass

    def init_weights_rel(self,m):
        if type(m)==nn.Linear:
            nn.init.normal_(m.weight,mean=0,std=self.args.rela_std)
            #nn.init.xavier_uniform(m.weight)
            #m.weight.data+=0.1
            if self.args.ifbias_nei:
                nn.init.constant(m.bias,0)

class LSTMCell(RNNCellBase):
    '''
    Copied from torch.nn
    '''
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__(input_size, hidden_size,bias=True,num_chunks=4)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(4 * hidden_size, hidden_size))

        self.bias_ih = Parameter(torch.Tensor(4 * hidden_size))
        self.bias_hh = Parameter(torch.Tensor(4 * hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    def forward(self, input, hx,update_mode=''):

        hx, cx = hx
        gates = F.linear(input, self.weight_ih, self.bias_ih) + F.linear(hx, self.weight_hh, self.bias_hh)
        ingate, forgetgate, cellgate, outgate_ = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate_ )

        cy = forgetgate * cx +ingate * cellgate
        hy = outgate * F.tanh(cy)

        return outgate,hy, cy

class GCN(nn.Module):
    def __init__(self,args,r_embed_size,output_size):
        super(GCN, self).__init__()
        self.args=args
        self.relu=nn.ReLU()
        self.R=r_embed_size
        self.D=output_size

        self.D1 = self.args.hidden_dot_size

        # Motion gate
        self.ngate = MakeMLP(self.args, 1, 'ngate', self.R+self.D + self.D, self.args.nei_hidden_size,
                                 self.D, 'sig', self.args.nei_drop, ifbias=self.args.ifbias_gate)

        # Relative spatial embedding layer
        self.relativeLayer = MakeMLP(self.args, self.args.rela_layers, 'rel', self.args.rela_input,
                                     self.args.rela_hidden_size,
                                     self.R, self.args.rela_ac, self.args.rela_drop, ifbias=True, iflastac=True)
        # Message passing transform
        self.W_nei = MakeMLP(self.args, self.args.nei_layers, 'nei', self.D, self.args.nei_hidden_size,
                         self.D, self.args.nei_ac, self.args.nei_drop, ifbias=self.args.ifbias_nei,iflastac=False)

        tmp=self.R+self.D*2

        # Attention
        self.WAr = MakeMLP(self.args,1,'attR',tmp,self.D1,1, self.args.WAr_ac, drop_ratio=0, ifbias=self.args.ifbias_WAr)

        #not used
        self.WAr1 = MakeMLP(self.args,1,'attR',tmp,self.D1,self.args.hidden_dot_size, '', drop_ratio=0, ifbias=False)
        self.WAr2 = MakeMLP(self.args, 1, 'attR2', self.D1, self.args.hidden_dot_size, 1, '', drop_ratio=0,ifbias=False)


    def forward(self, corr_index,nei_index,nei_num,lstm_state,W):
        '''
        States Refinement process.
        Params:
            corr_index: relative coords of each pedestrian pair
            nei_index: neighbor exsists flag
            nei_num: neighbor number
            lstm_state: output states of LSTM cell
            W: message passing weight, namely self.W_nei when train one SR layer
        Return:
            Refined states
            Tracked variable
        '''
        outgate, self_h, self_c = lstm_state

        # If you want to track some variables
        value1,value2,value3=torch.zeros(1),torch.zeros(1),torch.zeros(1)

        self.N = corr_index.shape[0]
        nei_inputs = self_h.repeat(self.N, 1)

        nei_index_t = nei_index.view((-1))

        corr_t=corr_index.view((self.N * self.N, -1))

        if corr_t[nei_index_t > 0].shape[0] == 0:
            # Ignore when no neighbor in this batch
            return lstm_state, (0, 0, 0),(0,0)

        r_t = self.relativeLayer.MLP(corr_t[nei_index_t > 0])
        inputs_part = nei_inputs[nei_index_t > 0]
        hi_t = nei_inputs.view((self.N, self.N, self.D)).permute(1, 0, 2).contiguous().view(-1, self.D)

        tmp = torch.cat((r_t, hi_t[nei_index_t > 0],inputs_part), 1)

        # Motion Gate
        nGate = self.ngate.MLP(tmp)
        inputs_part = inputs_part * nGate

        # Attention
        Pos_t = torch.full((self.N * self.N,1), 0, device=torch.device("cuda")).view(-1)
        tt = self.WAr.MLP(torch.cat((r_t, hi_t[nei_index_t > 0], nei_inputs[nei_index_t > 0]), 1)).view((-1))
        #have bug if there's any zero value in tt

        Pos_t[nei_index_t > 0] = tt
        Pos = Pos_t.view((self.N, self.N))
        Pos[Pos == 0] = -np.Inf
        Pos = torch.softmax(Pos, dim=1)
        Pos_t = Pos.view(-1)

        # Message Passing
        H = torch.full((self.N * self.N, self.D), 0, device=torch.device("cuda"))
        H[nei_index_t > 0] = inputs_part
        H[nei_index_t > 0] = H[nei_index_t > 0] * Pos_t[nei_index_t > 0].repeat(self.D, 1).transpose(0, 1)
        H = H.view(self.N, self.N, -1)
        H_sum = W.MLP(torch.sum(H, 1))

        # Update Cell states
        C = H_sum + self_c
        H = outgate * F.tanh(C)

        if self.args.ifdebug:
            value1 = torch.norm(H_sum[nei_num > 0]*self.args.nei_ratio ) / torch.norm(self_c[nei_num > 0])
            return (outgate, H, C), (value1.item(), value2.item(),value3.item())
        else:
            return (outgate, H, C), (0, 0, 0)
