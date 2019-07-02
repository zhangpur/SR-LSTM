'''
Author: Pu Zhang
Date: 2019/7/1
'''
from utils import *

import time
import torch.nn as nn
import yaml
class Processor():
    def __init__(self, args):
        self.args=args

        Dataloader=DataLoader_bytrajec2

        self.dataloader = Dataloader(args)
        model=import_class(args.model)
        self.net = model(args)
        self.set_optimazier()
        self.load_model()

        # self.load_weights_from_srlstm()
        # Uncomment to train the second SR layer

        if self.args.using_cuda:
            self.net=self.net.cuda()
        else:
            self.net=self.net.cpu()
        print(self.net)

        self.net_file = open(os.path.join(self.args.model_dir, 'net.txt'), 'a+')
        self.net_file.write(str(self.net))
        self.net_file.close()
        self.log_file_curve = open(os.path.join(self.args.model_dir, 'log_curve.txt'), 'a+')
    def parameters_update_seton(self):
        for p in self.net.parameters():
            p.requires_grad=True

    def parameters_update_seton_secondSR(self):
        for p in self.net.parameters():
            p.requires_grad=True

        self.net.cell.weight_ih.requires_grad = False
        self.net.cell.bias_ih.requires_grad = False
        self.net.cell.weight_hh.requires_grad = False
        self.net.cell.bias_hh.requires_grad = False
        self.net.inputLayer.weight.requires_grad = False
        self.net.inputLayer.bias.requires_grad = False
        self.net.outputLayer.weight.requires_grad = False
        self.net.outputLayer.bias.requires_grad = False
        self.net.gcn.ngate.MLP[0].bias.requires_grad = False
        self.net.gcn.ngate.MLP[0].weight.requires_grad = False
        self.net.gcn.relativeLayer.MLP[0].weight.requires_grad = False
        self.net.gcn.relativeLayer.MLP[0].bias.requires_grad = False
        self.net.gcn.W_nei.MLP[0].weight.requires_grad = False
        self.net.gcn.WAr.MLP[0].weight.requires_grad = False

    def load_weights_from_srlstm(self):
        if self.args.pretrain_load > 0:
            self.args.model_save_path = self.args.save_dir + '/' + self.args.pretrain_model + '/' + self.args.pretrain_model + '_' +\
                                   str(self.args.pretrain_load) + '.tar'
            if os.path.isfile(self.args.model_save_path):
                print('Loading checkpoint')
                checkpoint = torch.load(self.args.model_save_path)#,map_location={'cuda:0': 'cuda:0'})
                saved_weights=checkpoint['state_dict']

                self.net.inputLayer.weight.data=nn.Parameter(saved_weights['inputLayer.weight'])
                self.net.inputLayer.bias.data = nn.Parameter(saved_weights['inputLayer.bias'])

                self.net.cell.weight_ih.data=nn.Parameter(saved_weights['cell.weight_ih'])
                self.net.cell.bias_ih.data = nn.Parameter(saved_weights['cell.bias_ih'])
                self.net.cell.weight_hh.data=nn.Parameter(saved_weights['cell.weight_hh'])
                self.net.cell.bias_hh.data = nn.Parameter(saved_weights['cell.bias_hh'])
                self.net.gcn.ngate.MLP[0].weight.data=nn.Parameter(saved_weights['gcn.ngate.MLP.0.weight'])
                self.net.gcn.ngate.MLP[0].bias.data=nn.Parameter(saved_weights['gcn.ngate.MLP.0.bias'])
                self.net.gcn.WAr.MLP[0].weight.data=nn.Parameter(saved_weights['gcn.WAr.MLP.0.weight'])
                self.net.gcn.W_nei.MLP[0].weight.data=nn.Parameter(saved_weights['gcn.W_nei.MLP.0.weight'])
                self.net.gcn.relativeLayer.MLP[0].weight.data=nn.Parameter(saved_weights['gcn.relativeLayer.MLP.0.weight'])
                self.net.gcn.relativeLayer.MLP[0].bias.data=nn.Parameter(saved_weights['gcn.relativeLayer.MLP.0.bias'])
                self.net.outputLayer.weight.data=nn.Parameter(saved_weights['outputLayer.weight'])
                self.net.outputLayer.bias.data = nn.Parameter(saved_weights['outputLayer.bias'])

            self.parameters_update_seton_secondSR()

    def save_model(self,epoch):
        model_path= self.args.save_dir + '/' + self.args.train_model + '/' + self.args.train_model + '_' +\
                                   str(epoch) + '.tar'
        torch.save({
            'epoch': epoch,
            'state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, model_path)

    def load_model(self):
        if self.args.load_model > 0:
            self.args.model_save_path = self.args.save_dir + '/' + self.args.train_model + '/' + self.args.train_model + '_' + \
                                        str(self.args.load_model) + '.tar'
            if os.path.isfile(self.args.model_save_path):
                print('Loading checkpoint')
                checkpoint = torch.load(self.args.model_save_path)#,map_location={'cuda:1': 'cuda:2'})
                model_epoch = checkpoint['epoch']
                self.net.load_state_dict(checkpoint['state_dict'])
                print('Loaded checkpoint at epoch', model_epoch)

    def load_model_epoch(self,epoch):
        if epoch > 0:
            self.args.model_save_path = self.args.save_dir + '/' + self.args.train_model + '/' + self.args.train_model + '_' +\
                                   str(epoch) + '.tar'
            if os.path.isfile(self.args.model_save_path):
                print('Loading checkpoint')
                checkpoint = torch.load(self.args.model_save_path)#,map_location={'cuda:2': 'cuda:0'})
                model_epoch = checkpoint['epoch']
                self.net.load_state_dict(checkpoint['state_dict'])
                print('Loaded checkpoint at epoch', model_epoch)

    def set_optimazier(self):
        self.optimizer = torch.optim.Adam(self.net.parameters(),lr=self.args.learning_rate)
        self.criterion = nn.MSELoss(reduce=False)

    def playtest(self):
        print('Testing begin')
        test_error, test_final_error, _, _, _ = self.test_epoch(self.args.load_model)
        print('Set: {}, epoch: {:.5f},test_error: {:.5f} test_final_error: {:.5f}'.format(self.args.test_set,self.args.load_model,test_error,test_final_error))

    def playtrain(self):
        print('Training begin')
        find_result=[]
        test_error, test_final_error=0,0
        for epoch in range(self.args.num_epochs):

            train_loss,_=self.train_epoch(epoch)
            val_error,val_final,_,_,_= self.val_epoch(epoch)

            #test
            if epoch > self.args.start_test:
                test_error, test_final_error,_,look,_ = self.test_epoch(epoch)
                self.save_model(epoch)

            #log files
            self.log_file_curve.write(str(epoch) + ',' + str(train_loss) + ',' + str(
                val_error) + ',' + str(val_final) + ','+str(test_error) + ',' + str(test_final_error) + '\n')

            if epoch%10==0:
                self.log_file_curve.close()
                self.log_file_curve = open(os.path.join(self.args.model_dir, 'log_curve.txt'), 'a+')

            #console log
            print('----epoch {}, train_loss={:.5f}, valid_error={:.3f}, valid_final={:.3f},test_error={:.3f},valid_final={:.3f}'
                  .format(epoch, train_loss,val_error, val_final,test_error,test_final_error))

    def smaller(self,A,Aepoch,B,Bepoch):
        if A<B:
            return A,Aepoch
        else:
            return B,Bepoch
    def train_epoch(self,epoch):
        self.dataloader.reset_batch_pointer(set='train', valid=False)

        loss_epoch=0
        v1_sum,v2_sum,v3_sum=0,0,0

        for batch in range(self.dataloader.trainbatchnums):
            start = time.time()
            inputs,batch_id = self.dataloader.get_train_batch(batch)
            inputs = tuple([torch.Tensor(i) for i in inputs])
            inputs = tuple([i.cuda() for i in inputs])

            loss=torch.zeros(1).cuda()
            batch_abs, batch_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum = inputs
            inputs_fw = batch_abs[:-1], batch_norm[:-1], shift_value[:-1], seq_list[:-1], nei_list[:-1], nei_num[:-1], batch_pednum[:-1]

            self.net.zero_grad()

            outputs, _, _ ,look= self.net.forward(inputs_fw,iftest=False)

            v1, v2, v3=look
            v1_sum+=v1
            v2_sum+=v2
            v3_sum+=v3

            lossmask,num=getLossMask(outputs, seq_list[0],seq_list[1:],using_cuda=self.args.using_cuda)
            loss_o=torch.sum(self.criterion(outputs, batch_norm[1:,:,:2]),dim=2)

            loss += torch.sum(loss_o*lossmask)/num
            loss_epoch+=loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.clip)
            self.optimizer.step()
            end= time.time()
            if batch%self.args.show_step==0 and self.args.ifshow_detail:
                print(
                'train-{}/{} (epoch {}), train_loss = {:.5f}, time/batch = {:.5f}, value1={:.5f},value2={:.5f},value3={:.5f}'.format(batch,self.dataloader.trainbatchnums,
                                                                                epoch,
                                                                                loss.item(), end - start,v1,v2,v3))
        train_loss_epoch = loss_epoch / self.dataloader.trainbatchnums
        v1_sum=v1_sum / self.dataloader.trainbatchnums
        v2_sum=v2_sum / self.dataloader.trainbatchnums
        v3_sum=v3_sum / self.dataloader.trainbatchnums

        return train_loss_epoch,(v1_sum,v2_sum,v3_sum)

    def val_epoch(self, epoch):
        if self.dataloader.val_fraction==0:
            return 0,0,0
        self.dataloader.reset_batch_pointer(set='train', valid=True)
        error_epoch,final_error_epoch = 0,0
        error_cnt_epoch,final_error_cnt_epoch = 1e-5,1e-5

        v1_sum,v2_sum,v3_sum=0,0,0
        v1,v2,v3=0,0,0
        for batch in range(self.dataloader.valbatchnums):

            inputs,batch_id = self.dataloader.get_val_batch(batch)
            inputs = tuple([torch.Tensor(i) for i in inputs])
            inputs = tuple([i.cuda() for i in inputs])

            batch_abs, batch_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum = inputs
            inputs_fw = batch_abs[:-1], batch_norm[:-1], shift_value[:-1], seq_list[:-1], nei_list[:-1], nei_num[:-1], batch_pednum[:-1]
            forward = self.net.forward

            outputs_infer, _, _, look = forward(inputs_fw, iftest=True)
            lossmask, num = getLossMask(outputs_infer, seq_list[0], seq_list[1:], using_cuda=self.args.using_cuda)

            error, error_cnt, final_error, final_error_cnt, _ = L2forTest(outputs_infer, batch_norm[1:, :, :2],
                                                                          self.args.obs_length, lossmask)

            v1, v2, v3=look
            v1_sum+=v1
            v2_sum+=v2
            v3_sum+=v3

            error_epoch += error
            error_cnt_epoch += error_cnt
            final_error_epoch+=final_error
            final_error_cnt_epoch+=final_error_cnt

        v1_sum=v1_sum / self.dataloader.valbatchnums
        v2_sum=v2_sum / self.dataloader.valbatchnums
        v3_sum=v3_sum / self.dataloader.valbatchnums

        val_error=error_epoch/error_cnt_epoch
        final_error=final_error_epoch/final_error_cnt_epoch

        return val_error,final_error,0,0,(v1_sum,v2_sum,v3_sum)

    def test_epoch(self,epoch):
        self.dataloader.reset_batch_pointer(set='test')
        error_epoch,final_error_epoch,error_nl_epoch = 0,0,0
        error_cnt_epoch,final_error_cnt_epoch,error_nl_cnt_epoch= 1e-5,1e-5,1e-5

        value1_sum,value2_sum,value3_sum=0,0,0

        for batch in range(self.dataloader.testbatchnums):
            if batch%100==0:
                print('testing batch',batch,self.dataloader.testbatchnums)
            inputs, batch_id = self.dataloader.get_test_batch(batch)
            inputs = tuple([torch.Tensor(i) for i in inputs])

            if self.args.using_cuda:
                inputs = tuple([i.cuda() for i in inputs])

            batch_abs, batch_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum = inputs
            inputs_fw = batch_abs[:-1], batch_norm[:-1], shift_value[:-1], seq_list[:-1], nei_list[:-1], nei_num[:-1], batch_pednum[:-1]
            forward = self.net.forward
            self.net.zero_grad()
            outputs_infer, _, _, look = forward(inputs_fw, iftest=True)

            value1, value2, value3 = look
            value1_sum += value1
            value2_sum += value2
            value3_sum += value3

            lossmask, num = getLossMask(outputs_infer, seq_list[0], seq_list[1:], using_cuda=self.args.using_cuda)
            error, error_cnt, final_error, final_error_cnt, error_nl,error_nl_cnt,_ = L2forTest_nl(outputs_infer, batch_norm[1:, :, :2],
                                                                              self.args.obs_length, lossmask,seq_list[1:],nl_thred=0)

            error_epoch += error
            error_cnt_epoch += error_cnt
            final_error_epoch += final_error
            final_error_cnt_epoch += final_error_cnt
            error_nl_epoch+=error_nl
            error_nl_cnt_epoch+=error_nl_cnt

        value1_sum=value1_sum / self.dataloader.testbatchnums
        value2_sum=value2_sum / self.dataloader.testbatchnums
        value3_sum=value3_sum / self.dataloader.testbatchnums
        return error_epoch / error_cnt_epoch, final_error_epoch / final_error_cnt_epoch, \
              error_nl_epoch/error_nl_cnt_epoch, (value1_sum,value2_sum,value3_sum),error_cnt_epoch


