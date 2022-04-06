
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    # Modification of LSTNet for multistep forcasting
    def __init__(self, args):
        super(Model, self).__init__()
        # self.use_cuda = args.cuda
        self.P = args.seq_len
        self.m = args.enc_in
        self.pred_len = args.pred_len
        self.hidR = args.d_model
        self.hidC = args.d_model
        self.hidS = args.d_model_skip
        self.Ck = args.cnn_kernel_size
        self.skip = args.rnn_skip
        self.pt = int((self.P - self.Ck)/self.skip)
        self.hw = args.highway_window
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.m))
        self.GRU1 = nn.GRU(self.hidC, self.hidR)
        self.dropout = nn.Dropout(p=args.dropout)
        if self.skip > 0:
            self.GRUskip = nn.GRU(self.hidC, self.hidS)
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.pred_len * self.m)
        else:
            self.linear1 = nn.Linear(self.hidR, self.pred_len * self.m)
        if self.hw > 0:
            self.highway = nn.Conv1d(self.hw, self.pred_len, kernel_size=1)
        
        if args.output_fun == 'no':
            self.output = None
        elif args.output_fun == 'sigmoid':
            self.output = F.sigmoid
        elif args.output_fun == 'tanh':
            self.output = F.tanh
        else:
            raise NotImplementedError

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        x = x_enc
        batch_size = x.size(0)
        
        #CNN
        c = x.view(-1, 1, self.P, self.m)
        c = F.relu(self.conv1(c))
        c = self.dropout(c)
        c = torch.squeeze(c, 3)
        
        # RNN 
        r = c.permute(2, 0, 1).contiguous()
        _, r = self.GRU1(r)
        r = self.dropout(torch.squeeze(r,0))

        #skip-rnn
        if self.skip > 0:
            s = c[:,:,-self.pt * self.skip:].contiguous()
            s = s.view(batch_size, self.hidC, self.pt, self.skip)
            s = s.permute(2,0,3,1).contiguous()
            s = s.view(self.pt, batch_size * self.skip, self.hidC)
            _, s = self.GRUskip(s)
            s = s.view(batch_size, self.skip * self.hidS)
            s = self.dropout(s)
            r = torch.cat((r,s),1)
        
        res = self.linear1(r).view(-1, self.pred_len, self.m)
        
        #highway
        if self.hw > 0:
            z = x[:, -self.hw:, :]
            z = self.highway(z)
            res = res + z
            
        if self.output:
            res = self.output(res)
        return res
