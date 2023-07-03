# Fixed filter noise cancellation by Sample

import torch


class Fixed_filter_controller():
    def __init__(self, Filter_vector, fs):
        self.Filter_vector = torch.from_numpy(Filter_vector).type(torch.float)# torch.Size([Xseconds, 1024])
        Len = self.Filter_vector.shape[1]
        self.fs = fs
        self.Xd = torch.zeros(1, Len, dtype=torch.float)
        self.Current_Filter = torch.zeros(1, Len, dtype=torch.float)
    
    def noise_cancellation(self, Dis, Fx):
        Erro = torch.zeros(Dis.shape[0])
        j = 0
        for ii, dis in enumerate(Dis):
            self.Xd = torch.roll(self.Xd,1,1)
            self.Xd[0,0] = Fx[ii] # Fx[ii]: fixed-x signal
            yt = self.Current_Filter @ self.Xd.t()
            e = dis - yt
            Erro[ii] = e.item()
            if (ii + 1) % self.fs == 0:
                self.Current_Filter = self.Filter_vector[j]
                j += 1
        return Erro