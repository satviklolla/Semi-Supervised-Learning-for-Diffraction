from torch import nn
import torch
from transformers.modeling_outputs import SequenceClassifierOutput

class Generator(nn.Module):
    def __init__(self, ngf, ngf2):
        super(Generator, self).__init__()
        self.ngf = ngf
        self.ngf2 = ngf2
        self.main = nn.Sequential(
            nn.ConvTranspose1d(1, self.ngf, 17, 2, 0, bias=False),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(self.ngf, eps=1e-4),
            nn.ConvTranspose1d(self.ngf, self.ngf, 15, 1, 0, bias=False),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(self.ngf, eps=1e-4),
            nn.ConvTranspose1d(self.ngf, self.ngf, 14, 1, 0, bias=False),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(self.ngf, eps=1e-4),
            nn.ConvTranspose1d(self.ngf, self.ngf, 18, 3, bias=False),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(self.ngf, eps=1e-4),
            nn.ConvTranspose1d(self.ngf, self.ngf2, 13, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(self.ngf2, eps=1e-4),
            nn.ConvTranspose1d( self.ngf2, self.ngf2, 11, 4,bias=False),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(self.ngf2, eps=1e-4),            
            nn.ConvTranspose1d( self.ngf2, self.ngf2, 9,1, bias=False),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(self.ngf2, eps=1e-4),            
            nn.ConvTranspose1d( self.ngf2, self.ngf2, 9,1, bias=False),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(self.ngf2, eps=1e-4),   
            nn.ConvTranspose1d( self.ngf2, 1, 7, 1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, noise):
        logits=self.main(noise)
        loss=None
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )
    
from torch import nn
import torch
from transformers.modeling_outputs import SequenceClassifierOutput

class Generator2(nn.Module):
    def __init__(self, ngf, ngf2):
        super(Generator2, self).__init__()
        self.ngf = ngf
        self.ngf2 = ngf2
        self.main = nn.Sequential(
            nn.ConvTranspose1d(1, self.ngf, 17, 2, 0, bias=False),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(self.ngf, eps=1e-4),
            nn.ConvTranspose1d(self.ngf, self.ngf, 15, 1, 0, bias=False),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(self.ngf, eps=1e-4),
            nn.ConvTranspose1d(self.ngf, self.ngf, 14, 1, 0, bias=False),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(self.ngf, eps=1e-4),
            nn.ConvTranspose1d(self.ngf, self.ngf, 18, 3, bias=False),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(self.ngf, eps=1e-4),
            nn.ConvTranspose1d(self.ngf, self.ngf2, 13, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(self.ngf2, eps=1e-4),
            nn.ConvTranspose1d( self.ngf2, self.ngf2, 11, 4,bias=False),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(self.ngf2, eps=1e-4),            
            nn.ConvTranspose1d( self.ngf2, self.ngf2, 9,1, bias=False),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(self.ngf2, eps=1e-4),            
            nn.ConvTranspose1d( self.ngf2, self.ngf2, 9,1, bias=False),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(self.ngf2, eps=1e-4),   
            nn.ConvTranspose1d( self.ngf2, 1, 8, 1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, noise):
        logits=self.main(noise)
        loss=None
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )