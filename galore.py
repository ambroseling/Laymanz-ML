# Trainng code for the GaLoRE Optimizer

from torch.optim import Optimizer
import torch
import math

class GaLoREOptimizer(Optimizer):
    def  __init__(self,params,lr = 1e-6,betas = (0.99,0.9),eps=1e-6,weight_decay=0.0,rank=32,subspace_freq=10,lora_scale=1.0):
        defaults = dict(lr=lr,betas=betas,weight_decay=weight_decay,eps=eps, )
        self.subspace_freq = subspace_freq # corresponds to T, subspace change frequency
        self.lora_rank = rank # corresponds to LoRA rank
        self.scaling_factor = lora_scale
        super(GaLoREOptimizer,self).__init__(params,defaults=defaults)
    def step(self,closure=None):
        #Iterature through all the parameter 

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                import ipdb;ipdb.set_trace()
                grad = p.grad.data
                #p.data.shape (512,784)
                m = p.grad.data.shape[0] # 512
                n = p.grad.data.shape[-1] # 768
                r = self.lora_rank
                state = self.state[p]
                step_size = group['lr']
                # state is a dictionary that holds all the optimizer configurations for each parameter
                if len(state) ==0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros(r,n)
                    state['exp_avg_sqr'] = torch.zeros(r,n)
                    state['projection'] = torch.zeros(m,r)
                if state['step'] % self.subspace_freq == 0:
                    U,_,_ = torch.svd(grad)
                    #                      m x r
                    state['projection'] = U[:,:r] #state['projection'].shape (512,32)
                else:
                    pass

                # Project the gradient matrix to low rank (compact space)
                r_t = state['projection'].T @ grad # (32,512) * (512,784) = (32, 784) = r x n
                # Exponential moving average of **low-rank projection** of gradient values
                exp_avg = state['exp_avg'] #first moment estimate
                # Exponential moving average of the square of the **low-rank projection** of gradient values
                exp_avg_sqr = state['exp_avg_sqr'] # second moment estimate
                beta1,beta2 = group['betas'] # get betas
                exp_avg.mul_(beta1).add_((1-beta1),r_t) #update biased first moment estimate
                exp_avg_sqr.mul(beta2).addcmul_((1-beta2),r_t,r_t) # update biased second moment estimate
                denom = exp_avg_sqr.sqrt().add_(group['eps']) # denom.shape (32,784)

                # If there is bias correction
                if 'bias_correction' in group:
                    bias_corrected_first_moment = 1 - beta1 ** state['step']
                    bias_corrected_second_moment = 1 - beta2 ** state['step']
                    step_size = step_size * math.sqrt(bias_corrected_second_moment) / bias_corrected_first_moment
                n_t = exp_avg / denom 

                # Project low-rank graident matriz back t to original vectior subspace
                #                               m x r           r x n
                import ipdb;ipdb.set_trace()
                g_t = self.scaling_factor * state['projection'] @ n_t
                state['step'] += 1
                # Weight update
                p.data.add_(-step_size*g_t)