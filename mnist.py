import torch
import torch.ao.quantization
import torch.nn as nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader,Dataset
import torch.nn.functional as F
from torchvision.transforms import v2
from torchvision.transforms import functional as TF 
from tqdm import tqdm
from galore import GaLoREOptimizer

class MLP(nn.Module):
    def __init__(self,neurons):
        super(MLP,self).__init__()
        self.neurons = neurons
        self.linear_1 = nn.Linear(self.neurons[0],self.neurons[1],bias=False)
        self.linear_2 = nn.Linear(self.neurons[1],self.neurons[2],bias=False)
        self.linear_3 = nn.Linear(self.neurons[2],self.neurons[3],bias=False)
        self.linear_4 = nn.Linear(self.neurons[3],self.neurons[4],bias=False)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
    def forward(self,x):
        x = x.flatten(start_dim=1)
        x = self.act(self.linear_1(x))
        x = self.act(self.linear_2(x))
        x = self.act(self.linear_3(x))
        x = self.act(self.linear_4(x))
        x = self.softmax(x)
        return x

def train(dataloader,model,optimizer):
    global_step = 0
    progress_bar = tqdm(range(0,10*len(dataloader)),initial = global_step,desc="Steps: ")
    for epoch in range(10):
        for batch in dataloader:
            x = batch[0]
            y = batch[1].float().squeeze()
            y_pred = model(x).float()
            loss = F.mse_loss(y_pred.cpu(),y)
            y_pred = torch.argmax(y_pred.cpu(),dim=1)
            y = torch.argmax(y,dim=1)
            acc = sum(y_pred == y)
            logs = {"loss ":loss.detach().item(),"accuracy ":acc}
            progress_bar.set_postfix(**logs)
            global_step +=1
            progress_bar.update(1)
            loss.backward()
            import ipdb; ipdb.set_trace()
            optimizer.step()
            optimizer.zero_grad()

if __name__ == "__main__":
    
    dataset = MNIST('',train=True,download=True,transform=v2.Compose([v2.ToTensor()]),target_transform=v2.Compose([
                                 lambda x:torch.LongTensor([x]), # or just torch.tensor
                                 lambda x:F.one_hot(x,10)]))
   
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    import ipdb;ipdb.set_trace()
    model = MLP(neurons=[28*28,512,256,256,10])
    for name,module in model.named_modules():
        for name,param in module.named_parameters():
            if param.data.shape[0] < 32:
                param.requires_grad_ = False
            else:
                param.requires_grad_ = True

    trainable_params = [param for param in model.parameters() if param.requires_grad_]

    optimizer = GaLoREOptimizer(trainable_params, lr=0.001)
    train(dataloader,model,optimizer)

