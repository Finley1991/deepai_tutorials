import torch
def dice(pred,target):
    p=pred.reshape(pred.size(0),-1)
    g=target.reshape(target.shape[0],-1)

    d=(2*(p*g).sum()+1)/(p.sum()+g.sum()+1)
    return d

if __name__ == '__main__':
    output=torch.rand(10,3,224,224)
    label=torch.ones(10,3,224*224)
    print(dice(output,label))