import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms,datasets
from torchvision.utils import save_image
import os

class D_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.dnet=nn.Sequential(
            nn.Linear(784,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        return self.dnet(x)

class G_Net(nn.Module):
    def __init__(self):
        super(G_Net, self).__init__()
        self.gnet=nn.Sequential(
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Linear(512,784)
        )
    def forward(self,x):
        return self.gnet(x)

if __name__ == '__main__':
    num_epoch=10
    if not os.path.exists("./gan_img"):
        os.mkdir("./gan_img")
    if not os.path.exists("./gan_params"):
        os.mkdir("./gan_params")

    mnist=datasets.MNIST("./data",transform=transforms.ToTensor(),download=True)
    loader=DataLoader(mnist,100,shuffle=True)

    if torch.cuda.is_available():
        device=torch.device("cuda")
    else:
        device=torch.device("cpu")

    d_net=D_Net().to(device)
    g_net=G_Net().to(device)

    loss_fn=nn.BCELoss()
    d_opt=torch.optim.Adam(d_net.parameters())
    g_opt=torch.optim.Adam(g_net.parameters())

    for epoch in range(num_epoch):
        for i ,(x,y) in enumerate(loader):
            #训练判别器
            real_img=x.reshape(x.size(0),-1).to(device)
            real_label=torch.ones(x.size(0),1).to(device)
            fake_label=torch.zeros(x.size(0),1).to(device)
            real_out=d_net(real_img)#判别器对真数据的判断
            d_real_loss=loss_fn(real_out,real_label)#判别器对真数据的损失，标签为1
            z=torch.randn(x.size(0),128).to(device)
            fake_img=g_net(z)
            fake_out=d_net(fake_img)#判别器对假数据的判断
            d_fake_loss=loss_fn(fake_out,fake_label)#判别器对假数据的损失，标签为0
            d_loss=d_real_loss+d_fake_loss
            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            #训练生成器
            z=torch.randn(x.size(0),128).to(device)
            fake_img=g_net(z)
            fake_out=d_net(fake_img)
            #生成器损失：让生成器的输出能够看到真实数据的分布，所以这时候标签是1
            g_loss=loss_fn(fake_out,real_label)
            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()

            if i%100==0:
                print("Epoch[{}/{}],d_loss:{:.3f},g_loss:{:.3f},d_real_out:{:.3f},d_fake_out:{:.3f}"
                      .format(epoch,num_epoch,d_loss,g_loss,real_out.data.mean(),fake_out.data.mean()))

                real_image=real_img.cpu().data.reshape([-1,1,28,28])
                save_image(real_image,"./gan_img/{}-real_img.jpg"
                           .format(epoch+1),nrow=10,normalize=True,scale_each=True)
                fake_image=fake_img.cpu().data.reshape([-1,1,28,28])
                save_image(fake_image,"./gan_img/{}-fake_img.jpg"
                           .format(epoch+1),nrow=10,normalize=True,scale_each=True)
                torch.save(d_net.state_dict(),"./gan_params/d_net.pth")
                torch.save(g_net.state_dict(),"./gan_params/g_net.pth")




