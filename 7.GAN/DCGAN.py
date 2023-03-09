import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms,datasets
from torchvision.utils import save_image
import os

class D_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(1,128,5,2,2),
            nn.LeakyReLU(0.2),
        )#128*14*14
        self.conv2=nn.Sequential(
            nn.Conv2d(128,256,5,2,2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )#256*7*7
        self.conv3=nn.Sequential(
            nn.Conv2d(256,512,5,2,1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )#512*3*3
        self.conv4=nn.Sequential(
            nn.Conv2d(512,1,3,1,0),
            nn.Sigmoid()
        )#1*1*1
    def forward(self,x):
        y=self.conv1(x)
        y=self.conv2(y)
        y=self.conv3(y)
        y=self.conv4(y)
        return y

    #初始化判别器的权重
    def d_weight_init(self,m):
        if isinstance(m,nn.Conv2d):
            nn.init.normal_(m.weight,mean=0.0,std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias,0.0)

class G_Net(nn.Module):
    def __init__(self):
        super(G_Net, self).__init__()
        self.conv_transpose1=nn.Sequential(
            nn.ConvTranspose2d(128,512,3,1,0),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )#512,3,3
        self.conv_transpose2=nn.Sequential(
            nn.ConvTranspose2d(512,256,5,2,1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )#256,7,7
        self.conv_transpose3=nn.Sequential(
            nn.ConvTranspose2d(256,128,5,2,2,1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )#128,14,14
        self.conv_transpose4=nn.Sequential(
            nn.ConvTranspose2d(128,1,5,2,2,1),
            nn.Tanh()
        )#128,14,14
    def forward(self,x):
        y=self.conv_transpose1(x)
        y=self.conv_transpose2(y)
        y=self.conv_transpose3(y)
        y=self.conv_transpose4(y)
        return y
#初始化判别器的权重
    def g_weight_init(self,m):
        if isinstance(m,nn.Conv2d):
            nn.init.normal_(m.weight,mean=0.0,std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias,0.0)


if __name__ == '__main__':
    num_epoch=10
    if not os.path.exists("./dcgan_img"):
        os.mkdir("./dcgan_img")
    if not os.path.exists("./dcgan_params"):
        os.mkdir("./dcgan_params")
    transf=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5,],[0.5,])
    ])
    mnist=datasets.MNIST("./data",transform=transf,download=True)
    loader=DataLoader(mnist,100,shuffle=True)

    if torch.cuda.is_available():
        device=torch.device("cuda")
    else:
        device=torch.device("cpu")

    d_net=D_Net().to(device)
    g_net=G_Net().to(device)

    d_weight_file="r./dcgan_params/d_net.pth"
    g_weight_file="r./dcgan_params/g_net.pth"

    if os.path.exists(d_weight_file):
        d_net.load_state_dict(torch.load(d_weight_file))
        print("记载判别器保存参数成功")
    else:
        d_net.apply(d_net.d_weight_init)
        print("记载判别器随机参数成功")

    if os.path.exists(g_weight_file):
        g_net.load_state_dict(torch.load(g_weight_file))
        print("记载生成器保存参数成功")
    else:
        g_net.apply(g_net.g_weight_init)
        print("记载生成器随机参数成功")



    loss_fn=nn.BCELoss()
    d_opt=torch.optim.Adam(d_net.parameters(),lr=0.0002,betas=(0.5,0.999))
    g_opt=torch.optim.Adam(g_net.parameters(),lr=0.0002,betas=(0.5,0.999))

    for epoch in range(num_epoch):
        for i ,(x,y) in enumerate(loader):
            #训练判别器
            real_img=x.to(device)
            real_label=torch.ones(x.size(0),1,1,1).to(device)
            fake_label=torch.zeros(x.size(0),1,1,1).to(device)
            real_out=d_net(real_img)#判别器对真数据的判断
            d_real_loss=loss_fn(real_out,real_label)#判别器对真数据的损失，标签为1
            z=torch.randn(x.size(0),128,1,1).to(device)
            fake_img=g_net(z)
            fake_out=d_net(fake_img)#判别器对假数据的判断
            d_fake_loss=loss_fn(fake_out,fake_label)#判别器对假数据的损失，标签为0
            d_loss=d_real_loss+d_fake_loss
            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            #训练生成器
            z=torch.randn(x.size(0),128,1,1).to(device)
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
                save_image(real_image,"./dcgan_img/{}-real_img.jpg"
                           .format(epoch+1),nrow=10,normalize=True,scale_each=True)
                fake_image=fake_img.cpu().data.reshape([-1,1,28,28])
                save_image(fake_image,"./dcgan_img/{}-fake_img.jpg"
                           .format(epoch+1),nrow=10,normalize=True,scale_each=True)
                torch.save(d_net.state_dict(),"./dcgan_params/d_net.pth")
                torch.save(g_net.state_dict(),"./dcgan_params/g_net.pth")




