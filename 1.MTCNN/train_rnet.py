import nets
import train
import os
if __name__ == '__main__':
    net = nets.RNet()
    if not os.path.exists("./param"):
        os.makedirs("./param")
    trainer = train.Trainer(net, './param/r_net.pth', r"E:\datasets\train\24")
    trainer.train(0.6)
