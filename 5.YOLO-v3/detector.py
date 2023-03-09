from model import *
import cfg
import torch
import numpy as np
import PIL.Image as pimg
import PIL.ImageDraw as draw
import tool


class Detector(torch.nn.Module):

    def __init__(self, save_path):
        super(Detector, self).__init__()

        self.net = MainNet().cuda()
        self.net.load_state_dict(torch.load(save_path))
        self.net.eval()

    def forward(self, input, thresh, anchors):
        output_13, output_26, output_52 = self.net(input)
        idxs_13, vecs_13 = self._filter(output_13, thresh)
        boxes_13 = self._parse(idxs_13, vecs_13, 32, anchors[13])
        idxs_26, vecs_26 = self._filter(output_26, thresh)
        boxes_26 = self._parse(idxs_26, vecs_26, 16, anchors[26])
        idxs_52, vecs_52 = self._filter(output_52, thresh)
        boxes_52 = self._parse(idxs_52, vecs_52, 8, anchors[52])
        return torch.cat([boxes_13, boxes_26, boxes_52], dim=0)

    def _filter(self, output, thresh):
        output = output.permute(0, 2, 3, 1)
        output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)
        mask = torch.sigmoid(output[..., 0]) > thresh
        idxs = mask.nonzero()
        vecs = output[mask]
        print(idxs.shape)
        print(vecs.shape)
        return idxs, vecs

    def _parse(self, idxs, vecs, t, anchors):
        if len(idxs) == 0:
            return torch.randn(0, 6).cuda()
        else:
            anchors = torch.tensor(anchors, dtype=torch.float32).cuda()
            a = idxs[:, 3]  # 建议框:3

            # "压缩置信度值到0-1之间"
            confidence = torch.sigmoid(vecs[:, 0])
            _classify = vecs[:, 5:]
            classify = torch.argmax(_classify, dim=1).float()

            cy = (idxs[:, 1].float() + torch.sigmoid(vecs[:, 2])) * t
            cx = (idxs[:, 2].float() + torch.sigmoid(vecs[:, 1])) * t
            w = anchors[a, 0] * torch.exp(vecs[:, 3])
            h = anchors[a, 1] * torch.exp(vecs[:, 4])
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = x1 + w
            y2 = y1 + h
            # print(confidence)
            out = torch.stack([confidence, x1, y1, x2, y2, classify], dim=1)
            return out


if __name__ == '__main__':
    save_path = "models/net_yolo.pth"
    detector = Detector(save_path)
    # y = detector(torch.randn(3, 3, 416, 416), 0.3, cfg.ANCHORS_GROUP)
    # print(y.shape)

    img1 = pimg.open(r'data\images\1.jpg')
    img = img1.convert('RGB')
    img = np.array(img) / 255
    img = torch.Tensor(img)
    img = img.unsqueeze(0)
    img = img.permute(0, 3, 1, 2)
    img = img.cuda()

    out_value = detector(img, 0.3, cfg.ANCHORS_GROUP)
    boxes = []

    for j in range(10):
        classify_mask = (out_value[..., -1] == j)
        _boxes = out_value[classify_mask]
        boxes.append(tool.nms(_boxes))

    for box in boxes:
        try:
            img_draw = draw.ImageDraw(img1)
            c, x1, y1, x2, y2 = box[0, 0:5]
            print(c, x1, y1, x2, y2)
            img_draw.rectangle((x1, y1, x2, y2))
        except:
            continue
    img1.show()
