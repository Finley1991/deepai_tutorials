from PIL import Image
from PIL import ImageDraw
import os
import time
img_path = r"D:\dataset\CelebA\Img\img_celeba.7z\img_celeba"
anno_path = r"D:\dataset\CelebA\Anno\list_bbox_celeba.txt"
for i, line in enumerate(open(anno_path)):
    if i < 2:
        continue
    else:
        strs = line.split()
        filename=strs[0].strip()
        x1=int(strs[1].strip())
        y1=int(strs[2].strip())
        x2=int(strs[3].strip())+x1
        y2=int(strs[4].strip())+y1
        img=Image.open(os.path.join(img_path,filename))
        imgDraw = ImageDraw.Draw(img)
        print(x1,y1,x2,y2)
        imgDraw.rectangle((x1,y1,x2,y2),outline="red",width=3)
        img.show()
        time.sleep(2)
# img = Image.open(os.path.join(img_path,"000120.jpg"))
# imgDraw = ImageDraw.Draw(img)
# # 70   8  68 119
# imgDraw.rectangle((70, 8, 70 + 68, 8 + 119),outline="red")
# img.show()
