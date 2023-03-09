from xml.dom.minidom import parse
xml_doc = r"./data/1.xml"
# xml_doc = r"D:\pycharmprojects\dataset\VOCdevkit\VOC2012\Annotations/2007_000032.xml"

dom = parse(xml_doc)
root = dom.documentElement

img_name = root.getElementsByTagName("path")[0].childNodes[0].data

img_size= root.getElementsByTagName("size")[0]
img_w = img_size.getElementsByTagName("width")[0].childNodes[0].data
img_h = img_size.getElementsByTagName("height")[0].childNodes[0].data
img_c = img_size.getElementsByTagName("depth")[0].childNodes[0].data
print(img_name)
print(img_w,img_h,img_c)

item = root.getElementsByTagName("item")
for box in item:
    cls_name = box.getElementsByTagName("name")[0].childNodes[0].data
    x1 = int(box.getElementsByTagName("xmin")[0].childNodes[0].data)
    y1 = int(box.getElementsByTagName("ymin")[0].childNodes[0].data)
    x2 = int(box.getElementsByTagName("xmax")[0].childNodes[0].data)
    y2 = int(box.getElementsByTagName("ymax")[0].childNodes[0].data)
    print(cls_name,x1,y1,x2,y2)





