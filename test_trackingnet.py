# -*- coding: utf-8 -*-

#%%
import torch
from torch.autograd import Variable as V
import numpy as np
import cv2
import math
from axis import xywh_to_x1y1x2y2
#from PIL import Image
import os
from torchvision.transforms import functional as F
from axis import x1y1x2y2_to_xywh, point_center_crop, resize, x1y1wh_to_xywh
from PIL import Image
from SRPN import SiameseRPN

#%%
#interval = 1
#
#imgdir = os.listdir('./TrackigNet/')
#number = []
#for item in imgdir:
#    number.append(len(os.listdir('./TrackingNet/'+item+'/frames/')))
#
#number = [i-interval for i in number]
#
#sum1 = [number[0]]
#for a in range(1, len(number)):
#    sum1.append(sum1[a-1]+number[a])
#%%
def _transform(img, gtbox, area, size):
    img, pcc = point_center_crop(img, gtbox, area)
    img, ratio = resize(img, size)
    img = F.to_tensor(img)
#    img = F.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    img = img.unsqueeze(0)
    return img, pcc, ratio
#%%
def transform2(img, gtbox, area, size):
        img, _ = point_center_crop(img, gtbox, area)
        img, _ = resize(img, size)

        return img
#%%
def IOU(a, b):
        sa = (a[2] - a[0]) * (a[3] - a[1]) 
        sb = (b[2] - b[0]) * (b[3] - b[1])
        w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        area = w * h 
        return area / (sa + sb - area)
    
#%%
def show_output(root_dir, model, pth_file, use_gpu):

    print('load model params...')
    model.load_state_dict(torch.load(pth_file))
    model = model.train(False)  # Set model to evaluate mode
    
    print('test...')
    
    last_box = os.listdir(root_dir+'/anno')[0]
    with open(root_dir +'/anno/'+ last_box) as f:
        last_box = f.read().split(',')
    last_box = [float(i) for i in last_box]
    last_box = x1y1wh_to_xywh(last_box)
#    last_box = x1y1x2y2_to_xywh(last_box)
    

    template = os.listdir(root_dir+'/frames/')[0]
    template = Image.open(root_dir +'/frames/'+ template)
    
    template, _, _ = _transform(template, last_box, 1, 127)
    """"""
#    template = template.squeeze()
#    template = template.numpy()
#    import cv2
#    import numpy as np
#    import math
#    from axis import xywh_to_x1y1x2y2
#    template = np.transpose(template,(1,2,0))
#    template = cv2.cvtColor(template, cv2.COLOR_RGB2BGR)
#    cv2.imshow('img', template)
#    cv2.waitKey(0)
    """"""
    if use_gpu:
        template = V(template.cuda())
        model = model.cuda()
    else:
        template = V(template)

    fps = 20
    fourcc = cv2.cv2.VideoWriter_fourcc('M','J','P','G')
    videoWriter = cv2.VideoWriter(video_dir, fourcc, fps, video_size)
    
    RESET = True
    count = 0
    
    for index in range(1, len(os.listdir(root_dir+'/img/'))):
#    for index in range(0, 30):
#        print(index)
            
        gtbox = os.listdir(root_dir +'/label/')[index]
        with open(root_dir +'/label/' + gtbox) as f:
            gtbox = f.read().split(',')
        gtbox = [float(i) for i in gtbox]
        gtbox = x1y1wh_to_xywh(gtbox)
##        gtbox = x1y1x2y2_to_xywh(gtbox)
#        if RESET:
#            last_box = gtbox
#        """"""
#        """"""
#        template = os.listdir(root_dir+'/img/')[index]
#        template = Image.open(root_dir +'/img/'+ template)
#        template, _, _ = _transform(template, last_box, 1, 127)
#        if use_gpu:
#            template = V(template.cuda())
#            model = model.cuda()
#        else:
#            template = V(template)
        """"""
        print(last_box)
        detection = os.listdir(root_dir+'/frames/')[index]
        detection = Image.open(root_dir+'/frames/' + detection)
        detection, pcc, ratio = _transform(detection, last_box, 2, 255)
        """"""
#        detection = detection.squeeze()
#        detection = detection.numpy()
#        import cv2
#        import numpy as np
#        import math
#        from axis import xywh_to_x1y1x2y2
#        detection = np.transpose(detection,(1,2,0))
#        detection = cv2.cvtColor(detection, cv2.COLOR_RGB2BGR)
#        cv2.imshow('img', detection)
#        cv2.waitKey(0)        
        """"""
        if use_gpu:
            detection = V(detection.cuda())
        else:
            detection = V(detection)
        
        coutput, routput = model(template, detection)
        coutput, routput = coutput.squeeze(), routput.squeeze()
#        coutput_numpy = coutput.data.cpu().numpy()
#        pcc, ratio = pcc.squeeze(), ratio.squeeze()
                
        coutput = coutput.view(5, 2, 17, 17)
                
        coutput = torch.nn.Softmax2d()(coutput)
        coutput1 = coutput[:,1,:,:]

        if use_gpu:
            coutput1, routput = coutput1.data.cpu().numpy().astype(np.float64), routput.data.cpu().numpy()
        else:
            coutput1, routput = coutput1.data.numpy().astype(np.float64), routput.data.numpy()                    


        a = 64
        s = a**2
        r = [[3*math.sqrt(s/3.),math.sqrt(s/3.)], [2*math.sqrt(s/2.),math.sqrt(s/2.)], [a,a], [math.sqrt(s/2.),2*math.sqrt(s/2.)], [math.sqrt(s/3.),3*math.sqrt(s/3.)]]
        r = [list(map(round, i)) for i in r]
                
        center_size = 5
        #coutput and csize
        coutput1 = coutput1[:, (8-center_size):(10+center_size), (8-center_size):(10+center_size)]
        
        loc1 = np.where(coutput1 == np.max(coutput1)) 
#        loc1 = np.where(coutput1 > 0.1)
        img = cv2.imread(root_dir+'/frames/'+os.listdir(root_dir+'/frames/')[index])

#        img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
#        img = transform2(img, last_box, 2, 255)
#        img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
#        total = {}
        for where in range(len(loc1[0])):
#            where = 0
            loc = [loc1[0][where], loc1[1][where]+8-center_size, loc1[2][where]+8-center_size]

            anchor = [7+15*loc[1], 7+15*loc[2]] + r[loc[0]]

            reg = [routput[loc[0]*4, loc[1], loc[2]], routput[loc[0]*4+1, loc[1], loc[2]], routput[loc[0]*4+2, loc[1], loc[2]], routput[loc[0]*4+3, loc[1], loc[2]]]

            pro = [anchor[0]+reg[0]*anchor[2], anchor[1]+reg[1]*anchor[3], anchor[2]*math.exp(reg[2]), anchor[3]*math.exp(reg[3])]
#            pro = anchor

            pro = [pro[0]*ratio+pcc[2]-pcc[0], pro[1]*ratio+pcc[3]-pcc[1], pro[2]*ratio, pro[3]*ratio]
             
            list1 = xywh_to_x1y1x2y2(pro)
            list1 = list(map(lambda x:int(round(x)), list1))
            
#            total[','.join([str(i) for i in list1])] = sum(list1) - sum(gtbox)

            last_box = pro
            last_box = list(map(lambda x:int(round(x)), last_box))

#        list1 = list(total.keys())[list(total.values()).index(min(total.values()))].split(',')
#        list1 = [int(i) for i in list1]

        gtbox = xywh_to_x1y1x2y2(gtbox)
        gtbox = list(map(lambda x:int(round(x)), gtbox))
        try:
            cv2.rectangle(img, (list1[0],list1[1]), (list1[2],list1[3]), (0,255,0), 1)
        except OverflowError:
            print(list1)
        cv2.rectangle(img, (gtbox[0],gtbox[1]), (gtbox[2],gtbox[3]), (255,0,0), 1)
        
#        if IOU(gtbox, list1) < 0.1:
#            RESET = True
#            count += 1
#        cv2.imshow('img', img)
#        cv2.waitKey(0)

        videoWriter.write(img)
    videoWriter.release()
    print(count)
    return

#%%
if __name__ == '__main__':
    model = SiameseRPN()
    show_output(
            root_dir = './TrackingNet/Test/'
#            root_dir = './lq/'
            , 
            model = model
            ,
            pth_file = './TrackingNet/epoch_31.pth'
            ,
            use_gpu = True
            )
