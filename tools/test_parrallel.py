import torch
from torch._C import device
import torch.nn as nn
import threading
import sys

sys.path.append("../")
from public import models, decoder
from config.config_eval import Config
import numpy as np
import cv2
import queue
import time


class imageReader(threading.Thread):
    def __init__(self, img_list, device):
        super(imageReader, self).__init__()
        self.img_list = img_list
        self.device = device
        
        # self.q = im_q  # q = queue.Queue(10)
        self.stream = torch.cuda.Stream()

    def run(self):
        print("start load im!!")

        global im_q
        global exitFlag
        with torch.cuda.stream(self.stream):
            for img_path in self.img_list:
                for i in range(2):
                    im_stack= []
                    img = cv2.imread(img_path)
                    if img is None:
                        print('NoneType: %s' % img_path)

                    img = transform(img, self.device)
                    im_stack.append(img)
                threadLock.acquire()
                im_q.put(torch.cat(im_stack, 0))
                threadLock.release()

class Inference(threading.Thread):
    def __init__(self, model):
        super(Inference, self).__init__()
        self.model = model
        
    def run(self):
        exitFlag = 0
        global idx
        print("start inference!!")
        global im_q
        global out
        while not exitFlag:
            if not im_q.empty():
                with torch.no_grad():
                    im = im_q.get()
                    im = self.model(im)
                    out.append(1)
                    idx += 1
                    if idx == 500:
                        exitFlag=1
                    print(f"cur_out: {idx}", end="\r")

def normal_infer(im_list, model, device):
    global out
    idx = 0
    print("start normal infer!!!!")
    with torch.no_grad():
        for item in im_list:
            idx += 1
            im = cv2.imread(item)
            im = transform(im, device)
            im = model(im)
            out.append(1)
            print(f"cur_idx: {idx}", end="\r")


def transform(im, device):
    im = cv2.resize(im, (512,512))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) / 255.
    im = torch.from_numpy(im).float().to(device)
    im = im.permute(2,0,1).unsqueeze(0)
    return im

if __name__ == "__main__":
    
    idx = 0
    out = []
    threadLock = threading.Lock()
    im_list = ["./flag.jpg"]*500
    im_q = queue.Queue(8)
    device = torch.device("cuda:0")
    im_reader = imageReader(im_list, device)
    model = models.__dict__[Config.network](**{
        "pretrained": Config.pretrained,
        "backbone_type": Config.backbone_type, 
        "neck_type": Config.neck_type, 
        "neck_dict": Config.neck_dict, 
        "head_dict": Config.head_dict, 
        "backbone_dict": Config.backbone_dict
    })
    model = model.eval().to(device)
    print("success load model")
    inference = Inference(model)

    torch.cuda.synchronize()
    start = time.time()
    im_reader.start()
    inference.start()
    im_reader.join()
    inference.join()
    print(len(out))
    torch.cuda.synchronize()
    print("parallel time: {}".format(time.time() - start))

    del out
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    start = time.time()
    out = []
    normal_infer(im_list*2, model, device)
    print("normal time : {}".format(time.time() - start))
    