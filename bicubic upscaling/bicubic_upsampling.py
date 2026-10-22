"""
      __
     | (a+2)x^3 - (a+3)x^2 + 1           |x| <=1
w(x)=|  a|x|^3 - 5a|x|^2 + 8a|x| -4a     1 < |x| <2
     |  0                                otherwise
      ‾‾

"""
from PIL import Image
import time
import numpy as np


img=Image.open("test.png").convert("L")
img_arr=np.array(img)
start=time.time()
orig_greyscale=Image.open("test.png").convert("L")
orig_greyscale.save("orig_greyscale.png")

def bicubic_wgts_calc(x, a=-0.5):
    abs_x=np.abs(x)
    wgts=np.zeros_like(abs_x, dtype=np.float32)
    case1=abs_x<=1
    wgts[case1]=((a+2)*(abs_x[case1]**3)) - ((a+3)*(abs_x[case1]**2)) + 1

    case2=(abs_x>1) & (abs_x<2)
    wgts[case2]=(a*(abs_x[case2]**3)) - 5*a*(abs_x[case2]**2) + 8*a*abs_x[case2] - 4*a

    return wgts


def bicubic_upscale(image, factor):
    h_curr,w_curr=image.shape
    h_new=int(h_curr*factor)
    w_new=int(w_curr*factor)
    y,x=np.meshgrid(np.arange(h_new), np.arange(w_new), indexing='ij')
    y_old=y/factor
    x_old=x/factor
    dy=y_old-np.floor(y_old).astype(np.int32)
    dx=x_old-np.floor(x_old).astype(np.int32)
    w_y=np.stack([bicubic_wgts_calc(m-dy) for m in range(-1,3)], axis=-1)
    w_x=np.stack([bicubic_wgts_calc(m-dx) for m in range(-1,3)], axis=-1)
    output=np.zeros((h_new,w_new), dtype=np.float32)

    for i,p in enumerate(range(-1,3)):
        y_near=np.clip(np.floor(y_old).astype(np.int32)+p,0,h_curr-1)
        for j,q in enumerate(range(-1,3)):
            x_near=np.clip(np.floor(x_old).astype(np.int32)+p,0,w_curr-1)
            near_pixels=image[y_near,x_near]
            output+=near_pixels*w_y[...,i]*w_x[...,j]
    return np.clip(output, 0, 255).astype(np.uint8)


large_img_arr=bicubic_upscale(img_arr, factor=2.0)
print("time: "+str(round(time.time()-start, 4)))
result=Image.fromarray(large_img_arr)
result.save("output.jpg")