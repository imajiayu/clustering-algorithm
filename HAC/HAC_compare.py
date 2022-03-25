import os
from PIL import Image
import numpy as np

result_fileList=os.listdir('./HAC_result/')
gt_fileList=os.listdir('./data/gt/')

def compare(gt_matrix,re_matrix):#计算两个矩阵中不同元素的个数与总像素数的比值
    assert gt_matrix.shape==re_matrix.shape
    total=gt_matrix.shape[0]*gt_matrix.shape[1]
    return 1-len(np.argwhere(gt_matrix!=re_matrix))/total

for re_file in result_fileList:
    for gt_file in gt_fileList:
        if re_file==gt_file:
            pli_gt=Image.open('./data/gt/{}'.format(gt_file))
            gt_matrix=np.asarray(pli_gt.convert('L'))
            pli_re=Image.open('./HAC_result/{}'.format(re_file))
            re_matrix=np.asarray(pli_re.convert('L'))
            print(re_file.split('.')[0]+' '+str(compare(gt_matrix,re_matrix)))