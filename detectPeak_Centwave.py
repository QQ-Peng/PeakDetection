import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import collections
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] =False

class ROIList:
    """
    detect region of interest, in which the MZs are from a continues retention time and with a predefined deviation.
    """
    def __init__(self, mzList, rt, intsList,ppm=25*1e-6, min_num=6, minIntens=6000) -> None:
        #
        self.ROIs = {i:{'mzs':[mzList[i]], 'rts':[rt], 'ints': [intsList[i]] } for i in range(len(mzList))}
        self.ROI_means = {i:mzList[i] for i in range(len(mzList))} # mz_means
        self.ROI_mz_numbers = {i:1 for i in range(len(mzList))}
        self.current_ROI_idx = len(mzList) - 1
        #
        self.ppm = ppm
        self.min_num = min_num # 每个ROI至少包含的mz数目
        self.minIntens = minIntens
        #
        self.removed_ROIs = {}
        self.completed_ROIs = {}
    def add_ROI(self,mz,rt,ints):
        """
        add a new ROI.
        """
        self.current_ROI_idx += 1
        self.ROIs[self.current_ROI_idx] = {'mzs': [mz], 'rts': [rt],'ints':[ints]}
        self.ROI_means[self.current_ROI_idx] = mz
        self.ROI_mz_numbers[self.current_ROI_idx] = 1
    def check_and_clean(self,mark):
        for i, add in mark.items():
            # add为True为False，标记着ROI是否有新的mz加入
            if not add:
                if self.ROI_mz_numbers[i]>=self.min_num and max(self.ROIs[i]['ints'])>=self.minIntens:
                    self.completed_ROIs[i] = self.ROIs[i]
                    del self.ROIs[i]
                else:
                    self.removed_ROIs[i] = self.ROIs[i]
                    del self.ROIs[i]
    def update(self, mzList, rt, intsList,lastscan=False) -> None:
        """
        update the ROIList with next scan.
        """
        mark = {i:False for i in self.ROIs.keys()} # 标记每一个ROI是否有新的mz加入
        for mz, ints in zip(mzList,intsList):
            add = False # 标记是否添加到已有的ROI中
            for idx in self.ROIs.keys():
                means = self.ROI_means[idx]
                if abs(mz-means)<=self.ppm: # BUG: 检查计算公式，看看还是否需要除以means
                    self.ROIs[idx]['mzs'].append(mz)
                    self.ROIs[idx]['rts'].append(rt)
                    self.ROIs[idx]['ints'].append(ints)
                    self.ROI_means[idx] = np.mean(self.ROIs[idx]['mzs'])
                    self.ROI_mz_numbers[idx] += 1
                    add = True
                    mark[idx] = True
                    break
            if not add:
                self.add_ROI(mz, rt, ints)
        if lastscan: # 如果是最后一次扫描了，则对所有的ROI进行筛选
            mark = {i:False for i in self.ROIs.keys()}
        self.check_and_clean(mark)

def convolve_cir(signal, filter):
    assert len(signal) == len(filter)
    out = []
    for k in range(len(signal)):
        rolled_filter = np.roll(filter, k)  # 循环移动filter
        o = np.dot(signal, rolled_filter)  # 计算点积
        out.append(o)
    return np.array(out)


def CWT(x, scales = [1], wavelet = "mexh", extendLengthMSW = False):
    if wavelet == "mexh":
        psi_xval = np.linspace(-6, 6, 256)
        psi = (2/np.sqrt(3) * np.pi**(-0.25)) * (1 - psi_xval**2) *np.exp(-psi_xval**2/2)
    else:
        raise "Unsupported wavelet format!"
    
    oldLen = len(x)
    if extendLengthMSW:
        pass
    else:
        pass
    Len = len(x)
    nbscales = len(scales)
    wCoefs = [] # 储存不同尺度下的小波变换的系数
    psi_xval = psi_xval - psi_xval[0]
    dxval = psi_xval[1] # 取时间间隔
    xmax = psi_xval[len(psi_xval)-1] # 取最大的时间 12.0
    scales_val = []
    for i in range(len(scales)): 
        scale_i = scales[i] # 取出第i个scale
        f = np.zeros(Len) # 生成一个0向量，长度为x的长度
        j = np.int32(np.floor(np.arange(0,scale_i * xmax,1)/(scale_i * dxval))) # 当scale.i*dxval=1时，分子每增加1，index就增加1，当scale很小，scale*dxval很小，分子增加1，index增加大于1
        # j = j[1:]
        # j = np.append(j,255)
        # print(j)
        # scale=2: [  0  10  21  31  42  53  63  74  85  95 106 116 127 138 148 159 170 180 191 201 212 223 233 244 255]
        # scale=1: [  0  21  42  63  85 106 127 148 170 191 212 233 255]
        #以上计算的j实际上是index。
        
        if len(j) > Len:
            i = i-1; break ##  stop(paste("scale", scale.i, "is too large!"))
        scales_val.append(scale_i)
        lenWave = len(j)  # 可以看到，scale越大，index的元素越多，这个信息有什么含义吗？index元素越多，当与时间向量x的元素一一配对时，时间窗口更长，相当于小波被拉宽了
        f[0:lenWave] = psi[j][::1] - np.mean(psi[j]) # [::1]表示将该向量倒置，倒置的用处大吗？因为这个墨西哥小波是对称的。将f向量的前lenWave个元素设为小波的函数值
        #######################
        # 下面两行的代码是xcms里的源码，这个逻辑是否正确？因为从上面的代码可以看出，f的长度就是Len
        # if len(f) > Len:
        #     i = i-1; break ##  stop(paste("scale", scale.i, "is too large!"))
        #######################
        
        wCoefs_i = 1/np.sqrt(scale_i) * convolve_cir(x, f)
        new_order = np.concatenate((np.arange(Len - np.floor(lenWave/2),Len), np.arange(0,Len - np.floor(lenWave/2))))
        # print(new_order)
        # print("new:",new_order)
        # print( wCoefs_i)
        wCoefs_i = wCoefs_i[np.int32(new_order)]
        wCoefs.append(wCoefs_i.reshape(1,-1))
    wCoefs = np.concatenate(wCoefs)
    return psi_xval[j], psi[j], wCoefs,scales_val
    

def localMaximum(x, winSize = 5):
    # 输入x是一个向量
    ## from package MassSpecWavelet
    Len = len(x)
    rNum = np.ceil(Len/winSize)

    ## Transform the vector as a matrix with column length equals winSize
    ##		and find the maximum position at each row.
    y = np.concatenate([x, np.repeat(x[Len-1], rNum * winSize - Len)])
    y = y.reshape(-1,winSize).T

    y_maxInd = np.int32(np.apply_along_axis(np.argmax, axis=0, arr=y)) # 记录了每一列的最大值的行数
    
    # ## Only keep the maximum value larger than the boundary values
    def custom_function(x, winSize):
        return np.max(x) > x[0] and np.max(x) > x[winSize-1]
    # selInd <- which(apply(y, 2, function(x) max(x) > x[1] & max(x) > x[winSize]))
    selInd = np.array([i for i in range(y.shape[1]) if custom_function(y[:, i], winSize)],dtype=np.int32)# 记录了哪些列的最大值不在边界
    
    # ## keep the result
    localMax = np.repeat(0, Len)
    localMax[selInd * winSize + y_maxInd[selInd]] = 1

    # ## Shift the vector with winSize/2 and do the same operation
    shift = np.int32(np.floor(winSize/2))
    rNum = np.ceil((Len + shift)/winSize)
    y = np.concatenate([np.repeat(x[0], shift), x, np.repeat(x[Len-1], rNum * winSize - Len-shift)]) # 与上面的窗口岔开
    y = y.reshape(-1,winSize).T
    y_maxInd = np.int32(np.apply_along_axis(np.argmax, axis=0, arr=y)) # 记录了每一列的最大值的行数
    
    # ## Only keep the maximum value larger than the boundary values
    selInd = np.array([i for i in range(y.shape[1]) if custom_function(y[:, i], winSize)],dtype=np.int32) # 记录了哪些列的最大值不在边界
    localMax[selInd * winSize + y_maxInd[selInd]-shift] = 1

    # ## Check whether there is some local maxima have in between distance less than winSize
    maxInd  = np.where(localMax > 0)[0]
    selInd = np.where(np.diff(maxInd) < winSize)[0]

    if len(selInd) > 0: 
        selMaxInd1  = maxInd[selInd]
        selMaxInd2 = maxInd[selInd + 1]
        temp = x[selMaxInd1] - x[selMaxInd2]
        localMax[selMaxInd1[temp <= 0]] = 0
        localMax[selMaxInd2[temp > 0]] = 0

    return localMax


def getLocalMaximumCWT(wCoefs, scales=[1], minWinSize=5, amp_Th=0):
    """
    wCoefs: 行是scale，列是滤波移动长度
    """
    assert wCoefs.shape[0] == len(scales)
    localMax = []

    for i in range(len(scales)): 
        scale_i = scales[i]
        winSize_i = scale_i * 2 + 1
        if winSize_i < minWinSize: 
            winSize_i = minWinSize
        temp = localMaximum(wCoefs[i,:], winSize_i)
        localMax.append(temp.reshape(1,-1))
    localMax = np.concatenate(localMax)
    ## Set the values less than peak threshold as 0
    localMax[wCoefs < amp_Th] = 0
    
    return localMax

def getRidge(localMax, scales=None, iInit=None, step=-1, iFinal=0, minWinSize=3, gapTh=3, skip=None):
    """
    localMax: 行是scale，列是滤波移动长度
    """
    ## modified from package MassSpecWavelet
    localMax = localMax.T # 列是scale，行是滤波移动长度
    if iInit is None:
        iInit = localMax.shape[1] - 1
    else:
        assert iInit<localMax.shape[1]

    if scales is None:
        scales = np.arange(1,localMax.shape[1]+1)

    maxInd_curr = np.where(localMax[:, iInit] > 0)[0] # 最大尺度下的局部最大值的坐标
    nMz = localMax.shape[0] # 在质谱图里，是mz; 在色谱图里，是保留时间rt

    if skip is None:
        skip = iInit + 1

    ## Identify all the peak pathes from the coarse level to detail levels (high column to low column)
    ## Only consider the shortest path
    if localMax.shape[1] > 1:
        colInd = np.arange(iInit+step, iFinal+step, step) # 上面已经用最大的scale进行初始化了，因此接下来不再访问最大的scale
    else: 
        colInd = np.array([0]) # FIXED: 当输入的localMax只包含一个尺度时，那么将接下来要访问的尺度设置为第1=0+1个尺度，即为1
                                # 那么检出的岭线即为最大尺度的局部最大值
                               
    ## 初始化岭线
    ridgeList = collections.defaultdict(list)
    peakStatus = collections.defaultdict(int)
    for i in maxInd_curr.tolist():
        ridgeList[i].append(i)
        peakStatus[i] = 0

     ## orphanRidgeList keep the ridges disconnected at certain scale level
    ## Changed by Pan Du 05/11/06
    orphanRidgeList = {}
    # orphanRidgeName = None
    nLevel = len(colInd)

    for j in range(nLevel):
        # print("scales: ", scales)
        
        col_j = colInd[j] # 依次往细粒度尺度进行访问
        # print("col_j: ", col_j)
        scale_j = scales[col_j]
        if col_j == skip:
            # ridgeList <- lapply(ridgeList, function(x) c(x, x[length(x)]))
            # 如果跳过这一个scale，则将该scale下的最大值的坐标用上一个scala的坐标替换
            ridgeList = {k:ridgeList[k]+[ridgeList[k][-1]] for k in ridgeList.keys()}
            # print(ridgeList)
            continue

        ############ TODO: 这一段还需要进一步理解 #####
        ############ 理解：后面会更新maxInd_curr，当这个为空时，就需要重新计算
        if len(maxInd_curr) == 0: 
            maxInd_curr = np.where(localMax[:, col_j] > 0)[0]
            continue
        ##############################################

        ## The slide window size is proportional to the CWT scale
        ## winSize.j <- scale.j / 2 + 1
        # 这里的window size与找localMax的设置方法不一样
        winSize_j = np.int32(np.floor(scale_j/2)) # TODO: 这里可能是假阳性高的原因之一，因为window_size设置的相对较小。
        if winSize_j < minWinSize:
            winSize_j = minWinSize
        
        selPeak_j = []
        remove_j = []
        new_name_map = {}
        for k in range(len(maxInd_curr)):
            ind_k = maxInd_curr[k] # 记录当前局部最大值的index,也是记录岭线字典的Key
            start_k = 0 if ind_k - winSize_j < 0 else ind_k - winSize_j # 计算搜索范围的起始index
            end_k = nMz - 1 if ind_k + winSize_j > nMz - 1 else ind_k + winSize_j
            # 计算目前scale下scale_j在上一个scale的第k个最大值范围内的最大值index
            ind_curr = np.where(localMax[start_k:end_k+1, col_j] > 0)[0] + start_k
            
            if len(ind_curr) == 0:
                status_k = peakStatus[ind_k]
                ## bug  work-around
                if status_k is None:
                    status_k = gapTh +1
                
                if status_k > gapTh and scale_j >= 2:
                    temp = ridgeList[ind_k]
                    orphanRidgeList[f"{str(col_j + status_k + 1)}_{str(ind_k)}"] = temp[0:len(temp)-status_k]
                    remove_j.append(ind_k) # 把中断超过gapTh的ridge line删掉
                    continue
                else:
                    # 如果在当前scale下没有在上一个scale的第k个最大值的范围内找到最大值
                    # 则用上一个scale的第k个最大值替换
                    ind_curr = ind_k
                    peakStatus[ind_k] = status_k + 1
                    # print("peakstatus: ",peakStatus)
            else:
                peakStatus[ind_k] = 0
                if len(ind_curr) >= 2:
                    ind_curr = ind_curr[np.argmin(abs(ind_curr - ind_k))]
                    # print("ind_curr: ",ind_curr)
            ridgeList[ind_k].append(int(ind_curr)) # 将新的最大值点加入岭线中
            
            selPeak_j.append(int(ind_curr)) 
            new_name_map[int(ind_k)] = int(ind_curr) # 用来更新名字
            
        if len(remove_j) > 0:
            for rj in remove_j:
                del ridgeList[rj]
                del peakStatus[rj]
        
        ## TODO:应对重复的峰
        # ## Check for duplicated selected peaks and only keep the one with the longest path.
        # dupPeak_j <- unique(selPeak_j[duplicated(selPeak_j)])
        # print("*"*20)
        # print("scale_j: ", scale_j)
        selPeak_j_pd = pd.Series(selPeak_j)
        # print("selPeak_j: ",selPeak_j)
        
        ##### 当遇到重复的峰时，保留最长的ridge
        dupPeak_j = selPeak_j_pd[selPeak_j_pd.duplicated()].values
        for dp in dupPeak_j:
            dupPeak_j_name = []
            for rdg in ridgeList.keys():
                if dp in ridgeList[rdg]:
                    dupPeak_j_name.append(rdg)
            dupPeak_j_len = [len(ridgeList[rdg]) for rdg in dupPeak_j_name]
            maxLen_ridge = dupPeak_j_name[dupPeak_j_len.index(max(dupPeak_j_len))]
            # 删除非最长的ridge
            for rdg in dupPeak_j_name:
                if rdg != maxLen_ridge:
                    del ridgeList[rdg]
        selPeak_j = list(set(selPeak_j))
        # print('new_name_map',new_name_map)
        # print("ridgeList: ",ridgeList)
        # print("selPeak_j",selPeak_j)

       
        ## Update the names of the ridgeList as the new selected peaks
        if len(ridgeList) > 0:
            ridgeList = {new_name_map[pk]:ridgeList[pk] for pk in ridgeList.keys()}
        if len(peakStatus) > 0:
            peakStatus = {new_name_map[pk]:peakStatus[pk] for pk in peakStatus.keys()}
        # print(ridgeList)
        # print("*"*20)
        ## If the level is larger than 3, expand the peak list by including other unselected peaks at that level
        if scale_j >= 2: 
            maxInd_next = np.where(localMax[:, col_j] > 0)[0]
            unSelPeak_j = maxInd_next[~np.isin(maxInd_next, selPeak_j)]
            maxInd_curr = selPeak_j

            for uj in unSelPeak_j:
                uj = int(uj)
                ridgeList[uj] = [uj]
                peakStatus[uj] = 0
                maxInd_curr.append(uj)
        else:
            maxInd_curr = selPeak_j
        # maxInd_curr = selPeak_j
        # print("*"*20)
        # print("scale: ",scale_j)
        # print("ridgeList: ",ridgeList)
        # print("peakStatus: ",peakStatus)
        # print("*"*20)

    ## Attach the peak level at the beginning of the ridge names
    if len(ridgeList) > 0:
        ridgeList = {"1_"+str(pk):ridgeList[pk] for pk in ridgeList.keys()}
    # print(ridgeList)
    # print(orphanRidgeList)
    ridgeList.update(orphanRidgeList)
    return ridgeList

