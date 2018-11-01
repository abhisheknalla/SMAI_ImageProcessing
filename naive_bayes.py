import numpy as np
from PIL import Image#do SVD----------------------------------------
import sys
def func1(a1, N, d):
        k=0
        means = np.empty((0,d))
        # print(a1[0][0], a1[1][0],a1[2][0],a1[3][0],a1[4][0],a1[5][0],a1[6][0],a1[7][0],a1[8][0])
        for j in range(d):
            summ = 0
            for i in range(N):
                # if(i==0):
                #     summ = summ + a1[j]
                # if(k<10):
                #     print(a1[i][j])
                # k=k+1
                summ = summ + a1[i][j]
            # print(summ/N)
            means = np.append(means, [summ/N])
        # print(means[0])
        # a1 = a1 - means
        # a2 = a1*a1
        a2 = np.empty((0,d), int)
        for i in range(N):
            tempo = np.empty((0,d), int)
            for j in range(d):
                temp = (a1[i][j] - means[j])*(a1[i][j] - means[j])
                tempo = np.append(tempo, [temp])
            a2 = np.append(a2, [tempo], axis=0)
        # print(np.shape(a2))
        cov_1 = np.empty((0,d))
        for j in range(d):
            summ = 0
            for i in range(N):
                summ = summ + a2[i][j]
            cov_1 = np.append(cov_1, [summ])
        # print(cov_1[0])
        # print(np.shape(cov_1))
        for i in range(d):
            cov_1[i] = cov_1[i]/N
        # print(cov_1)
        cov_inds = np.argsort(-cov_1)
        #s[i]])
        # print(cov_inds)
        img_vals = np.empty((0,32))
        for i in range(N):
            tempo = np.empty((0,32), int)
            for j in  range(32):
                tempo = np.append(tempo,a1[i][cov_inds[j]])
            img_vals = np.append(img_vals, [tempo], axis=0)
        # print(img_vals[0])
        return img_vals, cov_inds

def rgb2gray(rgb):
    r = rgb[:,:,0]
    g = rgb[:,:,1]
    b = rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def gaussian(x, mu, sig):
    return np.exp(-((x - mu)**2)/(2*sig))/((sig)**0.5)

def main():
    d = 64*64
    a1=[]
    # a1 = np.empty((0,d), int)
    class_arr = {}
    each = []
    N=0
    train_path = sys.argv[1]
    with open(train_path,'r') as f:
        for line in f:
            line = line.split(' ')
            im = line[0]
            cl = line[1]
            tt = class_arr.get(cl)
            if(tt==None):
                class_arr[cl] = 1
            else:
                class_arr[cl] += 1
            img = Image.open(im)
            # img  = img.resize((64,64), Image.ANTIALIAS)
            img = rgb2gray(np.asarray(img))
            # img = img.resize((64,64),Image.ANTIALIAS)
            # img.load()
            data = img.ravel()
            # data = np.asarray(img)
            # data = data.reshape(-1)
            tup1 = (cl, data)
            each.append(tup1)
            #print(data)
            a1.append(data)
            # a1 = np.append(a1, [data], axis=0)
            N=N+1
    a1 = np.array(a1)
    # print(a1)
    # for x,y in class_arr.items():
    #     print("classes ",x, y)
    # print(a1)
    # print(a1[0])
    img_vals, cov_inds = func1(a1,N,d)
    # print(img_vals)
    # for i in range(N):
    #     print("important image values ",i,img_vals[i])

    # print(len(each))
    # for i in range(len(each)):
    #     print(np.shape(each[i][1]))
    mean_full = np.empty((0,32))
    var_full = np.empty((0,32))
    for x,y  in class_arr.items():
        eigvec_cl = np.empty((0,32))
        # for j in range(y):
        for k in range(len(each)):
            if(each[k][0]==x):
                temp = np.array(img_vals[k])
                eigvec_cl = np.append(eigvec_cl, [temp], axis=0)
                # temp = np.array(a1[k])
                # temp_m = func1(temp, 1, d)
                # temp_m = np.array(temp_m)
                # eigvec_cl = np.append(eigvec_cl, [temp_m], axis=0)
                # print(temp)

        mean_cl = np.empty((0,d))
        for j in range(32):
            summ = 0
            for i in range(y):
                summ = summ + eigvec_cl[i][j]
            mean_cl = np.append(mean_cl, [summ/y])
        # print(mean_cl)
        mean_full = np.append(mean_full, [mean_cl], axis=0)
        var_cl = np.empty((0,d))
        for j in range(32):
            summ = 0
            for i in range(y):
                summ = summ + (eigvec_cl[i][j] - mean_cl[j])*(eigvec_cl[i][j] - mean_cl[j])
            # print(summ)
            var_cl = np.append(var_cl, [summ/y])
        # print(var_cl)
        var_full = np.append(var_full, [var_cl], axis=0)
    test_path = sys.argv[2]
    with open(test_path,'r') as f:
        for line in f:
            im = line.split()[0]
            img = Image.open(im)
            img  = rgb2gray(np.asarray(img))
            data = img.ravel()
            # print(np.shape(data))
            # data = data.reshape(-1)
            img_vals = np.empty((0,32))
            tempo = np.empty((0,32))
            for j in  range(32):
                tempo = data[cov_inds[j]]
                img_vals = np.append(img_vals, [tempo])
            # print(img_vals)
            probs_full = np.empty((0,d))
            # print(len(class_arr))
            for i in range(len(class_arr)):
                probs_1 = 1
                for j in range(32):
                    x = gaussian(img_vals[j], mean_full[i][j], var_full[i][j])
                    probs_1 = probs_1*x
                prob1 = probs_1
                prob1 = prob1*list(class_arr.values())[i]
                # print(list(class_arr.values())[i])
                probs_full = np.append(probs_full, [prob1])

            # print(probs_full)
            maxv = -1000
            maxi = -1
            for i in range(len(probs_full)):
                if(probs_full[i] > maxv):
                    maxv = probs_full[i]
                    maxi = i
            print(list(class_arr.keys())[maxi], end='')
            # print(img_vals)
    # print(mean_full)

if __name__ == "__main__":
    main()
