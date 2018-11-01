import numpy as np
from PIL import Image
import sys

# weight_matrix = []

def iterate(iterations, weight_matrix, reg, red_img, y):
    losses_history = []
    learn_rate = 0.000001
    # print(y)
    for i in range(iterations):
        loss = 0
        grad = np.zeros_like(weight_matrix)
        scores = np.matmul(weight_matrix, red_img)
        num_train = red_img.shape[1]
        scores = scores - np.max(scores)
        dim = red_img.shape[0]
        scores_exp = np.exp(scores)
        correct_scores_exp = scores_exp[y, range(num_train)]
        scores_exp_sum = np.sum(scores_exp,axis=0)
        scores_exp_normalized = scores_exp / scores_exp_sum
        scores_exp_normalized[y, range(num_train)] -= 1
        loss = -np.sum(np.log(correct_scores_exp / scores_exp_sum))
        loss /= num_train
        loss = loss + 0.5*reg*np.sum(weight_matrix*weight_matrix)
        losses_history.append(loss)
        grad = scores_exp_normalized.dot(red_img.T)
        grad /= num_train
        grad = grad + reg*weight_matrix
        weight_matrix -= learn_rate * grad

    return losses_history, weight_matrix

def func(init, train_arr, mean_train, N, d, y):
    no_cl = init
    train_np = train_arr
    mean_arr = [mean_train]*N
    eig_pairs = []
    cov = train_np - mean_arr
    eig_val, eig_vec = np.linalg.eigh(np.matmul(cov.T, cov))
    len1 = len(eig_val)
    K = 2*32
    for i in range(len1):
        eig_pairs.append((eig_val[i], eig_vec[:,i]))
    # eig_pairs = eig_pairs.append((eig_val[i], eig_vec[i]) for i in range(len(eig_val)))
    img1 = train_arr[1]
    eig_pairs.sort()
    eig_pairs.reverse()
    eig_val.sort()
    sorted_eigenvectors = np.zeros((d,d))
    sorted_eigenvalues = np.zeros((d,1))
    for i in  range(len(eig_val)):
        sorted_eigenvalues[i] = eig_pairs[i][0]
    for i in range(len(eig_val)):
        sorted_eigenvectors[:,i] = eig_pairs[i][1]
    comps = sorted_eigenvectors.T[:K]
    red_img = np.matmul(train_np, (sorted_eigenvectors.T[:K]).T).T
    weight_matrix = np.random.randn(no_cl, red_img.shape[0])*0.001
    losses_history, weight_matrix = iterate(1000, weight_matrix,100, red_img, y)
    # print(comps)
    return comps,losses_history,red_img,weight_matrix



def main():
    d = 32*32
    class_arr = {}
    img_dict = {}
    train_arr = []
    valtoname = {}
    each = []
    init = 0
    N = 0
    mean_img = [[0]*32]*32
    mean_train = [0]*32*32
    y = []
    imstr = 'images'
    valstr = 'value'
    train_path = sys.argv[1]
    N = sum(1 for line in open(train_path))
    with open(train_path,'r') as f:
        for line in f:
            line = line.split(' ')
            im = line[0]
            cl = line[1]
            tt = class_arr.get(cl)
            if(cl not in img_dict):
                img_dict[line[1]] = {}
                img_dict[line[1]][imstr] = []
                img_dict[line[1]][valstr] = []
                img_dict[line[1]][valstr] = init
                valtoname[init] = line[1]
                init += 1
            y.append(img_dict[line[1]][valstr])
            # print(img_dict[cl][valstr])
            temp = Image.open(im).convert('L').resize((32,32), Image.BILINEAR)
            temp  = np.array(temp)
            temp = temp.flatten()
            img_dict[line[1]][imstr].append(temp)
            mean_train = mean_train + (temp/N)
            train_arr.append(temp)
            #print(data)

    for i in img_dict:
        img_dict[i][valstr] = []
        img_dict[i][valstr] = init
        init += 1

    comps,losses_history,reduced_img,weight_matrix = func(init, np.array(train_arr), mean_train, N, d, y)
    train_dict = img_dict
    # print(red_img)
    # img_vals, cov_inds = func1(a1,N,d)
    s = 32
    answer = []
    with open(sys.argv[2],'r') as f:
        for line in f:
            img = line.split()[0]
            img = Image.open(img).convert('L').resize((32, 32), Image.BILINEAR)
            img = np.array(img)
            img = img.flatten().reshape(1,d)
            scores = weight_matrix.dot(np.matmul(img, comps.T)[0])
            pred_ys = np.argmax(scores, axis = 0)
            answer.append(valtoname[pred_ys])
            print(valtoname[pred_ys], end='')


if __name__ == "__main__":
    main()
