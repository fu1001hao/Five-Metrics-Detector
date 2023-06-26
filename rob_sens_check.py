import torch
from skimage import filters
import scipy
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
from sklearn.svm import OneClassSVM
from skimage.util import random_noise
from config import *
import sys
from models import *
from sklearn import metrics
from sklearn.metrics import roc_auc_score, auc
from cuda import *

roc = True

def load():
    model = AAA()
    model.load_state_dict(torch.load('a2o/a2o.pth'))
    model = model.to(device)
    model.eval()

    bdx, bdy = torch.load('a2o/bd_inputs.pt'), torch.load('a2o/bd_labels.pt')
    valx, valy = torch.load('data/val_inputs.pt'), torch.load('data/val_labels.pt')
    clx, cly = torch.load('data/clean_inputs.pt'), torch.load('data/clean_labels.pt')

    k = 30
    valx, valy = valx[:k], valy[:k]

    return model, valx, valy, bdx, bdy, clx, cly

def copy_past(inputs, single, indices):
    x1, y1, x2, y2 = indices
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    inputs[:, :, x1:x2, y1:y2] = single[:, x1:x2, y1:y2]
    tmpx = torch.Tensor([single.tolist()]*len(inputs) )
    tmpx[:, :, x1:x2, y1:y2] = inputs[:,:, x1:x2, y1:y2]
    return inputs, tmpx

def expectation(model, rawx, valy, x1, x2):
    rob, weak, sen, inver = [], [], [], []
    batch = 5000
    valy = valy.to(device)
    with torch.no_grad():
        for i in range(0, len(rawx), batch):
            j = min(i+batch,len(rawx))

            inputs = rawx[i:j]
            inputs = inputs.to(device)
            pred = model(inputs).argmax(1)

            inputs = x1[i:j]
            inputs = inputs.to(device)
            pred1 = model(inputs).argmax(1)

            rob += (pred == pred1).float().tolist()
            weak += (pred1 == valy).float().tolist()

            inputs = x2[i:j]
            inputs = inputs.to(device)
            pred2 = model(inputs).argmax(1)

            sen += (pred == pred2).float().tolist()
            inver += (pred2 == valy).float().tolist()

    return [rob, weak, sen, inver]

def add_noise(model, x, r):
    noise = torch.randn(x[0].shape)*r
    rawx = x+noise
    inv = []
    batch = 5000
    with torch.no_grad():
        for i in range(0, len(rawx), batch):
            j = min(i+batch,len(rawx))

            inputs = x[i:j]
            inputs = inputs.to(device)
            pred = model(inputs).argmax(1)

            inputs = rawx[i:j]
            inputs = inputs.to(device)
            pred1 = model(inputs).argmax(1)

            inv += (pred == pred1).float().tolist()

    return inv


def efficient(model, raw_x, raw_y, valx, valy, poison = False):
    stat = []
    lenx, leny, lenc = raw_x.shape[2], raw_x.shape[3], raw_x.shape[1]
    for k in box_ratio:
        indices = (lenx*(0.5-k/2), leny*(0.5-k/2), lenx*(0.5+k/2), leny*(0.5+k/2))
        tmp = []
        for i in range(len(valx)):
            handcraft1, handcraft2 = copy_past(raw_x.clone(), valx[i], indices)
            tmp.append(expectation(model, raw_x, valy[i], handcraft1, handcraft2))
        tmp = np.array(tmp)
        tmp = tmp.mean(0)
        stat.append(tmp.tolist())
    stat = np.array(stat)
    noise = []
    for r in noise_intense:
        tmp = []
        for i in range(len(valx)):
            tmp.append( add_noise(model, raw_x.clone(), r))
        tmp = np.array(tmp)
        tmp = tmp.mean(0)
        noise.append(tmp.tolist())
    noise = np.array(noise)
    stat = stat.transpose(2, 1, 0)
    noise = noise.transpose()

    return stat, noise

def evaluate_roc(score_clean, score_bd):

    m = 100
    thres_pools = np.arange(1, m)*1.0/m
    fpr, tpr = [0], [0]
    pre, rec = [1], [0]
    for ratio in thres_pools:
        thres = score_clean[int(len(score_clean)*ratio)]
        TP =  np.sum(score_bd <= thres)*1.0
        FP =  np.sum(score_clean <= thres)*1.0
        Pre = TP / (TP + FP) 
        Rec  = TP / (TP + np.sum(score_bd > thres))
        tpr.append(TP/len(score_bd))
        fpr.append(FP/len(score_clean))
        pre.append(Pre)
        rec.append(Rec)
    tpr.append(1)
    fpr.append(1)
    pre.append(0)
    rec.append(1)
    AUROC = metrics.auc(fpr, tpr)
    AUPR = auc(rec, pre)
    i = 1
    print("TPR {:.4f} FPR {:.4F} AUROC {:.4f} AUPR {:.4f}".format(tpr[i], fpr[i], AUROC, AUPR))


def main_run(model, valx, valy, bdx, bdy, clx, cly ):
    print('Val Testing....')
    valstat,  valnoi  = efficient(model, valx, valy, valx, valy)
    metric = ['robust', 'weak', 'sens', 'inverse']
    cl = []
    detect = []
    nn = 4
    for i in range(nn):
        tmp = LocalOutlierFactor(novelty = True)
        tmp.fit(valstat[:,i,:])
        cl.append(tmp)
        detect.append(cl[i].decision_function(valstat[:,i, :]))
    clnoi = LocalOutlierFactor(novelty = True)
    clnoi.fit(valnoi)
    detect.append(clnoi.decision_function(valnoi))
    detect = np.vstack(detect).transpose()
    metacl = LocalOutlierFactor(novelty = True)
    metacl.fit(detect)
    print('Backdoor Testing....')
    bdstat,  bdnoi  = efficient(model, bdx, bdy, valx, valy, poison = True)
    detect = []
    for i in range(nn):
        tmp = cl[i].decision_function(bdstat[:, i, :])
        detect.append(tmp)
    detect.append(clnoi.decision_function(bdnoi))
    detect = np.vstack(detect).transpose()
    y1 = metacl.decision_function(detect)

    print('Clean Testing....')
    clstat,  cl_noi = efficient(model, clx, cly, valx, valy)
    detect = []
    for i in range(nn):
        tmp = cl[i].decision_function(clstat[:, i, :])
        detect.append(tmp)
    detect.append(clnoi.decision_function(cl_noi))
    detect = np.vstack(detect).transpose()
    y2 = metacl.decision_function(detect)
 
    y1 = np.sort(y1, kind = 'mergesort')
    y2 = np.sort(y2, kind = 'mergesort')
    evaluate_roc(y2, y1)

model, valx, valy, bdx, bdy, clx, cly = load()
main_run(model, valx, valy, bdx, bdy, clx, cly )

