import math
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from scipy import integrate

def configureA():
    mean0 = np.array([-1, -1, -1, -1])
    cov0 = np.array([[5, 3, 1, -1],
                     [3, 5, -2, -2],
                     [1, -2, 6, 3],
                     [-1, -2, 3, 4]])
    det0 = np.linalg.det(cov0)

    mean1 = np.array([1, 1, 1, 1])
    cov1 = np.array([[1.6, -0.5, -1.5, -1.2],
                     [-0.5, 8, 6, -1.7],
                     [-1.5, 6, 6, 0],
                     [-1.2, -1.7, 0, 1.8]])
    det1 = np.linalg.det(cov1)

    total_samples = 10000
    threshold = 0.35
    return total_samples, threshold, mean0, mean1, cov0, cov1

def configureC():
    mean0 = np.array([-1, -1, -1, -1])
    cov0 = np.array([[5, 0, 0, 0], [0, 5, 0, 0], [0, 0, 6, 0], [0, 0, 0, 4]])
    det0 = np.linalg.det(cov0)

    mean1 = np.array([1, 1, 1, 1])
    cov1 = np.array([[1.6, 0, 0, 0],
                     [0, 8, 0, 0],
                     [0, 0, 6, 0],
                     [0, 0, 0, 1.8]])
    det1 = np.linalg.det(cov1)

    total_samples = 10000
    threshold = 0.35
    return total_samples, threshold, mean0, mean1, cov0, cov1

def classifier(decision_threshold, multivariate0, multivariate1, sample_result):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in sample_result:
        random_vector = i['samples']
        prob0 = multivariate0.pdf(random_vector)
        prob1 = multivariate1.pdf(random_vector)

        if (prob1 / prob0) > decision_threshold:
            pred = 1
            if i['label'] == 1:
                TP += 1
            else:
                FP += 1
        else:
            pred = 0

            if i['label'] == 1:
                FN += 1
            else:
                TN += 1
    return TP, TN, FP, FN
def get_samples(total_samples, threshold, mean0, mean1, cov0, cov1 ):
    sample_result = []
    for i in range(total_samples):
        result = {}
        if random.uniform(0, 1) < threshold:
            result['label'] = 0
            result['samples'] = np.random.multivariate_normal(mean0, cov0, 1)

        else:
            result['label'] = 1
            result['samples'] = np.random.multivariate_normal(mean1, cov1, 1)

        sample_result.append(result)
    return sample_result

def sample_procedure(total_samples, threshold,  mean0, mean1, cov0, cov1, samples=None):
    if samples!=None:
        sample_result=get_samples(total_samples, threshold,mean0, mean1, cov0, cov1)
    else:
        sample_result=samples

    TPR=[]
    FPR=[]
    decision_thresholds=[]
    empirical_error=[]
    multivariate0= multivariate_normal(mean=mean0, cov=cov0)
    multivariate1= multivariate_normal(mean=mean1, cov=cov1)
    decision_threshold = 0.54
    TP, TN, FP, FN = classifier(decision_threshold, multivariate0, multivariate1, sample_result)
    decision_thresholds.append(decision_threshold)
    empirical_error.append((FP + FN) / total_samples)
    TPR.append(TP / (TP + FN))
    FPR.append(FP / (FP + TN))

    decision_threshold = 0
    while(decision_threshold<10000000):
        TP, TN, FP, FN = classifier(decision_threshold, multivariate0, multivariate1, sample_result)
        decision_thresholds.append(decision_threshold)
        if decision_threshold == 0:
            decision_threshold += 0.000001
        empirical_error.append((FP + FN) / total_samples)
        decision_threshold *= 1.05
        print(decision_threshold)
        TPR.append(TP / (TP + FN))
        FPR.append(FP / (FP + TN))


    theoretical_indice = 0
    min_indice = empirical_error.index(min(empirical_error))

    plt.plot(FPR, TPR)
    plt.scatter(FPR[theoretical_indice], TPR[theoretical_indice], color='blue', label='theoretical point')
    plt.scatter(FPR[min_indice], TPR[min_indice], color='red', label='min point')
    print(FPR[min_indice], TPR[min_indice])
    print(FPR)
    print(TPR)
    plt.title('ROC')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.show()

    plt.plot(decision_thresholds, empirical_error)
    plt.scatter(decision_thresholds[theoretical_indice], empirical_error[theoretical_indice], color='blue', label='theoretical point')
    plt.scatter(decision_thresholds[min_indice], empirical_error[min_indice], color='red', label='min point')
    print(decision_thresholds[theoretical_indice], empirical_error[theoretical_indice])
    print(decision_thresholds[min_indice], empirical_error[min_indice])
    plt.title('ERROR CURVE')
    plt.xlabel('y')
    plt.ylabel('empirical_error')
    plt.legend()
    plt.show()

    return sample_result



def Chernoff_bound():
    total_samples, threshold, mean0, mean1, cov0, cov1 = configureA()
    beta=0
    new_mean = mean1 - mean0
    results=[]
    betas=[]
    while(beta<1):
        new_cov = beta*cov0 + (1-beta)*cov1
        k_beta = ((beta * (1 - beta)) / 2)
        scalar1 = k_beta*np.dot(new_mean.transpose(), np.dot(np.linalg.inv(new_cov), new_mean))
        det_cov0 = np.power(np.linalg.det(cov0), beta)
        det_cov1 = np.power(np.linalg.det(cov1), 1-beta)
        det_new_cov = np.linalg.det(new_cov)
        scalar2= 0.5*np.log(det_new_cov/(det_cov0*det_cov1))
        result = scalar1 + scalar2
        beta+=0.005
        results.append(np.power(0.35, beta)*np.power(0.65, 1-beta)*np.power(math.e, -result))
        betas.append(beta)

    min_indice = results.index(min(results))
    B_indice=99
    plt.plot(betas, results)
    plt.scatter(betas[min_indice], results[min_indice], color='red', label='min point')
    plt.scatter(betas[B_indice], results[B_indice], color='blue', label='hattacharyya bounds')
    plt.title('Chernoff_bound')
    plt.xlabel('beta')
    plt.ylabel('bounds')
    plt.legend()
    plt.show()

    print(betas[min_indice], results[min_indice])
    print(betas[ B_indice], results[ B_indice])


Chernoff_bound()



def sampling_process():
    total_samples, threshold, mean0, mean1, cov0, cov1 = configureA()
    sample_result = sample_procedure(total_samples, threshold, mean0, mean1, cov0, cov1)

    multivariate0 = multivariate_normal(mean=mean0, cov=cov0)
    multivariate1 = multivariate_normal(mean=mean1, cov=cov1)

    def error_f(x1, x2, x3, x4):
        x = np.array([x1, x2, x3, x4])
        prob0 = multivariate0.pdf(x)
        prob1 = multivariate1.pdf(x)
        return min(prob0 * 0.35, prob1 * 0.65)


    print(integrate.nquad(error_f, [[-5, 5],[-5, 5], [-5, 5], [-5, 5]],
            opts = {'epsabs': 1.49e-3, 'epsrel': 1.49e-3, 'limit': 100}))


    total_samples, threshold, mean0, mean1, cov0, cov1 = configureC()
    sample_procedure(total_samples, threshold, mean0, mean1, cov0, cov1, samples=sample_result)
    multivariate0 = multivariate_normal(mean=mean0, cov=cov0)
    multivariate1 = multivariate_normal(mean=mean1, cov=cov1)

    print(integrate.nquad(error_f, [[-5, 5], [-5, 5], [-5, 5], [-5, 5]], opts = {'epsabs': 1.49e-3, 'epsrel': 1.49e-3, 'limit': 100}))

    centralized_sample=[]
    for sample in sample_result:
        if sample['label']==0:
            centralized_sample.append(sample['samples'] - mean0)
        else:
            centralized_sample.append(sample['samples'] - mean1)

    nums = len(centralized_sample)
    centralized_sample= np.squeeze(np.array(centralized_sample))
    new_cov=np.dot(centralized_sample.transpose(), centralized_sample)/nums

    mean0=mean0
    mean1=mean1
    cov0=new_cov
    cov1=new_cov
    sample_procedure(total_samples, threshold, mean0, mean1, cov0, cov1, samples=sample_result)
    multivariate0 = multivariate_normal(mean=mean0, cov=cov0)
    multivariate1 = multivariate_normal(mean=mean1, cov=cov1)

    print(integrate.nquad(error_f, [[-5, 5], [-5, 5], [-5, 5], [-5, 5]], opts = {'epsabs': 1.49e-3, 'epsrel': 1.49e-3, 'limit': 100}))

    


def y_function():
    B=0.01
    y=0.35/(B*0.65)
    B_List=[B]
    y_List=[y]

    total_samples, threshold, mean0, mean1, cov0, cov1 = configureA()
    sample_result = get_samples(total_samples, threshold,mean0, mean1, cov0, cov1)
    multivariate0 = multivariate_normal(mean=mean0, cov=cov0)
    multivariate1 = multivariate_normal(mean=mean1, cov=cov1)
    def error_f():

        TP, FP, TN, FN=0, 0, 0, 0
        for i in sample_result:
            random_vector = i['samples']
            prob0 = multivariate0.pdf(random_vector)
            prob1 = multivariate1.pdf(random_vector)

            if (prob1 / prob0) > y:
                pred = 1
                if i['label'] == 1:
                    TP += 1
                else:
                    FP += 1
            else:
                pred = 0
                if i['label'] == 1:
                    FN += 1
                else:
                    TN += 1

        return (FP + FN) / total_samples
    error=error_f()
    error_List=[error]
    print(error)
    while(B<5):
        B+=0.05
        B_List.append(B)
        y=0.35/(B*0.65)
        y_List.append(y)
        error=error_f()
        error_List.append(error)
        print(error)


    plt.plot(B_List,y_List)
    plt.title('y curve')
    plt.xlabel('B')
    plt.ylabel('y')
    plt.legend()
    plt.show()

    plt.plot(B_List, error_List)
    plt.title('error curve')
    plt.xlabel('B')
    plt.ylabel('error')
    plt.legend()
    plt.show()

y_function()