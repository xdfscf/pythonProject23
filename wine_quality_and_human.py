import csv
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.stats import multivariate_normal
# Replace 'your_file.csv' with the path to your CSV file
import numpy as np


def whiten(data):
    # Center the data
    centralize_dt = np.mean(data, axis=0)
    centralize_dt = data - centralize_dt

    # Calculate the covariance matrix
    covariance_matrix = np.cov(centralize_dt, rowvar=False)

    # Eigenvalue decomposition of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Whitening matrix
    whitening_matrix = np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T

    # Whiten the data
    X_whitened = centralize_dt @ whitening_matrix

    return X_whitened, whitening_matrix

def pca(data, num_components=3):
    # Center the data
    centralize_dt = np.mean(data, axis=0)
    centralize_dt = data - centralize_dt

    # Calculate the covariance matrix
    covariance_matrix = np.cov(centralize_dt, rowvar=False)

    # Eigenvalue decomposition of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Keep only the specified number of components
    if num_components is not None:
        eigenvectors = eigenvectors[:, :num_components]

    # Project the data onto the principal components
    X_pca = centralize_dt @ eigenvectors

    return X_pca

def read_csv_data(file_paths):
    # Open the CSV file and create a csv.reader object
    with open(file_paths[0], 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        # Iterate over each row in the CSV file

        all_variate = []
        all_labels = []
        counter = 0
        for row in csv_reader:
            counter += 1
            content = row[0]
            content = content.split(';')
            dt = np.array([float(i) for i in content])
            dt = dt[:-1]
            all_variate.append(dt)
            label = content[-1]
            all_labels.append(label)
    return counter, all_labels, all_variate

def read_txt_data(file_paths):
    # Open the CSV file and create a csv.reader object
    with open(file_paths[0], 'r') as file:
        all_variate = []
        counter = 0
        for line in file:
            counter += 1
            content = line.split()
            content = np.array([float(i) for i in content])
            all_variate.append(content)

    with open(file_paths[1], 'r') as file:
        all_labels = []
        for line in file:
            content = line.split()
            content = float(content[0])
            label = round(content)
            all_labels.append(label)

    return counter, all_labels, all_variate

def calculate_paras(all_labels, all_variate, counter):
    labeled_data = {}
    for index, label in enumerate(all_labels):
        if label not in labeled_data:
            labeled_data[label] = [all_variate[index]]
        else:
            labeled_data[label].append(all_variate[index])

    labeled_data = dict(sorted(labeled_data.items()))
    priori = [len(labeled_data[i]) / counter for i in labeled_data]
    priori = np.array(priori)

    count = 0
    sorted_labels = {}
    for i in labeled_data:
        sorted_labels[i] = count
        count += 1

    means = []
    covs = []
    multivariates = []
    for i in labeled_data:
        dt = np.array(labeled_data[i])
        num_dt = len(dt)
        mean = np.mean(dt, axis=0)
        means.append(mean)
        centralize_dt = dt - mean
        cov = np.dot(centralize_dt.transpose(), centralize_dt) / num_dt
        cov += 0.005 * np.eye(len(cov))
        covs.append(cov)
        multivariate = multivariate_normal(mean=mean, cov=cov)
        multivariates.append(multivariate)
    return priori, sorted_labels, labeled_data, multivariates

def calculate_confusion_matrix(labeled_data, all_variate, all_labels, sorted_labels, multivariates, priori):
    confusion_matrix = np.zeros((len(labeled_data), len(labeled_data)), dtype=int)

    correctly_classified = []
    Incorrectly_classified = []
    for index, dt in enumerate(all_variate):

        label = all_labels[index]
        label = sorted_labels[label]
        condition_prob = np.array([gaussian.pdf(dt) for gaussian in multivariates])
        joint_pos = condition_prob * priori
        max_index = np.argmax(joint_pos)
        confusion_matrix[label][max_index] += 1
        if (label == max_index):
            correctly_classified.append(index)
        else:
            Incorrectly_classified.append(index)
    print(confusion_matrix)
    return correctly_classified, Incorrectly_classified

def data_process(file_paths, file_type='csv', do_whiten = False, do_PCA = False):
    if(file_type=='csv'):
        counter, all_labels, all_variate=read_csv_data(file_paths)
    if(file_type=='txt'):
        counter, all_labels, all_variate=read_txt_data(file_paths)

    if do_whiten:
        all_variate = whiten(all_variate)

    priori, sorted_labels, labeled_data, multivariates = calculate_paras(all_labels, all_variate, counter)
    print('original confusion matrix')
    correctly_classified, Incorrectly_classified = calculate_confusion_matrix(labeled_data, all_variate, all_labels, sorted_labels, multivariates, priori)
    print('Incorrectly classified: ', len(Incorrectly_classified))
    print('error rate: ', len(Incorrectly_classified)/len(all_variate))


    if do_PCA:
        i=math.ceil(len(all_variate[0])/20)
        while(i<len(all_variate[0])):
            print('confusion matrix with {:} primary features'.format(i))
            all_variate_test_pca = pca(all_variate, i)
            priori, sorted_labels, labeled_data, multivariates = calculate_paras(all_labels, all_variate_test_pca, counter)
            test_correctly_classified, test_Incorrectly_classified = calculate_confusion_matrix(labeled_data, all_variate_test_pca,
                                                                                      all_labels, sorted_labels,
                                                                                      multivariates, priori)
            print('Incorrectly classified: ', len(test_Incorrectly_classified))
            print('error rate: ', len(test_Incorrectly_classified) / len(all_variate))
            print('\n')
            i+=math.ceil(len(all_variate[0])/20)

    if file_type == 'txt':
        all_variate_best_pca = pca(all_variate, 522)
        priori, sorted_labels, labeled_data, multivariates = calculate_paras(all_labels, all_variate_best_pca, counter)
        correctly_classified, Incorrectly_classified = calculate_confusion_matrix(labeled_data, all_variate_best_pca,
                                                                              all_labels, sorted_labels,
                                                                              multivariates, priori)
        print('Incorrectly classified: ', len(Incorrectly_classified))
        print('error rate: ', len(Incorrectly_classified) / len(all_variate))
        print('\n')


    all_variate_plot_pca = pca(all_variate)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Scatter plot for correctly classified points
    ax.scatter(all_variate_plot_pca[correctly_classified, 0], all_variate_plot_pca[correctly_classified, 1],
                   all_variate_plot_pca[correctly_classified, 2], c='green', label='Correctly Classified')

    # Scatter plot for incorrectly classified points
    ax.scatter(all_variate_plot_pca[Incorrectly_classified, 0],all_variate_plot_pca[Incorrectly_classified, 1],
                   all_variate_plot_pca[Incorrectly_classified, 2], c='red', label='Incorrectly Classified')

    ax.set_title('Data After PCA with Correctly and Incorrectly Classified Points')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.legend()

    plt.show()



file_paths1 = ['winequality-white.csv']
data_process(file_paths1, file_type='csv', do_whiten = False, do_PCA = False)

file_paths2=['X_train.txt', 'y_train.txt']
data_process(file_paths2, file_type='txt', do_whiten = False, do_PCA = False)