import numpy as np

def PCA(X, num_PCs=None) :
    '''
    Principal component analysis (matlab-style)

    :param X :       (samples)x(features) array of data
    :param num_PCs:  number of components to estract

    :returns:  pca_data   array of projected data (samples)x(components) 
               E          matrix of the eigenvectors
               d          eigenvalues 
    '''
    
    # control the number of components 
    # (if none given we get all of them) 
    if num_PCs is None:
        num_PCs = X.shape[1]
    
    # estract the mean from the dataset
    X -= X.mean(0).reshape(1, -1)
    # compute the covariance matrix  
    C = np.cov(X.T)
    
    # find eigenvalues and eigenvectors 
    d, E = np.linalg.eigh(C)
    
    # find indices of eigenvalues in descending sort order
    idcs = np.argsort(abs(d))
    # consequently sort the eigenvectors
    E = E[:, idcs[::-1]]
    # chose the first 'num_PCs' components
    E = E[:,:num_PCs]

    # project the original data on the chosen components
    pca_data = np.dot(X, E)

    return pca_data, E, d[idcs[::-1]][:num_PCs]

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    import mnist

    data = mnist.train_images()[:200]/255.0
    labels = mnist.train_labels()[:200]

    num_patterns, pattern_side, _ = data.shape
    pattern_len = pattern_side*pattern_side
    data = data.reshape(num_patterns, pattern_len)

    data_out, E, d = PCA(data, 2)
    
    plt.imshow(E[:,1].reshape(pattern_side, 
        pattern_side))
    plt.show()
    plt.hist(labels)
    plt.show()

