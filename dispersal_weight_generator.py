import pickle
import numpy as np
class dispersal_weight_generator:
    def __init__(self):
        self.foo = 0
        self.c = 1 # scaling constant for characteristic distance
        self.target_mean_incoming = 2 # target mean incoming connectivity for normalization

    def uniform_dist(self, patchnum, kerneltype='exponential',normalization=0):
        '''
        get coordinates of patches and calculate dispersal weights between patches based on the distance and kernel type.
        '''
        # get 2D coordinates for patches
        patch_coords = np.random.rand(patchnum, 2) # random coordinates for patches
        # calculate distance matrix between patches
        dist_matrix = np.linalg.norm(patch_coords[:, np.newaxis] - patch_coords, axis=-1)
        # calculate dispersal weights 
        l = self.c/np.sqrt(patchnum) # characteristic distance for dispersal, where c is a scaling constant
        if kerneltype == 'exponential':
            weights = np.exp(-dist_matrix/l) # exponential decay of weights with distance
        elif kerneltype == 'gaussian':
            weights = np.exp(-dist_matrix**2/(2*l**2)) # gaussian decay of weights with distance
        else:
            raise ValueError('Invalid kernel type. Choose "exponential" or "gaussian".')
        
        # remove self-dispersal if desired
        np.fill_diagonal(weights, 0)

        # normalize weights
        if normalization == 1: # row-stochastic normalization
            weights = weights / np.sum(weights, axis=1, keepdims=True) # normalize weights by row
        elif normalization == 2: # fixed mean incoming connectivity
            incoming = np.sum(weights, axis=0)
            mean_incoming = np.mean(incoming)
            if mean_incoming > 0:
                scale = self.target_mean_incoming / mean_incoming
                weights = weights * scale

        return weights, patch_coords
    




if __name__ == "__main__":
    dispgen = dispersal_weight_generator()
    # generate 100 weights for patch size 4:22
    for patchnum in range(4,23):
        weights_collection = []
        coord_collection = []
        for i in range(100):
            weight, patch_coords = dispgen.uniform_dist(patchnum=patchnum, kerneltype='exponential', normalization=2)
            weights_collection.append(weight)
            coord_collection.append(patch_coords)
        with open(f'./dispersal_weights/uniform_dispersal_weights_patchnum{patchnum}.pkl', 'wb') as f:
            pickle.dump(weights_collection, f)
        with open(f'./dispersal_weights/uniform_dispersal_coords_patchnum{patchnum}.pkl', 'wb') as f:
            pickle.dump(coord_collection, f)
            
