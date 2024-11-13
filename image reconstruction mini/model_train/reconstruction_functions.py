import bart
import numpy as np


# perform inverse fourier transform
def fourier_transform(kspace):
    """This function reconstrucs an image using the fourier transform. """
    dim1 = 0
    dim2 = 1
    # dofftshift and ifftshift because data is spread over four corners
    image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace, axes=(dim1, dim2)),
                    axes=(dim1, dim2)), axes=(dim1, dim2))
    return image


def compressed_sensing(kspace, sensitivity_maps, device='cpu', reg_wt=0.05, num_iters=60):
    """ This function reconstructs an undersample image using BART.
    image: undersampled image"""
    if device == 'gpu':
        # l1 wavelet regularization
        recon = bart.bart(1, f"pics -g -S -R W:6:0:{reg_wt} -i {num_iters}", kspace, sensitivity_maps)

    else:
        # l1 wavelet regularization
        recon = bart.bart(1, f"pics -S -R W:6:0:{reg_wt} -i {num_iters}", kspace, sensitivity_maps)
    return recon

