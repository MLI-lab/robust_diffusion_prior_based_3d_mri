import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import fastmri
from tqdm import tqdm
import multiprocessing as mp

def import_bart(
    base_path : str
):
    global bart
    sys.path.insert(0, os.path.join(base_path, "python"))
    os.environ['TOOLBOX_PATH'] = base_path
    import bart

# from https://github.com/MLI-lab/untrained-motion-correction/blob/master/only_trans_more_states.ipynb
def _ecalib_cmd(ecalib_calib_size=None):
    cmd = 'ecalib -d0 -m1'
    if ecalib_calib_size is not None:
        cmd += f' -r{int(ecalib_calib_size)}'
    return cmd


def compute_sens_maps(masked_ksp, ecalib_calib_size=None):
    ### compute sensitivity maps
    masked_ksp = masked_ksp[...,0] + 1j*masked_ksp[...,1]
    sens_maps = bart.bart(1, _ecalib_cmd(ecalib_calib_size), np.array([np.moveaxis(masked_ksp.detach().cpu().numpy(),0,2)]))
    return np.moveaxis(sens_maps[0],2,0)

def compute_sens_maps_3d(masked_ksp, ecalib_calib_size=None):
    kspace_full_complex = torch.view_as_complex(masked_ksp)
    kspace_full_complex_np = kspace_full_complex.moveaxis(0, -1).cpu().numpy() # moved coil dim to last dim
    sens_maps = bart.bart(1, _ecalib_cmd(ecalib_calib_size), kspace_full_complex_np)
    return sens_maps

def coil_compress_kspace_3d(masked_ksp, virtual_coils=None):
    if virtual_coils is None:
        return masked_ksp
    virtual_coils = int(virtual_coils)
    if virtual_coils <= 0:
        return masked_ksp
    kspace_full_complex = torch.view_as_complex(masked_ksp.contiguous())
    kspace_full_complex_np = kspace_full_complex.moveaxis(0, -1).cpu().numpy()
    kspace_cc_np = bart.bart(1, f'cc -p {virtual_coils}', kspace_full_complex_np)
    kspace_cc = torch.from_numpy(np.moveaxis(kspace_cc_np.astype(np.complex64), -1, 0))
    return torch.view_as_real(kspace_cc).to(masked_ksp.device)

def compute_sens_maps_np(masked_ksp, ecalib_calib_size=None):
    ### compute sensitivity maps
    masked_ksp = masked_ksp[...,0] + 1j*masked_ksp[...,1]
    sens_maps = bart.bart(1, _ecalib_cmd(ecalib_calib_size), np.array([np.moveaxis(masked_ksp,0,2)]))
    return np.moveaxis(sens_maps[0],2,0)

def compute_sens_maps_mp(masked_ksp, pool_size=8):
    # assume (Z, coils, Y, X, 2)
    #device = masked_ksp.get_device()
    iterates = list(masked_ksp.cpu().numpy()) if torch.is_tensor(masked_ksp) else list(masked_ksp)
    return np.array(mp.Pool(pool_size).map(compute_sens_maps_np, iterates))

def compute_l1_wavelet_solution(kspace, sensmaps, reg_param=4e-4):
    kspace_full_complex = torch.view_as_complex(kspace)
    kspace_full_complex_np = kspace_full_complex.moveaxis(0, -1).cpu().numpy()
    result_np = bart.bart(1, f'pics -l1 -r{reg_param}', kspace_full_complex_np, sensmaps.cpu().squeeze().numpy())
    return torch.view_as_real(torch.from_numpy(result_np)).to(kspace.device)

#def compute_sens_maps(masked_ksp):
    #### compute sensitivity maps
    #masked_ksp = masked_ksp[...,0] + 1j*masked_ksp[...,1]
    ## format ()
    #sens_maps = bart.bart(1, f'ecalib -d0 -m1', np.array([np.moveaxis(masked_ksp.detach().cpu().numpy(),0,2)]))
    ## C, Y, X -> 
    #return np.moveaxis(sens_maps[0],2,0)

#def compute_sens_maps_sigpy(masked_ksp):
    ## shape: (Nc, W, H, 2)
    #Nc, W, H, C = masked_ksp.shape
    #masked_ksp = masked_ksp[...,0] + 1j*masked_ksp[...,1]
    #masked_ksp_complex_np = torch.view_as_complex(masked_ksp).numpy()
    #sens_maps = mr.app.EspiritCalib(masked_ksp_complex_np)
    ## shape (H, W, Nc, 1)
    #return sens_maps.view(Nc, W, H, 1)

def vis_sense_maps(sens_maps):
    fig = plt.figure(figsize=(40,40))
    for i,s in enumerate(sens_maps):
        ax = fig.add_subplot(6,6,i+1)
        ax.imshow(np.abs(s),'gray')
        ax.set_title('coil {}'.format(i+1),fontsize=28)
        ax.axis('off')
    plt.show()
    return fig