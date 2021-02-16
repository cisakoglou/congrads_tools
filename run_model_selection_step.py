
import os, subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nilearn
import seaborn as sns

import defs

## PART 1. Run for all subjects (that have not been exluded)
# This will lead to every subject folder containing two subfolders one for each hemisphere
#  and in each of these subfolders one folder with characteristics for each model order will be created

MAX_NO_ORDER = 12 # select the max model order for you which you want  to check

average_pre_path = 'outputs/average_n404/'
roi_l_img = nilearn.image.smooth_img(defs.ROIS_RS + 'roi_left_adapted_all.nii.gz', fwhm=None)
roi_l = roi_l_img.get_data()

#right
roi_r_img = nilearn.image.smooth_img(defs.ROIS_RS + 'roi_right_adapted_all.nii.gz', fwhm=None)
roi_r = roi_r_img.get_data()

#
for hemisphere in ['right', 'left']:
    for no_order in range(1, MAX_NO_ORDER):
        print(no_order)
        connectopy = defs.CONGRADS_OUTPUT_W1 + average_pre_path + 'roi_'+ hemisphere + '_adapted_all.cmaps.nii.gz'
        print(connectopy)
        if(os.path.exists(defs.CONGRADS_OUTPUT_W1 + average_pre_path + "/TSM_" + hemisphere + "/Order_" + str(no_order)) == False):
            os.makedirs(defs.CONGRADS_OUTPUT_W1 + average_pre_path + "/TSM_" + hemisphere + "/Order_" + str(no_order))
        cmd = ['cd', '/home/mrstats/chrisa/CODE/congrads;fslpython trendsurf.py',#source activate /project/3022014.02/chrisa/conda_envs/my_ext_root/;f
               '-i', connectopy,
               '-r', defs.ROIS_RS + 'roi_' + hemisphere + '_adapted_all.nii.gz',
               '-o', defs.CONGRADS_OUTPUT_W1 + average_pre_path + "TSM_" + hemisphere + "/Order_" + str(no_order),
               '-b', str(no_order)]
        cmd_cluster = ['echo', "\""] + cmd + ["\"", "|", "qsub", '-l', 'procs=1,mem=50gb,walltime=06:00:00',
                                              '-N', 'TSM_' + hemisphere + '_average' ,
                                              '-o', '/home/mrstats/chrisa/CODE/torque_logs/congrads',
                                              '-e', '/home/mrstats/chrisa/CODE/torque_logs/congrads']
        subprocess.call(' '.join(cmd_cluster), shell=True)

## PART 2. Create figure for model selection
for hemisphere in ['left','right']:
    nll = []
    bic = []
    ev = []
    scores_mixed_average = pd.DataFrame(columns=["Order", "BIC", "EV", "NLL"])
    if (hemisphere == 'left'):
        no_voxels = np.count_nonzero(roi_l == 1)  # len(np.where(roi_left !=0)[0])
    elif (hemisphere == 'right'):
        no_voxels = np.count_nonzero(roi_r == 1)
    for no_order in range(1, MAX_NO_ORDER):
        no_coefs = 3 * no_order + 1
        nll_file = defs.CONGRADS_OUTPUT_W1 + average_pre_path + 'TSM_' + hemisphere + '/Order_' + str(no_order) + \
                   '/roi_' + hemisphere + '_adapted_all.cmaps.tsm.negloglik.txt'

        nll_i = pd.read_csv(nll_file, sep=" ", header=None).values[0][0]
        nll.append(np.float(nll_i))

        bic_i = np.log(no_voxels) * no_coefs + 2 * nll_i #
        bic.append(np.float(bic_i))

        ev_file = defs.CONGRADS_OUTPUT_W1 + average_pre_path + 'TSM_' + hemisphere + '/Order_' + str(no_order) + \
                  '/roi_' + hemisphere + '_adapted_all.cmaps.tsm.explainedvar.txt'
        ev_i = pd.read_csv(ev_file, sep=" ", header=None).values[0][0]
        ev.append(np.float(ev_i))
        scores_mixed_average.loc[len(scores_mixed_average)] = [no_order,bic_i, ev_i, nll_i]

    #ax1 = sns.lineplot(x="Order", y="NLL", data=scores_mixed_average)
    plt.figure()
    ax1 = sns.lineplot(x="Order", y="BIC", data=scores_mixed_average)
    ax2 = ax1.twinx()
    sns.lineplot(x="Order", y="EV", data=scores_mixed_average, ax=ax2, color='r')
    plt.show()
    plt.savefig(defs.CONGRADS_OUTPUT_W1 + average_pre_path + 'BIC_EV_' + hemisphere+ '.png')
    plt.close()
