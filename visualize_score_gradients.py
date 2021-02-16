#imports
import os
import numpy as np
import nibabel as nib

#from koen's code
from bayesreg import BLR
np.seterr(invalid='ignore')

#functions
def save_nifti(data,filename,examplenii,maskIndices):

    # load example image
    ex_img = nib.load(examplenii)
    ex_img.shape
    dim = ex_img.shape[0:3]
    if len(data.shape) < 2:
        nvol = 1
        data = data[:, np.newaxis]
    else:
        nvol = int(data.shape[1])

    # write data
    array_data = np.zeros((np.prod(dim),nvol))
    array_data[maskIndices,:] = data
    array_data = np.reshape(array_data,dim+(nvol,))
    array_img = nib.Nifti1Image(array_data,ex_img.affine,ex_img.header)
    nib.save(array_img,filename)

def load_data(datafile, maskfile):
    dataImg = nib.load(datafile)
    data = dataImg.get_data()
    dim = data.shape
    if len(dim) <= 3:
        dim = dim + (1,)
    data = np.reshape(data, (np.prod(dim[0:3]), dim[3]))

    maskImg = nib.load(maskfile)
    mask = maskImg.get_data()
    mask = np.reshape(mask, (np.prod(dim[0:3])))
    maskIndices = np.where(mask == 1)[0]
    data = data[maskIndices, :]

    i, j, k = np.meshgrid(np.linspace(0, dim[0] - 1, dim[0]),
                          np.linspace(0, dim[1] - 1, dim[1]),
                          np.linspace(0, dim[2] - 1, dim[2]), indexing='ij')

    world = np.vstack((i.ravel(), j.ravel(), k.ravel(), np.ones(np.prod(i.shape), float))).T
    world = np.dot(world, dataImg.affine.T)[maskIndices, 0:3]


    return data, world, maskIndices

def create_basis(X,dimpoly):

    dimx = X.shape[1]
    print('Generating polynomial basis set of degree',str(dimpoly),'...')
    Phi = np.zeros((X.shape[0],X.shape[1]*dimpoly))
    colid = np.arange(0,dimx)
    for d in range(1, dimpoly+1):
        Phi[:,colid] = X**d
        colid += dimx

    return Phi

#---- INPUT NEEDED HERE

hemisphere = 'left'
scoreName = 'Vineland-II Daily Living'
filename = defs.CONGRADS_OUTPUT_HARIRI + 'outputs/test/roi_left_adapted_all.cmaps.nii.gz' #the average gradient
maskfile = defs.ROIS_RS + 'roi_' + hemisphere + '_adapted_all.nii.gz'
outdir = defs.CONGRADS_OUTPUT_HARIRI + 'outputs/test/reconstructions/' + scoreName
no_order = 6 #model order of tsm after model selection step
basis = no_order
ard = False
score_points = [50, 80, 110] # select the points across the symptom score scale on which you want to generate the reconstructions
coef_nos = [5, 11]

#-----UNTIL HERE

# load data
print("Processing data in",filename)
Y,X,maskIndices = load_data(filename,maskfile)
Y = np.round(10000*Y) / 10000  # truncate precision to avoid numerical probs
if len(Y.shape) == 1:
    Y = Y[:,np.newaxis]
N = Y.shape[1]

# standardize responses and covariates
mY = np.mean(Y,axis=0)
sY = np.std(Y,axis=0)
Yz = (Y-mY)/sY
mX = np.mean(X,axis=0)
sX = np.std(X,axis=0)
Xz = (X-mX)/sX

# create basis set and set starting hyperparamters
Phi = create_basis(Xz,basis)
if ard is True:
    hyp0 = np.zeros(Phi.shape[1]+1)
else:
    hyp0 = np.zeros(2)

# estimate the models
yhat = np.zeros_like(Yz)
ys2  = np.zeros_like(Yz)
nlZ  = np.zeros(N)
hyp  = np.zeros((N,len(hyp0)))
rmse = np.zeros(N)
ev   = np.zeros(N)
m    = np.zeros((N,Phi.shape[1]))
bs2  = np.zeros((N,Phi.shape[1]))

#for i in range(0, N):
i=0
print("Estimating model ",i+1,"of",N)
breg = BLR()
hyp[i,:] = breg.estimate(hyp0,Phi,Yz[:,i],'powell')
m[i,:] = breg.m
nlZ[i] = breg.nlZ

# compute marginal variances
bs2[i] = np.sqrt(np.diag(np.linalg.inv(breg.A)))

# compute predictions and errors
yhat[:,i],ys2[:,i] = breg.predict(hyp[i,:],Phi,Yz[:,i],Phi)
yhat[:,i] = yhat[:,i]*sY[i] + mY[i]
rmse[i] = np.sqrt(np.mean((Y[:,i]-yhat[:,i])**2))
ev[i] = 100*(1-(np.var(Y[:,i]-yhat[:,i])/np.var(Y[:,i])))

print("Variance explained =",ev[i],"% RMSE =",rmse[i])

print("Mean (std) variance explained =",ev.mean(),"(",ev.std(),")")
print("Mean (std) RMSE =",rmse.mean(),"(",rmse.std(),")")

print("Writing output ...")


try:
    os.mkdir(outdir)
except OSError:
    print ("Creation of the directory %s failed" % outdir)
else:
    print ("Successfully created the directory %s " % outdir)

out_base_name = outdir + "/" + 'init_reconstruction_' + hemisphere # filename.split('/')[-1].split('.nii')[0]
#np.savetxt(out_base_name + ".tsm.trendcoeffvar.txt", bs2, delimiter='\t', fmt='%5.8f')
save_nifti(yhat, out_base_name + '.tsm.yhat.nii.gz', filename, maskIndices) #filename should be the examble file used for creating a new nifti file


# utils.view_connectopy(defs.CONGRADS_OUTPUT_HARIRI, 'outputs/test/reconstructions/' + scoreName + '/init_reconstruction_left.tsm.yhat.nii.gz','')#,
#                       0, 'outputs/test/reconstructions/' + scoreName + '/init_reconstruction')

# generate the reconstructions that correspond to the score points across the symptom scale that you select

y_average = yhat
a = np.linalg.pinv([y_average[:, 0]])
i = 0 # only for first gradient
# slopes and intercepts from run_univariate.py
for tp in score_points:

    for (ii,coef_no) in enumerate(coef_nos):
        breg.m[coef_no] = (slopes[ii] * tp + intercepts[ii])

    # compute predictions and errors
    yhat[:,i],ys2[:,i] = breg.predict(hyp[i,:],Phi,Yz[:,i],Phi)
    yhat[:,i] = yhat[:,i]*sY[i] + mY[i]
    rmse[i] = np.sqrt(np.mean((Y[:,i]-yhat[:,i])**2))

    #get the residuals after fitting to the average reconstruction (e=g_i - m*pinv(m)) where i_{th} timepoint, m: average reconstruction
    yresiduals = yhat
    yresiduals[:, 0] = yhat[:, 0] - np.multiply(y_average[:, 0], np.transpose(a))

    # normalize to range 0-1
    yresiduals[:, 0] = np.divide(yresiduals[:, 0]-min(yresiduals[:, 0]), (max(yresiduals[:, 0]) - min(yresiduals[:, 0])))


    print("Writing output ...")
    out_base_name = outdir + "/" + 'tp' + str(tp) + '_fit_n_' + hemisphere # filename.split('/')[-1].split('.nii')[0]
    save_nifti(yresiduals, out_base_name + '.tsm.yhat.nii.gz', filename, maskIndices)

# for tp in score_points:
#     out_base_name = outdir + "/" + 'tp' + str(tp) + '_fit_n_' + hemisphere
#     utils.view_connectopy(out_base_name + '.tsm.yhat.nii.gz')