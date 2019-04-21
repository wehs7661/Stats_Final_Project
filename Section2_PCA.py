import numpy as np
import pandas as pd
import mdtraj as mdt
import mdtraj as md
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from matplotlib import rc

traj = mdt.load('Section 2/Input files for Python analysis/md1_backbone.xtc', \
                top='Section 2/Input files for Python analysis/ref.pdb')   # supply topology information using .pdb file
# traj = <mdtraj.Trajectory with 1842 frames, 486 atoms, 162 residues, and unitcells at 0x1c5756139e8>
top = traj.topology        # <mdtraj.Topology with 1 chains, 162 residues, 486 atoms, 485 bonds at 0x23d949317b8>
traj.superpose(traj, 0)    # We use the initial structure as the reference

xyz = traj.xyz            # traj.xyz output the cartesian coordinates of backbone at each time frame (from 0th to 2000th)
pca_input = xyz.reshape(traj.n_frames, traj.n_atoms*3)   # transform to the trajectory of all coodinates



pca = PCA(n_components=10)                            # here n_features = 1458 (=486 x 3)
reduced_coordinates = pca.fit_transform(pca_input)    # dimension reduction of the backbone trajectory (projected to first 10 PCs)
per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)  # convert into percentages
accum_per_var = [i for i in [np.sum(per_var[:j]) for j in range(1,11)]] # accumulated values of per_var


# Set the global font to be DejaVu Sans, size 10 (or any other sans-serif font of your choice!)
rc('font',**{'family':'sans-serif','sans-serif':['DejaVu Sans'],'size':10})
# Set the font used for MathJax - more on this later
rc('mathtext',**{'default':'regular'})
plt.rc('font', family='serif')

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18,5.5))

# Scree plot
labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]

ax1 = plt.subplot(1,2,1)
ax2 = ax1.twinx()
ax1.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels, alpha = 0.85)
ax2.plot(range(1,len(per_var)+1), accum_per_var, color = 'r', marker = 'o')
xlocs, xlabs = plt.xticks()

# adding value labels
for i, v in enumerate(per_var):
    ax1.text(xlocs[i] - 0.35, v+0.5, str(v)+'%')

for i, v in enumerate(accum_per_var):
    ax2.text(xlocs[i] - 0.25, v+4, "{0:.1f}%".format(v))
    
ax1.set_ylabel('Percentage of explained variance (%)', size='12')
ax2.set_ylabel('Accumulated explained variance (%)', size='12')
ax1.set_xlabel('Principal Components', size='12')
ax1.set_ylim([0,27])
ax2.set_ylim([0,110])
plt.title('Scree Plot', size='14')
plt.grid(True)

# Scatter plot  
plt.subplot(1,2,2)
plt.scatter(reduced_coordinates[:, 0], reduced_coordinates[:,1], marker='x', c=traj.time/1000) # plot as a function PC1, PC2
plt.xlabel("PC1 ({0:.2f}%)".format(per_var[0]), size='12')
plt.ylabel("PC2 ({0:.2f}%)".format(per_var[1]), size='12')
plt.title('Cartesian coordinate PCA:  Bacteriophage T4 Lysozyme', size='14')
plt.grid(True)
cbar = plt.colorbar()
cbar.set_label('Time [ns]', size='12')

plt.savefig('Section2_PCA.png', dpi=600)