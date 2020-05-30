import shutil
import os
import matplotlib.pyplot as plt
import sys

# img_dir = "Pose-Transfer/gan_model_e400/images_best"
img_dir = sys.argv[1]

# epochs = ['001', '012', '023', '031', '099', '195', '250', '314', '400']
# ids = ['0188', '0763', '0267', '0577', '0292', '0925', '0993', '0137', '0260', '0725', '1070', '1362', '0092', '0694', '0089', '0155']

epochs = ['001', '012', '023', '031', '099', '195', '250', '314', '400']
ids = ['0188', '0725', '0267', '0577', '1362', '0089', '0092']

fig, axes = plt.subplots(len(epochs), len(ids))
for epoch_idx, epoch in enumerate(epochs):
    for idv_idx, idv in enumerate(ids):
        ax = axes[epoch_idx, idv_idx]
        filename = "epoch%s_%s.png" % (epoch, idv)
        filepath = os.path.join(img_dir, filename)
        ax.imshow(plt.imread(filepath))
        ax.axis('off')
        if epoch_idx == 0:
            ax.set_title(idv, fontsize=12)
        # if idv_idx == (len(ids) - 1):
        #     ax.text(320, 64, epoch, size=12, verticalalignment='center', rotation=270)

plt.margins(0,0)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, "stitch.png"))
plt.show()
