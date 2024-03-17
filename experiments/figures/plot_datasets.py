from matplotlib import pyplot as plt
import pytorch_lightning as L
import sys
sys.path.append("../..")
from shapelearningtheory.datasets import make_dataset

L.seed_everything(0)

def normalize_image(img):
    minimum = min(img.min(), 0.)
    maximum = max(img.max(), 1.0)
    return (img - minimum) / (maximum - minimum)


patterns = ["color", "striped"]
shapes = ["rectangles", "LvT"]
sizes = ["small", "large"]
classes = ["class 1", "class 2"]

rows = len(sizes) * len(classes)
cols = len(shapes) * len(patterns)

fig, ax = plt.subplots(rows, cols,
    figsize=(3*rows, 3*cols))

for s, shape in enumerate(shapes):
    for p, pattern in enumerate(patterns):
        for n, size in enumerate(sizes):
            data = make_dataset(shape, pattern, size, "standard")
            data.prepare_data()
            shapename = "Rectangles" if shape == "rectangles" else "L-or-T"
            n_imgs = len(data.val)
            idx = n_imgs // 4
            x = len(patterns) * s + p
            y = 2 * n
            ax[y][x].imshow(normalize_image(data.val[idx][0].permute(1,2,0)))
            ax[y+1][x].imshow(normalize_image(data.val[n_imgs//2+idx][0].permute(1,2,0)))
            if y == 0:
                ax[y][x].set_title(pattern + " " + shapename, fontsize="large", fontweight="bold")
            if x == 0:
                ax[y][x].set_ylabel(classes[0] + ", " + size, fontsize="large", fontweight="bold")
                ax[y+1][x].set_ylabel(classes[1] + ", " + size, fontsize="large", fontweight="bold")

# Formatting
# remove ticks
for a in ax.flat:
    a.set_xticks([])
    a.set_yticks([])
# show output
fig.savefig("datasets.png", bbox_inches="tight")