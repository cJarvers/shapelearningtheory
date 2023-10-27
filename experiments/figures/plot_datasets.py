from matplotlib import pyplot as plt
import sys
sys.path.append("../..")
from shapelearningtheory.datasets import make_rectangles_color, \
    make_rectangles_texture, make_LT_color, make_LT_texture
from shapelearningtheory.colors import RedXORBlue, NotRedXORBlue, RandomGrey
from shapelearningtheory.datasets.rectangledataset import RectangleDataset
from shapelearningtheory.datasets.LTPlusDataset import LTDataset
from shapelearningtheory.textures import HorizontalGrating, VerticalGrating

def normalize_image(img):
    minimum = min(img.min(), 0.)
    maximum = max(img.max(), 1.0)
    return (img - minimum) / (maximum - minimum)


datasets = {
    "color rectangles": make_rectangles_color(),
    "striped rectangles": make_rectangles_texture(),
    "color LvT": make_LT_color(),
    "striped LvT": make_LT_texture()
}

imgs_per_dataset = 3
fig, ax = plt.subplots(imgs_per_dataset, len(datasets),
    figsize=(3*len(datasets), 3*imgs_per_dataset))

for i, (name, data) in enumerate(datasets.items()):
    data.prepare_data()
    n_imgs = len(data.dataset)
    ax[0][i].imshow(data.dataset[0][0].permute(1,2,0))
    ax[1][i].imshow(data.dataset[n_imgs//2][0].permute(1,2,0))
    ax[2][i].imshow(data.dataset[-1][0].permute(1,2,0))
    ax[0][i].set_title(name, fontsize="large", fontweight="bold")

# Formatting
# remove ticks
for a in ax.flat:
    a.set_xticks([])
    a.set_yticks([])
# show output
fig.savefig("datasets.png", bbox_inches="tight")