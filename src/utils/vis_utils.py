import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt

# label_to_color = {
#     0: (0, 0, 0),         # 'noise'
#     1: (112, 128, 144),   # Slategrey       # 'barrier'
#     2: (220, 20, 60),     # Crimson         # 'bicycle'
#     3: (255, 127, 80),    # Coral           # 'bus'
#     4: (255, 158, 0),     # Orange          # 'car'
#     5: (233, 150, 70),    # Darksalmon      # 'construction_vehicle'
#     6: (255, 61, 99),     # Red             # 'motorcycle'
#     7: (0, 0, 230),       # Blue            # 'pedestrian'
#     8: (47, 79, 79),      # Darkslategrey   # 'traffic_cone'
#     9: (255, 140, 0),     # Darkorange      # 'trailer'
#     10: (255, 99, 71),    # Tomato          # 'truck'
#     11: (0, 207, 191),    # nuTonomy green  # 'driveable_surface'
#     12: (175, 0, 75),                       # 'other_flat'
#     13: (75, 0, 75),                        # 'sidewalk'
#     14: (112, 180, 60),                     # 'terrain'
#     15: (222, 184, 135),  # Burlywood       # 'manmade'
#     16: (0, 175, 0),      # Green           # 'vegetation'
#     17: (255, 255, 255),  # white           # 'free'
# }
label_to_color = {
    0: (0, 0, 0),         # 'noise'
    1: (255, 120, 50),    # 'barrier'
    2: (255, 192, 203),   # 'bicycle'
    3: (255, 255, 0),     # 'bus'
    4: (0, 150, 245),     # 'car'
    5: (0, 255, 255),     # 'construction_vehicle'
    6: (200, 180, 0),     # 'motorcycle'
    7: (255, 0, 0),       # 'pedestrian'
    8: (255, 240, 150),   # 'traffic_cone'
    9: (135, 60, 0),      # 'trailer'
    10: (160, 32, 240),   # 'truck'
    11: (255, 0, 255),    # 'driveable_surface'
    12: (139, 137, 137),  # 'other_flat'
    13: (75, 0, 75),      # 'sidewalk'
    14: (150, 240, 80),   # 'terrain'
    15: (213, 213, 213),  # 'manmade'
    16: (0, 175, 0),      # 'vegetation'
    17: (255, 255, 255),  # white           # 'free'
}
def visualize_depth(depth, min=None, max=None, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    x = depth.cpu().detach().numpy()
    x = np.nan_to_num(x) # change nan to 0
    if min:
        mi = min
    else:
        mi = np.min(x) # get minimum depth
    if max:
        ma = max
    else:
        ma = np.max(x)
    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_) # (3, H, W)
    return x_

def visualize_semantic(semantic):
    """
    semantic: (H, W)
    """
    h, w = semantic.shape
    x = semantic.cpu().detach().numpy()
    x = np.nan_to_num(x)
    # x = x / 17
    # x = (255 * x).astype(np.uint8)
    a = np.vectorize(label_to_color.__getitem__)(x.reshape(-1))
    b = np.concatenate((a[0][:, None], a[1][:, None], a[2][:, None]), axis=-1)
    x = b.reshape(h, w, 3).astype(np.uint8)
    x_ = Image.fromarray(x)
    x_ = T.ToTensor()(x_)
    return x_

def visualize_geomxyz(geom_xyz, mode='bev', norm=False):
    """
    Args:
    geom_xyz (Tensor): bx6xDxHxWx3.
    """
    _, ax = plt.subplots(1, 1, figsize=(9, 9))
    pa = geom_xyz[0].reshape(6, -1, 3).cpu().detach().numpy()
    if mode == 'bev':
        ax.scatter(pa[0, :, 0], pa[0, :, 1], c='orange', s=0.2)
        ax.scatter(pa[1, :, 0], pa[1, :, 1], c='green', s=0.2)
        ax.scatter(pa[2, :, 0], pa[2, :, 1], c='blue', s=0.2)
        ax.scatter(pa[3, :, 0], pa[3, :, 1], c='red', s=0.2)
        ax.scatter(pa[4, :, 0], pa[4, :, 1], c='yellow', s=0.2)
        ax.scatter(pa[5, :, 0], pa[5, :, 1], c='pink', s=0.2)
        if norm:
            ax.set_xlim(-1., 1.)
            ax.set_ylim(-1., 1.)
            ax.plot(-1., -1., 'x', color='red')
            ax.plot(-1., 1., 'x', color='red')
            ax.plot(1., -1., 'x', color='red')
            ax.plot(1., 1., 'x', color='red')
        else:
            ax.set_xlim(-51.2, 51.2)
            ax.set_ylim(-51.2, 51.2)
            ax.plot(-51.2, -51.2, 'x', color='red')
            ax.plot(-51.2, 51.2, 'x', color='red')
            ax.plot(51.2, -51.2, 'x', color='red')
            ax.plot(51.2, 51.2, 'x', color='red')
    elif mode == 'front':
        # ax.invert_xaxis()
        ax.scatter(pa[0, :, 1], pa[0, :, 2], c='orange', s=0.2)
        ax.scatter(pa[1, :, 1], pa[1, :, 2], c='green', s=0.2)
        ax.scatter(pa[2, :, 1], pa[2, :, 2], c='blue', s=0.2)
        # ax.scatter(pa[3, :, 1], pa[3, :, 2], c='red', s=0.2)
        # ax.scatter(pa[4, :, 1], pa[4, :, 2], c='yellow', s=0.2)
        # ax.scatter(pa[5, :, 1], pa[5, :, 2], c='pink', s=0.2)
        if norm:
            ax.set_xlim(-1., 1.)
            ax.set_ylim(-1., 1.)
        else:
            ax.set_xlim(-51.2, 51.2)
            ax.set_ylim(-5., 5.)
        ax.invert_xaxis()
    ax.plot(0, 0, 'x', color='red')
    ax.axis('off')
    plt.show()


# if __name__ == "__main__":
#     import torch
#     a = torch.arange(17).expand(10, 17)
#     b = visualize_semantic(a)
#     plt.imshow(b.permute(1, 2, 0).cpu().detach().numpy())
#     plt.show()
#     print(1)