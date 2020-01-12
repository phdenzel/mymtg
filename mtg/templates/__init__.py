import os
import numpy as np
import cv2
from cairosvg import svg2png


__all__ = ["svgs", 'svg2img']

root = os.path.dirname(__file__)


svg_0 = os.path.join(root, '0.svg')
svg_1 = os.path.join(root, '1.svg')
svg_2 = os.path.join(root, '2.svg')
svg_3 = os.path.join(root, '3.svg')
svg_4 = os.path.join(root, '4.svg')
svg_5 = os.path.join(root, '5.svg')
svg_6 = os.path.join(root, '6.svg')
svg_7 = os.path.join(root, '7.svg')
svg_8 = os.path.join(root, '8.svg')
svg_9 = os.path.join(root, '9.svg')
svg_10 = os.path.join(root, '10.svg')
svg_11 = os.path.join(root, '11.svg')
svg_12 = os.path.join(root, '12.svg')
svg_13 = os.path.join(root, '13.svg')
svg_14 = os.path.join(root, '14.svg')
svg_15 = os.path.join(root, '15.svg')
svg_W = os.path.join(root, 'W.svg')
svg_W_old = os.path.join(root, 'W_old.svg')
svg_U = os.path.join(root, 'U.svg')
svg_B = os.path.join(root, 'B.svg')
svg_BP = os.path.join(root, 'BP.svg')
svg_R = os.path.join(root, 'R.svg')
svg_G = os.path.join(root, 'G.svg')

svgs = {
    "0": svg_0, "1": svg_1, "2": svg_2, "3": svg_3,
    "4": svg_4, "5": svg_5, "6": svg_6, "7": svg_7,
    "8": svg_8, "9": svg_9, "10": svg_10, "11": svg_11,
    "12": svg_12, "13": svg_13, "14": svg_14, "15": svg_15,
    "W": svg_W, "W_old": svg_W_old, "U": svg_U,
    "B": svg_B, "BP": svg_BP, "R": svg_R, "G": svg_G}


def svg2img(filename):
    with open(filename, 'rb') as f:
        svg = svg2png(file_obj=f)
    nparr = np.frombuffer(svg, np.float32)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

