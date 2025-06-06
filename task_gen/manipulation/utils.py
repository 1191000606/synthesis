import numpy as np

def parse_center(center):   
    if center.startswith("(") or center.startswith("["):
        center = center[1:-1]

    center = center.split(",")
    center = [float(x) for x in center]
    return np.array(center)