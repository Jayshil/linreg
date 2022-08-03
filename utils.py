import numpy as np


def gls(des, y, yerr):
    """
    GLS estimate of weights and covariance matrix
    """
    sig = np.diag(yerr**2)
    sig1 = np.linalg.inv(sig)
    cov = np.linalg.inv(des.T @ sig1 @ des)
    bet = cov @ des.T @ sig1 @ y
    return bet, cov

def des_mat(deg, roll):
    """
    Given the degree and roll, this function will 
    generate the designe matrix made up of sin and
    cosine of roll angle upto provided degree.
    """
    ab = np.ones(len(roll))
    for i in range(deg):
        cos1 = np.cos(np.deg2rad((i+1)*roll))
        sin1 = np.sin(np.deg2rad((i+1)*roll))
        ab = np.vstack((ab, sin1))
        ab = np.vstack((ab, cos1))
    ab = np.transpose(ab)
    return ab