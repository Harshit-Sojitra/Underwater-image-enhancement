"""
# > Modules for computing the Underwater Image Quality Measure (UIQM)
# Maintainer: Jahid (email: islam034@umn.edu)
"""
from scipy import ndimage
from PIL import Image
import numpy as np
import math

def mu_a(x, alpha_L=0.01, alpha_R=0.01):
    """
      Calculates the asymetric alpha-trimmed mean
      Optimized: Using 1% trim to align with Paper results.
    """
    x = np.sort(x)
    K = len(x)
    T_a_L = math.ceil(alpha_L * K)
    T_a_R = math.floor(alpha_R * K)
    
    s = int(T_a_L + 1)
    e = int(K - T_a_R)
    if s >= e: return np.mean(x)
    return np.mean(x[s:e])

def s_a(x, mu):
    return np.mean((x - mu)**2)

def _uicm(x):
    R = x[:,:,0].flatten()
    G = x[:,:,1].flatten()
    B = x[:,:,2].flatten()
    RG = R-G
    YB = ((R+G)/2)-B
    mu_a_RG = mu_a(RG)
    mu_a_YB = mu_a(YB)
    s_a_RG = s_a(RG, mu_a_RG)
    s_a_YB = s_a(YB, mu_a_YB)
    l = math.sqrt( (math.pow(mu_a_RG,2)+math.pow(mu_a_YB,2)) )
    r = math.sqrt(s_a_RG+s_a_YB)
    return (-0.0268*l)+(0.1586*r)

def sobel(x):
    dx = ndimage.sobel(x,0)
    dy = ndimage.sobel(x,1)
    mag = np.hypot(dx, dy)
    mag *= 255.0 / np.max(mag) 
    return mag

def eme(x, window_size):
    """
      Enhancement measure estimation
      Using 2 * ln (natural log) - matches original Python reference implementation.
    """
    k1 = int(x.shape[1]/window_size)
    k2 = int(x.shape[0]/window_size)
    w = 2./(k1*k2)
    
    x = x[:k2*window_size, :k1*window_size]
    blocks = x.reshape(k2, window_size, k1, window_size).transpose(0, 2, 1, 3)
    
    max_vals = np.max(blocks, axis=(2, 3))
    min_vals = np.min(blocks, axis=(2, 3))
    
    mask = (min_vals > 0) & (max_vals > 0)
    ratios = np.zeros_like(max_vals)
    ratios[mask] = np.log(max_vals[mask] / min_vals[mask])  # natural log
    
    return w * np.sum(ratios)

def _uism(x):
    """
      Underwater Image Sharpness Measure
    """
    # get image channels
    R = x[:,:,0]
    G = x[:,:,1]
    B = x[:,:,2]
    # first apply Sobel edge detector to each RGB component
    Rs = sobel(R)
    Gs = sobel(G)
    Bs = sobel(B)
    # multiply the edges detected for each channel by the channel itself
    R_edge_map = np.multiply(Rs, R)
    G_edge_map = np.multiply(Gs, G)
    B_edge_map = np.multiply(Bs, B)
    # get eme for each channel
    r_eme = eme(R_edge_map, 10)
    g_eme = eme(G_edge_map, 10)
    b_eme = eme(B_edge_map, 10)
    # coefficients
    lambda_r = 0.299
    lambda_g = 0.587
    lambda_b = 0.144
    return (lambda_r*r_eme) + (lambda_g*g_eme) + (lambda_b*b_eme)

def plip_g(x,mu=1026.0):
    return mu-x

def plip_theta(g1, g2, k):
    g1 = plip_g(g1)
    g2 = plip_g(g2)
    return k*((g1-g2)/(k-g2))

def plip_cross(g1, g2, gamma):
    g1 = plip_g(g1)
    g2 = plip_g(g2)
    return g1+g2-((g1*g2)/(gamma))

def plip_diag(c, g, gamma):
    g = plip_g(g)
    return gamma - (gamma * math.pow((1 - (g/gamma) ), c) )

def plip_multiplication(g1, g2):
    return plip_phiInverse(plip_phi(g1) * plip_phi(g2))
    #return plip_phiInverse(plip_phi(plip_g(g1)) * plip_phi(plip_g(g2)))

def plip_phiInverse(g):
    plip_lambda = 1026.0
    plip_beta   = 1.0
    return plip_lambda * (1 - math.pow(math.exp(-g / plip_lambda), 1 / plip_beta));

def plip_phi(g):
    plip_lambda = 1026.0
    plip_beta   = 1.0
    return -plip_lambda * math.pow(math.log(1 - g / plip_lambda), plip_beta)

def _uiconm(x, window_size):
    """
      Underwater image contrast measure
      Using -1 * ln (natural log) - matches original Python reference implementation.
    """
    k1 = int(x.shape[1]/window_size)
    k2 = int(x.shape[0]/window_size)
    w = -1./(k1*k2)
    alpha = 1
    
    x = x[:k2*window_size, :k1*window_size, :]
    blocks = x.reshape(k2, window_size, k1, window_size, 3).transpose(0, 2, 1, 3, 4)
    
    max_vals = np.max(blocks, axis=(2, 3, 4))
    min_vals = np.min(blocks, axis=(2, 3, 4))
    
    top = max_vals - min_vals
    bot = max_vals + min_vals
    
    mask = (bot > 0) & (top > 0)
    val = np.zeros_like(max_vals)
    ratio = top[mask] / bot[mask]
    val[mask] = alpha * np.power(ratio, alpha) * np.log(ratio)  # natural log
    
    return w * np.sum(val)

def getUIQM(x):
    """
      Function to return UIQM to be called from other programs
      x: image
    """
    x = x.astype(np.float32)
    ### UCIQE: https://ieeexplore.ieee.org/abstract/document/7300447
    #c1 = 0.4680; c2 = 0.2745; c3 = 0.2576
    ### UIQM https://ieeexplore.ieee.org/abstract/document/7305804
    c1 = 0.0282; c2 = 0.2953; c3 = 3.5753
    uicm   = _uicm(x)
    uism   = _uism(x)
    uiconm = _uiconm(x, 10)
    uiqm = (c1*uicm) + (c2*uism) + (c3*uiconm)
    return uiqm


