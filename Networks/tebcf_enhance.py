import cv2
import numpy as np

def CCI_Calc(image, tolerance=0.1):
    """
    Calculates the Contrast Code Image (CCI).
    Adapted and fixed from evan-person/TEBCF_python
    """
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    patch_sizes = [3, 5, 7, 9, 11, 13, 15]
    
    # Tolerance array to prioritize bigger patch sizes
    tol = 1 - (tolerance / 100.0)
    # Using powers of tol as weights: [tol^6, tol^5, ..., tol^0]
    tolerance_weights = [tol**(6-i) for i in range(7)]
    
    scores = []
    for i, size in enumerate(patch_sizes):
        # Calculate local standard deviation
        # Optimization: use cv2.blur to calc square mean for faster std calculation
        mean = cv2.blur(img_gray, (size, size))
        mean_sq = cv2.blur(img_gray**2, (size, size))
        std = np.sqrt(np.maximum(mean_sq - mean**2, 0))
        
        # Apply tolerance weight
        scores.append(std * tolerance_weights[i])
    
    score_stack = np.stack(scores, axis=2)
    # The index with minimum score indicates the appropriate patch size
    cci = np.argmin(score_stack, axis=2)
    return cci

def apply_clahe(img, clip=1.2):
    """Gentle CLAHE to avoid over-exposure in bright scenes."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def partial_gray_world(img, strength=0.4):
    """
    Partial gray-world white balance.
    strength=0.0 → no change, strength=1.0 → full gray-world.
    Using 0.4 so we correct underwater cast partially without washing out.
    """
    result = img.copy().astype(np.float32)
    avg_b = np.mean(result[:, :, 0])
    avg_g = np.mean(result[:, :, 1])
    avg_r = np.mean(result[:, :, 2])
    avg   = (avg_b + avg_g + avg_r) / 3.0

    scale_b = 1.0 + strength * (avg / (avg_b + 1e-6) - 1.0)
    scale_g = 1.0 + strength * (avg / (avg_g + 1e-6) - 1.0)
    scale_r = 1.0 + strength * (avg / (avg_r + 1e-6) - 1.0)

    result[:, :, 0] = np.clip(result[:, :, 0] * scale_b, 0, 255)
    result[:, :, 1] = np.clip(result[:, :, 1] * scale_g, 0, 255)
    result[:, :, 2] = np.clip(result[:, :, 2] * scale_r, 0, 255)
    return result.astype(np.uint8)

def white_balance(img):
    # Try using cv2.xphoto if available
    try:
        wb = cv2.xphoto.createSimpleWB()
        return wb.balanceWhite(img)
    except:
        # Fallback to Gray World white balance
        result = img.copy().astype(np.float32)
        avg_r = np.mean(result[:, :, 2])
        avg_g = np.mean(result[:, :, 1])
        avg_b = np.mean(result[:, :, 0])
        avg = (avg_r + avg_g + avg_b) / 3.0
        result[:, :, 2] *= (avg / avg_r)
        result[:, :, 1] *= (avg / avg_g)
        result[:, :, 0] *= (avg / avg_b)
        return np.clip(result, 0, 255).astype(np.uint8)

def laplacian_pyramid_fusion(img1, img2, weights1, weights2, levels=5):
    # Normalized weights
    sum_w = weights1 + weights2 + 1e-12
    w1 = weights1 / sum_w
    w2 = weights2 / sum_w

    # Image pyramids
    lp1 = [img1.astype(np.float32)]
    lp2 = [img2.astype(np.float32)]
    wp1 = [w1.astype(np.float32)]
    wp2 = [w2.astype(np.float32)]

    for i in range(levels - 1):
        lp1.insert(0, cv2.pyrDown(lp1[0]))
        lp2.insert(0, cv2.pyrDown(lp2[0]))
        wp1.insert(0, cv2.pyrDown(wp1[0]))
        wp2.insert(0, cv2.pyrDown(wp2[0]))

    # Convert to Laplacian for images
    for i in range(levels - 1):
        size = (lp1[i+1].shape[1], lp1[i+1].shape[0])
        lp1[i+1] = lp1[i+1] - cv2.pyrUp(lp1[i], dstsize=size)
        lp2[i+1] = lp2[i+1] - cv2.pyrUp(lp2[i], dstsize=size)

    # Fusion
    fused_pyramid = []
    for i in range(levels):
        # Weights need to be matching channels
        w1_m = cv2.merge([wp1[i]]*3) if len(lp1[i].shape) == 3 else wp1[i]
        w2_m = cv2.merge([wp2[i]]*3) if len(lp2[i].shape) == 3 else wp2[i]
        fused_pyramid.append(lp1[i] * w1_m + lp2[i] * w2_m)

    # Reconstruction
    reconstructed = fused_pyramid[0]
    for i in range(1, levels):
        size = (fused_pyramid[i].shape[1], fused_pyramid[i].shape[0])
        reconstructed = cv2.pyrUp(reconstructed, dstsize=size) + fused_pyramid[i]

    return np.clip(reconstructed, 0, 255).astype(np.uint8)

def TEBCF_Enhance(image):
    """
    True TEBCF: Texture/Blurriness-based Color Fusion.
    input1 = partial gray-world corrected (40% correction — keeps some underwater feel)
    input2 = gentle CLAHE enhanced version
    Fused by texture/blurriness weights. No aggressive white balance.
    """
    # 1. Partial gray-world (mild) + gentle CLAHE
    input1 = partial_gray_world(image, strength=0.4)
    input2 = apply_clahe(input1, clip=1.2)

    # 2. Weights based on blurriness (CCI) + saliency + luminance
    cci = CCI_Calc(image)
    cci_normalized = cci.astype(float) / 6.0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(float) / 255.0
    saliency  = np.abs(cv2.GaussianBlur(gray, (5, 5), 0) - gray)
    luminance = np.exp(-0.5 * ((gray - 0.5)**2) / (0.25**2))

    w2 = saliency * luminance * (1.0 + cci_normalized)
    w1 = luminance

    # 3. Laplacian pyramid fusion
    enhanced = laplacian_pyramid_fusion(input1, input2, w1, w2)
    return enhanced


if __name__ == "__main__":
    import sys
    import os
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        img = cv2.imread(img_path)
        if img is not None:
            enhanced = TEBCF_Enhance(img)
            cv2.imwrite("enhanced_test.png", enhanced)
            print("Enhanced image saved as enhanced_test.png")
