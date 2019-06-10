"""
Trabalho 4 - Panorama
Nathália Harumi Kuromiya
RA 175188
"""

import os
import cv2
import numpy as np

def save_image(image, filename, sufix):
    """ Save images at "results" directory based on its previous filename

        Args:
        image (np.ndarray): image to be saved
        filename (str): the filepath of the image
        sufix (str): the sufix to be add to the new filename

    """

    # Create directory
    if not os.path.exists("results"):
        os.mkdir("results")

    # Set filepath
    saving_filename = "results/" + filename.split("/")[-1][:-5] + "_" + sufix \
                      + ".jpg"

    # Save image
    cv2.imwrite(saving_filename, image)

def get_keypoints(img_1, img_2, detector, descriptor):
    gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

    kp_1 = detector.detect(gray_1, None)
    kp_2 = detector.detect(gray_2, None)

    kp_1, desc_1 = descriptor.compute(img_1, kp_1)
    kp_2, desc_2 = descriptor.compute(img_2, kp_2)

    return (kp_1, desc_1), (kp_2, desc_2)

def get_detector_descriptor_and_normtype(option):
    surf = cv2.xfeatures2d.SURF_create()
    sift = cv2.xfeatures2d.SIFT_create()
    star = cv2.xfeatures2d.StarDetector_create()
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    orb = cv2.ORB_create()
    return {
    "SURF" : (surf, surf, cv2.NORM_L2),
    "SIFT" : (sift, sift, cv2.NORM_L2),
    "BRIEF" : (star, brief, cv2.NORM_HAMMING),
    "ORB" : (orb, orb, cv2.NORM_HAMMING)
    }.get(option, (None, None))

def get_first_20_matches(bf, desc_1, desc_2):
    # Match descriptors.
    matches = bf.match(desc_1,desc_2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    return len(matches), matches[:20]

def get_matches_ratio_test(bf, desc_1, desc_2):
    matches = bf.knnMatch(desc_1, desc_2, k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    return len(matches), good

def get_matches(bf, desc_1, desc_2, match_filter_method):
    if match_filter_method == "FIRST_20":
        return get_first_20_matches(bf, desc_1, desc_2)
    elif match_filter_method == "RATIO_TEST":
        return get_matches_ratio_test(bf, desc_1, desc_2)
    else:
        return None

def get_points(matches, kp_1, kp_2):
    pts_1 = np.float32([ kp_1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    pts_2 = np.float32([ kp_2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

    return pts_1, pts_2

def order_parameters(img_1, img_2, src_pts, dst_pts):
    h_desl = 0
    w_desl = 0
    for s, d in zip(src_pts, dst_pts):
        [[sh, sw]] = s
        [[dh, dw]] = d

        h_desl = h_desl + dh/img_2.shape[0] - sh/img_1.shape[0]
        w_desl = w_desl + dw/img_2.shape[1] - sw/img_1.shape[1]

    main_desl = h_desl if abs(h_desl) > abs(w_desl) else w_desl

    if main_desl > 0:
        src = src_pts
        dst = dst_pts
        right_img = img_1
        left_img = img_2
    else:
        src = dst_pts
        dst = src_pts
        right_img = img_2
        left_img = img_1

    return left_img, right_img, src, dst

def merge_images(left_img, right_img, src_pts, dst_pts):
    (hl, wl) = left_img.shape[:2]
    (hr, wr) = right_img.shape[:2]

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    right_img_warped = cv2.warpPerspective(right_img, M, (wr + wl, hr + hl))

    vis = right_img_warped.copy()
    vis[0:hl, 0:wl] = left_img
    return vis

def apply_method(img_1, img_2, desc_method, filename, match_filter_method):
    detector, descriptor, normType = get_detector_descriptor_and_normtype(desc_method)

    if detector == None:
        print("No decriptor method found")
        return None

    (kp_1, desc_1), (kp_2, desc_2) = get_keypoints(img_1, img_2, detector,
                                                   descriptor)
    n_total_matches, best_matches = get_matches(cv2.BFMatcher(normType), desc_1,
                                                desc_2, match_filter_method)
    if n_total_matches == None:
        print("No match filter method found")
        return None
    elif n_total_matches < 4:
        print("Not enough keypoint matches for " + desc_method)
        return n_total_matches

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   flags = 2)
    matches_img = cv2.drawMatches(img_1, kp_1, img_2, kp_2, best_matches,
                                  outImg=np.array([]), **draw_params)
    pts_1, pts_2 = get_points(best_matches, kp_1, kp_2)
    left_img, right_img, src, dst = order_parameters(img_1, img_2, pts_1, pts_2)
    panorama = merge_images(left_img, right_img, src, dst)
    save_image(panorama, filename, desc_method + "_" + match_filter_method)
    save_image(matches_img, filename, desc_method + "_matches_" + match_filter_method)
    return n_total_matches

def compare_methods(filename_1, filename_2):
    img_1 = cv2.imread(filename_1)
    img_2 = cv2.imread(filename_2)
    print("Total of matches for image '" + filename_1[:-5] + "':")
    sift_n_matches_first_20 = apply_method(img_1, img_2, "SIFT", filename_1, "FIRST_20")
    sift_n_matches_ratio_test = apply_method(img_1, img_2, "SIFT", filename_1, "RATIO_TEST")
    surf_n_matches_first_20 = apply_method(img_1, img_2, "SURF", filename_1, "FIRST_20")
    surf_n_matches_ratio_test = apply_method(img_1, img_2, "SURF", filename_1, "RATIO_TEST")
    brief_n_matches_first_20 = apply_method(img_1, img_2, "BRIEF", filename_1, "FIRST_20")
    brief_n_matches_ratio_test = apply_method(img_1, img_2, "BRIEF", filename_1, "RATIO_TEST")
    orb_n_matches_first_20 = apply_method(img_1, img_2, "ORB", filename_1, "FIRST_20")
    orb_n_matches_ratio_test = apply_method(img_1, img_2, "ORB", filename_1, "RATIO_TEST")
    print("FIRST 20: \nsift: {}; surf: {}; brief: {}; orb: {}\n\n".format(sift_n_matches_first_20,
          surf_n_matches_first_20, brief_n_matches_first_20, orb_n_matches_first_20))
    print("RATIO TEST: \nsift: {}; surf: {}; brief: {}; orb: {}\n\n".format(sift_n_matches_ratio_test,
          surf_n_matches_ratio_test, brief_n_matches_ratio_test, orb_n_matches_ratio_test))

def main():
    filenames = [
    ("../images/jpg/foto1A.jpg", "../images/jpg/foto1B.jpg"),
    ("../images/jpg/foto2A.jpg", "../images/jpg/foto2B.jpg"),
    ("../images/jpg/foto3A.jpg", "../images/jpg/foto3B.jpg"),
    ("../images/jpg/foto4A.jpg", "../images/jpg/foto4B.jpg"),
    ("../images/jpg/foto5A.jpg", "../images/jpg/foto5B.jpg")
    ]
    for f in filenames:
        (f1, f2) = f
        compare_methods(f1, f2)

if __name__ == '__main__':
    main()
