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

def get_matches(bf, kp_1, desc_1, kp_2, desc_2):
    # Match descriptors.
    matches = bf.match(desc_1,desc_2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    return len(matches), matches[:10]

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

    #points = np.ones([4, 2, 1]).astype('float32')
    #print(points.shape)
    #warped_points = cv2.perspectiveTransform(points, M)

    vis = right_img_warped.copy()
    vis[0:hl, 0:wl] = left_img
    return vis

def apply_method(img_1, img_2, method, filename):
    detector, descriptor, normType = get_detector_descriptor_and_normtype(method)

    if detector == None:
        print("No method found")
        return None

    (kp_1, desc_1), (kp_2, desc_2) = get_keypoints(img_1, img_2, detector,
                                                   descriptor)
    n_total_matches, best_matches = get_matches(cv2.BFMatcher(normType), kp_1,
                                                desc_1, kp_2, desc_2)

    if n_total_matches < 4:
        print("Not enough keypoint matches for " + method)
        return n_total_matches
    matches_img = cv2.drawMatches(img_1, kp_1, img_2, kp_2, best_matches,
                                  flags=2, outImg=np.array([]))
    pts_1, pts_2 = get_points(best_matches, kp_1, kp_2)
    left_img, right_img, src, dst = order_parameters(img_1, img_2, pts_1, pts_2)
    panorama = merge_images(left_img, right_img, src, dst)
    save_image(panorama, filename, method)
    save_image(matches_img, filename, method + "_matches")
    return n_total_matches

def compare_methods(filename_1, filename_2):
    img_1 = cv2.imread(filename_1)
    img_2 = cv2.imread(filename_2)
    print("Total of matches for image '" + filename_1[:-5] + "':")
    sift_n_matches = apply_method(img_1, img_2, "SIFT", filename_1)
    surf_n_matches = apply_method(img_1, img_2, "SURF", filename_1)
    brief_n_matches = apply_method(img_1, img_2, "BRIEF", filename_1)
    orb_n_matches = apply_method(img_1, img_2, "ORB", filename_1)
    print("sift: {}; surf: {}; brief: {}; orb: {}".format(sift_n_matches,
          surf_n_matches, brief_n_matches, orb_n_matches))

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
