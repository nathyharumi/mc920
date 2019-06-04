import cv2
import numpy as np
import matplotlib.pyplot as plt

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

def sift_keypoints(img_1, img_2):
    gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    kp_1, desc_1 = sift.detectAndCompute(gray_1, None)
    kp_2, desc_2 = sift.detectAndCompute(gray_2, None)

    r1 = cv2.drawKeypoints(img_1,kp_1, outImage=np.array([]))
    r2 = cv2.drawKeypoints(img_2,kp_2, outImage=np.array([]))

    return (kp_1, desc_1), (kp_2, desc_2)

def get_matches(bf, kp_1, desc_1, kp_2, desc_2):
    # Match descriptors.
    matches = bf.match(desc_1,desc_2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    # Draw first 20 matches.
    print(len(matches))
    #img3 = cv2.drawMatches(img_1, kp_1, img_2, kp_2, matches[:20], flags=2, outImg=np.array([]))
    #plt.imshow(img3),plt.show()
    good = matches[:50]

    return good

def get_points(matches, kp_1, kp_2):
    pts_1 = np.float32([ kp_1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    pts_2 = np.float32([ kp_2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

    return pts_1, pts_2

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
    plt.imshow(vis), plt.show()

def surf_keypoints(img_1, img_2):
    gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

    surf = cv2.xfeatures2d.SURF_create()
    kp_1, desc_1 = surf.detectAndCompute(gray_1, None)
    kp_2, desc_2 = surf.detectAndCompute(gray_2, None)

    r1 = cv2.drawKeypoints(img_1,kp_1, outImage=np.array([]))
    r2 = cv2.drawKeypoints(img_2,kp_2, outImage=np.array([]))
    cv2.imwrite("surf1A.jpg", r1)
    cv2.imwrite("surf1B.jpg", r2)

def run(img_1, img_2):
    (kp_1, desc_1), (kp_2, desc_2) = sift_keypoints(img_1, img_2)
    matches = get_matches(cv2.BFMatcher(), kp_1, desc_1, kp_2, desc_2)
    pts_1, pts_2 = get_points(matches, kp_1, kp_2)
    left_img, right_img, src, dst = order_parameters(img_1, img_2, pts_1, pts_2)
    merge_images(left_img, right_img, src, dst)

def main():
    filename_1 = "../images/jpg/foto1A.jpg"
    filename_2 = "../images/jpg/foto1B.jpg"
    img_1 = cv2.imread(filename_1)
    img_2 = cv2.imread(filename_2)
    run(img_1, img_2)
    #surf_keypoints(img_1, img_2)

if __name__ == '__main__':
    main()
