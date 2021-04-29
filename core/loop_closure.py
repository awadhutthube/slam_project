import numpy as np
import scipy.io
import cv2
import copy
from .utils import *

class LoopClosure():
    def __init__(self, path, dataset, intrinsic_mat):
        gt_loop_data = scipy.io.loadmat(path + 'gnd_kitti00.mat')
        self.neighbours = gt_loop_data['gnd']            # This is a numpy array of shape (num_images, 1)
        self.dataset = dataset
        self.K = intrinsic_mat
        return
    
    def check_loop_closure(self, idx, frame_new):
        loop_closure_flag = False
        pose, matched_idx = None, None
        best_kp1, best_kp2, best_matches = None, None, None
        local_neighbours = self.neighbours[idx][0][0]         # numpy array of neighbours
        # import ipdb; ipdb.set_trace()
        valid_neighbours = local_neighbours[local_neighbours < idx]

        # TODO: Check similarity with all valid neighbours and choose 1 or 2 to create edges
        max_num_matches = 0
        for img_idx in valid_neighbours:
            frame_old, _, _ = self.dataset[img_idx]
            # Check similarity using keypoint matches

            kp1, kp2, matches = self.find_matches(frame_new, frame_old)
            # Can also set min number of required matches

            if len(matches) > max_num_matches:
                max_num_matches = len(matches)
                target_frame = frame_old.copy()
                best_kp1 = kp1
                best_kp2 = kp2
                best_matches = matches
                matched_idx = img_idx

        # Compute R and t for maximally matching neighbours
        if max_num_matches > 0:
            # import ipdb; ipdb.set_trace()
            matched_kp1 = []
            matched_kp2 = []
            for mat in best_matches:
                matched_kp1.append(best_kp1[mat[0].queryIdx].pt)
                matched_kp2.append(best_kp2[mat[0].trainIdx].pt)
            # matched_kp1 = [best_kp1[mat.queryIdx].pt for mat in best_matches]
            # matched_kp2 = [best_kp2[mat.trainIdx].pt for mat in best_matches]
            matched_kp1 = np.array(matched_kp1)
            matched_kp2 = np.array(matched_kp2)
            E, _ = cv2.findEssentialMat(matched_kp1, matched_kp2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            _, R, t, mask = cv2.recoverPose(E, matched_kp1, matched_kp2, self.K)
            pose = convert_to_4_by_4(convert_to_Rt(R,t))
            loop_closure_flag = True
            # cv2.imshow('Current', frame_new)
            # cv2.waitKey(0)
            # cv2.imshow('Target', target_frame)
            # cv2.waitKey(0)
        return loop_closure_flag, pose, matched_idx

    def find_matches(self, img1, img2, return_ratio = 1):
        sift = cv2.SIFT_create()

        kp1, descriptors_1 = sift.detectAndCompute(img1,None)
        kp2, descriptors_2 = sift.detectAndCompute(img2,None)

        # Nearest matches for lowe's ratio test
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors_1,descriptors_2,k=1)
        # good = []
        # for m,n in matches:
        #     if m.distance < 0.75*n.distance:
        #         good.append(m)
        # matches = sorted(good, key = lambda x:x.distance)
        return kp1, kp2, matches