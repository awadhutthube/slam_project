import numpy as np
import scipy.io
import cv2
import copy
class LoopClosure():
    def __init__(self, path, dataset):
        gt_loop_data = scipy.io.loadmat(path)
        self.neighbours = gt_loop_data['gnd']            # This is a numpy array of shape (num_images, 1)
        self.dataset = dataset
        return
    
    def check_loop_closure(self, idx, frame_new):
        local_neighbours = self.neighbours[idx][0]         # numpy array of neighbours
        valid_neighbours = local_neighbours[local_neighbours < idx]

        # TODO: Check similarity with all valid neighbours and choose 1 or 2 to create edges
        max_num_matches = 0
        for img_idx in valid_neighbours:
            frame_old, _, _ = dataset[img_idx]
            kp1, kp2, matches = find_matches(frame_new, frame_old)
            # Can also set min number of required matches
            if len(matches) > max_num_matches:
                target_frame = frame_old.copy()
                best_kp1 = kp1.copy()
                best_kp2 = kp2.copy()
                best_matches = matches.copy()

        # Check similarity using keypoint matches
        # Compute R and t for maximally matching neighbours
        # Add edge between appropriate nodes
        return pose

    def find_matches(img1, img2, return_ratio = 1):
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