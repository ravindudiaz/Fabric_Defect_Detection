import numpy as np
import cv2
import math
import statistics as stat
from matplotlib import pyplot as plt
import os
import json

matching_ref_loc ="./Assets/Seg_Module/Output/Matching/ref/"
matching_test_loc = "./Assets/Seg_Module/Output/Matching/test/"
nonmatching_ref_loc = "./Assets/Seg_Module/Output/Non_Matching/ref/"
nonmatching_test_loc="./Assets/Seg_Module/Output/Non_Matching/test/"
nonmatching_ref_conflict = "./Assets/Seg_Module/Output/Non_Matching/ref/conflict/"
nonmatching_test_conflict = "./Assets/Seg_Module/Output/Non_Matching/test/conflict/"

#might need to be adjusted as per segment rois
ref_seg_roi_loc = "./Assets/QA_Module/Output/rois"
test_seg_roi_loc = "./Assets/QA_Module/Output/rois"

# matching_ref
mr_file_list = os.listdir(matching_ref_loc)
no_of_matching_ref_segs = len(mr_file_list)


#matching_test
mt_file_list = os.listdir(matching_test_loc)
no_of_matching_test_segs = len(mt_file_list)

#nonmatching_ref
nmr_file_list = os.listdir(nonmatching_ref_loc)
no_of_nonmatching_ref_segs = len(nmr_file_list)

#non_matching_ref_conflict
nmr_conflict_file_list = os.listdir(nonmatching_ref_conflict)

#non_matching_test
nmt_file_list = os.listdir(nonmatching_test_loc)
no_of_nonmatching_test_segs = len(nmt_file_list) - 1

#non_matching_test_conflict
nmt_conflict_file_list = os.listdir(nonmatching_test_conflict)

#ref artwork & cloth loc
ref_artwork_mask_loc = "Assets/BR_Module/Output/mask/ref/artwork/"
ref_or_cloth_loc = "Assets/BR_Module/Output/mask/ref/cloth/"  #outer removed

#test artwork &cloth loc
test_artwork_mask_loc = "Assets/BR_Module/Output/mask/test/artwork/"
test_or_cloth_loc = "Assets/BR_Module/Output/mask/test/cloth/"  #outer removed

#ref isolated artwork loc
ref_artwork_loc = "./Assets/BR_Module/Output/ref/isolated_artwork/"


def match_segments(nm_ref_loc, nm_test_loc, m_ref_loc, m_test_loc):
        print("Conflict Segment Matching Started...........................................................")
        print(nm_test_loc)
        if no_of_nonmatching_ref_segs - 1 != 0:
                print("Missing segment in test artwork")
                for segf in nmr_file_list:
                        if segf.endswith('.jpg') or segf.endswith('.jpeg'):
                                print("Files", segf)
                                def_seg = cv2.imread(nm_ref_loc + segf)
                                cv2.imshow("Missing segment " + segf, def_seg)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        elif no_of_nonmatching_test_segs - 1 != 0:
                print("Damaged Printwork")
                for segf in nmt_file_list:
                        if segf.endswith('.jpg') or segf.endswith('.jpeg'):
                                print(segf)
                                def_seg = cv2.imread(nm_test_loc + segf)
                                cv2.imshow(segf , def_seg)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        elif len(nmr_conflict_file_list) != 0:
                for i in range(len(nmr_conflict_file_list)):
                        nmrc_seg = cv2.imread(nonmatching_ref_conflict + "nm_" + str(i))
                        nmrc_seg_gr = cv2.cvtColor(nmrc_seg, cv2.COLOR_BGR2GRAY)

                        #Reference keypoints
                        rc_kp = []
                        matching_kp_list = []
                        sift = cv2.SIFT_create()
                        kprc, desr = sift.detectAndCompute(nmrc_seg_gr, None)
                        tc_segs = []
                        for j in range(len(nmt_conflict_file_list)):
                                nmtc_seg = cv2.imread(nonmatching_test_conflict + "nm_"+str(i)+str(j))
                                tc_segs.append(nmtc_seg)
                                nmtc_seg_gr = cv2.cvtColor(nmtc_seg, cv2.COLOR_BGR2GRAY)
                                kptc, dest = sift.detectAndCompute(nmtc_seg_gr, None)
                                flann = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
                                matches = flann.match(dest, desr)
                                matches = sorted(matches, key = lambda x: x.distance)
                                # matches = list(filter(lambda x: x.distance <=100, matches))

                                points1 = np.zeros((len(matches), 2), dtype=np.float32)
                                points2 = np.zeros((len(matches), 2), dtype=np.float32)

                                for l, match in enumerate(matches):
                                        points1[l, :] = kptc[match.queryIdx].pt
                                        points2[l, :] = kprc[match.queryIdx].pt

                                h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

                                matching_result = cv2.drawMatches(nmtc_seg_gr, kptc, nmrc_seg_gr, matches[:20], None)
                                cv2.namedWindow("matched conflict kps", cv2.WINDOW_NORMAL)
                                cv2.resizeWindow("matched conflict kps", 900, 1200)
                                cv2.imshow("matched conflict kps", matching_result)

                                ransac_count = 0
                                ransac_pos = []
                                for k in range(len(mask)):
                                        if mask[k] == [1]:
                                                ransac_count += 1
                                                ransac_pos.append(k)

                                rc_kp.append(ransac_count)
                                matching_kp_list.append(ransac_pos)
                        max = 0
                        max_idx = 0
                        for x in range(len(rc_kp)):
                                if max < rc_kp[x]:
                                        max = rc_kp[x]
                                        max_idx = x
                        best_match = x

                        curr_matching_ref_list = os.listdir(matching_ref_loc)
                        curr_files = len(curr_matching_ref_list)
                        cv2.imwrite(matching_ref_loc+"m_"+str(curr_files), nmrc_seg)
                        os.remove(nonmatching_ref_conflict+"nm_"+str(i))
                        curr_matching_test_list = os.listdir(matching_test_loc)
                        cv2.imwrite(matching_test_loc+"m_"+str(curr_files), tc_segs[best_match])
                        os.remove(nonmatching_test_conflict + "nm_" + str(i) + str(x))
        print("Conflict Segment Matching Done...............................................")


def check_artwork_position(ref_artwork_loc, ref_cloth_loc, test_artwork_loc, test_cloth_loc):

        test_artwork = cv2.imread(test_artwork_loc + "Mask_girlb2_tex_blueflower1.jpg")  #outer background removed test image
        ref_artwork = cv2.imread(ref_artwork_loc + "Mask_girlb2_tex_blueflower.jpg") #isolated artwork

        ref_artwork_gr = cv2.cvtColor(ref_artwork, cv2.COLOR_BGR2GRAY)
        nz_artwork_locs = np.argwhere(ref_artwork)

        x = []
        y = []
        for pt in nz_artwork_locs:
                x.append(pt[1])
                y.append(pt[0])
        x = sorted(x)
        y = sorted(y)

        # ymin = y[0]
        # ymax = y[len(y) - 1]
        # xmin = x[0]
        # xmax = x[len(x)-1]
        # cropped_artwork = ref_artwork[ ymin:ymax , xmin:xmax ]
        # cropped_artwork_gr = cv2.cvtColor(cropped_artwork, cv2.COLOR_BGR2GRAY)
        cropped_artwork_gr = cv2.cvtColor(ref_artwork, cv2.COLOR_BGR2GRAY)

        #SIFT Operator
        sift = cv2.SIFT_create()
        kpr, desr = sift.detectAndCompute(cropped_artwork_gr, None) #For the cropped artwork
        kpt, dest = sift.detectAndCompute(ref_artwork_gr, None)   #For the outer removed test cloth

        #Using FLANN BASED matcher
        flann = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
        matches = flann.match(dest, desr)
        print("FLANN matches : ",len(matches))

        matches = sorted(matches, key=lambda x: x.distance)

        if len(kpt) >= len(kpr):
                matches = matches[:(len(kpr)-1)]
        else:
                matches = matches[:(len(kpt)-1)]

        matches = list(filter(lambda x: x.distance <= 100, matches))

        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        print(len(kpt), len(kpr))

        for l, match in enumerate(matches):
                points1[l, :] = kpt[match.queryIdx].pt
                points2[l, :] = kpr[match.queryIdx].pt
        h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

        matching_result = cv2.drawMatches(ref_artwork_gr, kpt, cropped_artwork_gr, kpr, matches[:10], None)
        # print(mask)
        cv2.imshow("Matching Result ", matching_result)
        cv2.waitKey(0)

        #Find the relevent keypoints that should be taken from the test and ref artworks for comparison
        ransac_count = 0
        ransac_pos = []
        for k in range(len(mask)):
                if mask[k] == [1]:
                        ransac_count += 1
                        ransac_pos.append(k)
        print(ransac_count)
        print(ransac_pos)

        ref_pois = []
        test_pois = []

        for pos in ransac_pos:
                ref_pois.append([int(kpr[pos].pt[0]), int(kpr[pos].pt[1])])
                test_pois.append([int(kpt[pos].pt[0]), int(kpt[pos].pt[1])])
        print("Ref printwork pois  : ", ref_pois)
        print("Test printwork pois : ", test_pois)

        #Ref Cloth mask or the Outer removed cloth - ref ----------------------
        ref_or_cloth = cv2.imread(ref_or_cloth_loc + "girlb2_tex_blueflower.jpg")
        ref_or_cloth_gr = cv2.cvtColor(ref_or_cloth, cv2.COLOR_BGR2GRAY)

        ref_or_cloth_nz = np.argwhere(ref_or_cloth_gr)
        # print(ref_or_cloth_nz)
        xr = []
        yr = []
        for pt in ref_or_cloth_nz:
                xr.append(pt[1])
                yr.append(pt[0])
        xr = sorted(x)
        yr = sorted(y)
        xr_max = x[len(x)-1]
        xr_min = x[0]
        yr_max = y[len(y)-1]
        yr_min = y[0]
        xr_diff = xr_max - xr_min


        #Ref image reference points

        pr1 = [xr_min + int(xr_diff*(1/3)), yr_min]
        pr2 = [xr_min + int(xr_diff*(2/3)), yr_min]
        rtotal_distance_1 = 0
        rtotal_distance_2 = 0
        for pt in ref_pois:
                rdist1 = (pt[0] - pr1[0])**2 + (pt[1]-pr1[0])**2
                rdist2 = (pt[0] - pr2[0])**2 + (pt[1]-pr2[0])**2
                rtotal_distance_1 += rdist1
                rtotal_distance_2 += rdist2

        rdist = (rtotal_distance_1/len(ref_pois)) + (rtotal_distance_2/len(ref_pois))
        print("Reference printwork distance : ", rdist)


        # Test Cloth mask or the Outer removed cloth - Test --------------------------
        test_or_cloth = cv2.imread(test_or_cloth_loc + "girlb2_tex_blueflower2.jpg")
        test_or_cloth_gr = cv2.cvtColor(test_or_cloth, cv2.COLOR_BGR2GRAY)

        test_or_cloth_nz = np.argwhere(test_or_cloth_gr)
        xt = []
        yt = []
        for pt in test_or_cloth_nz:
                xt.append(pt[1])
                yt.append(pt[0])
        xt = sorted(xt)
        yt = sorted(yt)
        xt_max = xt[len(xt) - 1]
        xt_min = xt[0]
        yt_max = yt[len(yt) - 1]
        yt_min = yt[0]

        # Test image reference points

        pt1 = [xt_min + int(xr_diff * (1 / 3)), yt_min]
        pt2 = [xt_min + int(xr_diff * (2 / 3)), yt_min]
        ttotal_distance_1 = 0
        ttotal_distance_2 = 0
        for pt in ref_pois:
                dist1 = (pt[0] - pt1[0]) ** 2 + (pt[1] - pt1[0]) ** 2
                dist2 = (pt[0] - pt2[0]) ** 2 + (pt[1] - pt2[0]) ** 2
                ttotal_distance_1 += dist1
                ttotal_distance_2 += dist2

        tdist = (ttotal_distance_1 / len(test_pois)) + (ttotal_distance_2 / len(test_pois))

        print("test printwork distance : ", tdist)

        dist_diff = abs(tdist - rdist) / rdist

        print("Distance difference : ", dist_diff)

        if dist_diff >= 0.16:
                print("Printwork position defect...")

        #Cloth mask of the Outer removed cloth - test
        # test_or_cloth = cv2.imread(test_or_cloth_loc+"")

        # #Harris Corner Detector
        # ref_or_cloth_gr[ref_or_cloth_gr != 0] = 255
        # kernel = np.ones((5,5), np.uint8)
        # ref_or_cloth_gr = cv2.erode(ref_or_cloth_gr, kernel, iterations=3)
        # # cv2.imshow("eroded",test_or_cloth_gr)
        # ref_or_cloth_gr = np.float32(ref_or_cloth_gr)
        # dst = cv2.cornerHarris(ref_or_cloth_gr, 2, 3, 0.04)
        # # print(dst)
        #
        # print(ref_or_cloth_gr.shape)
        # ref_or_cloth_dr = ref_or_cloth
        #
        # arr = np.argwhere(dst)
        # print(len(arr))
        #
        # # test_or_cloth_dr[test_or_cloth_dr != [0,0,0]] = 255
        # ref_or_cloth_dr[dst > 0.040*dst.max()] = [0,0,255]
        # print(dst)
        # pois = np.where(dst <0.025*dst.max())
        # # print(pois, len(pois))
        # cv2.imshow('dst', ref_or_cloth_dr)
        # cv2.waitKey(0)

        # nz_locs_ref = np.argwhere(ref_or_cloth_gr)




def detect_features(no_of_matching_ref_segs, ref_img_check):
        print("Reference Image Feature Extraction Started.........................................")

        ref_features=[]
        thresholded_segments = []
        ref_segs = []

        for i in range(no_of_matching_ref_segs):
                print("Reference Image Segment ", str(i))
                seg_features = []
                if ref_img_check == 1:
                        seg = cv2.imread(matching_ref_loc+"m"+"_"+str(i)+".jpg")
                else:
                        seg = cv2.imread(matching_test_loc + "m" + "_" + str(i) + ".jpg")
                ref_segs.append(seg)
                gr_seg = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)

                #Detect Shape
                huMoments, moments, thresh_seg = detect_shape(gr_seg)
                print(huMoments)

                #Detect Size
                segmentContourArea = detect_size(thresh_seg)
                print(segmentContourArea)

                #Detect Rotation
                ref_avg_comment_measure = detect_rotation(moments)
                print(ref_avg_comment_measure)

                #Detect Placement
                # segment center
                dimensions = seg.shape
                ref_nz = np.argwhere(seg)
                xg = int(dimensions[1] / 2)
                yg = int(dimensions[0] / 2)
                garment_center = [xg, yg]
                segment_placement_measures = detect_placement(moments, garment_center)
                print(segment_placement_measures)

                #Detect Color
                segment_color_measures = detect_color(seg, thresh_seg)

                #Detect Minima Maxima
                # seg_zero_crossings, seg_point_measures = detect_minima_maxima(thresh_seg, seg)

                # ref_seg_curvature_list, contours_ref = detMinMax(thresh_seg, seg)

                # seg_features = [huMoments[6],huMoments[0],huMoments[1],huMoments[2],segmentContourArea, ref_avg_comment_measure, segment_placement_measures, segment_color_measures, seg_zero_crossings, seg_point_measures]
                # seg_features = [huMoments[6], huMoments[0], huMoments[1], huMoments[2], segmentContourArea, ref_avg_comment_measure, segment_placement_measures, segment_color_measures, ref_seg_curvature_list, contours_ref]
                seg_features = [huMoments[6], huMoments[0], huMoments[1], huMoments[2], segmentContourArea, ref_avg_comment_measure, segment_placement_measures, segment_color_measures]
                # cv2.imshow("Init Contours", thresh_seg)

                # cv2.waitKey(0)

                ref_features.append(seg_features)
                thresholded_segments.append(thresh_seg)
        # print(ref_features)
        return ref_features, thresholded_segments, dimensions, ref_segs

def detect_and_compare_matching_segments(no_of_segments,ref_features,test_img_check, reference_thresh_segs, ref_dimensions, ref_segs):
        print("Detect and Match test image stage reached...........................")

        no_def_segs = 0

        shape_def = []
        color_def = []
        placement_def = []
        rotation_def = []
        size_def = []
        minmax_def = []


        for i in range(no_of_matching_test_segs):
                print(no_of_matching_test_segs)

                print("Matching Test Segment :", str(i))
                if test_img_check == 1:
                        test_seg = cv2.imread(matching_test_loc+"m"+"_"+str(i)+".jpg")

                gr_test_seg = cv2.cvtColor(test_seg, cv2.COLOR_BGR2GRAY)

                # Detect and Compare Shape
                huMoments, moments, thresh_seg = detect_shape(gr_test_seg)
                if ref_features[i][0]*huMoments[6] < 0:
                        shape_defect = {
                                "type": "Shape Error",
                                "status": "Mirrored segment"
                        }
                        print(shape_defect)
                        shape_defect_json = json.dumps(shape_defect)
                        no_def_segs += 1
                        shape_def.append([i, test_seg])
                else:
                        #shape_deviation_measure
                        shape_deviation = ((ref_features[i][1]-huMoments[0])**2 + (ref_features[i][2]-huMoments[1])**2 + (ref_features[i][3]-huMoments[2])**2 )/(ref_features[i][1]+ref_features[i][2]+ref_features[i][3])
                        print("Shape Deviation : ",shape_deviation)
                        # Ts = 0.15       #Shape Deviation Threshold - Experimental
                        Ts = 100000
                        if shape_deviation >= Ts:
                                shape_defect = {
                                        "type": "Shape Defect",
                                        "status" : "Shape deviation = "+str(shape_deviation)+". Exceeded threshold."
                                }
                                no_def_segs += 1
                                shape_def.append([i, test_seg])
                                shape_defect_json = json.dumps(shape_defect)
                        else:
                                #Detect and Compare Size
                                segmentContourArea = detect_size(thresh_seg)
                                size_deviation = abs(ref_features[i][4] - segmentContourArea)/ ref_features[i][4]
                                print("Size Deviation",size_deviation)
                                # Ta = 0.15       #Size Deviation Threshold- Experimental
                                Ta = 0.12
                                if size_deviation >= Ta:
                                        size_defect = {
                                                "type": "Size Defect",
                                                "status" : "Size Deviation = " + str(size_deviation)+ ". Exceeded threshold. "
                                        }
                                        no_def_segs += 1
                                        size_def.append([i, test_seg])
                                        size_defect_json = json.dumps(size_defect)
                                else:
                                        #Detect and Compare Rotation
                                        rotation_measure = detect_rotation(moments)
                                        rotation_deviation = abs((ref_features[i][5] - rotation_measure)/ref_features[i][5])
                                        print("Rotation Deviation : ", rotation_deviation)
                                        Tr = 0.6

                                        # Tr = 100000

                                        if rotation_deviation >= Tr:
                                                rotation_defect = {
                                                        "type": "Rotation Defect",
                                                        "status": "Size Deviation " + str(rotation_deviation) + ". Exceeded threshold. "
                                                }
                                                no_def_segs += 1
                                                rotation_defect_json = json.dumps(rotation_defect)
                                                rotation_def.append([i, test_seg])
                                        else:
                                                #Detect and Compare Placement
                                                test_img_dimensions = test_seg.shape
                                                xg = int(test_img_dimensions[1]/2)
                                                yg = int(test_img_dimensions[0]/2)
                                                test_seg_center = [xg, yg]
                                                testseg_placement_measures = detect_placement(moments,test_seg_center)
                                                angle_deviation = ((ref_features[i][6][1]-testseg_placement_measures[1])**2)/ref_features[i][6][1]
                                                distance_deviation = ((ref_features[i][6][0] - testseg_placement_measures[0])**2)/ref_features[i][6][0]
                                                total_deviation = np.sqrt(angle_deviation + distance_deviation)
                                                print("Placement Deviation Measure : " ,total_deviation)
                                                # Tp = 0.5
                                                Tp = 100000
                                                if total_deviation >= Tp:
                                                        placement_defect = {
                                                                "type" : "Placement Defect",
                                                                "status" : "Placement measure deviation "+ str(total_deviation) + " . Exceeded threshold"
                                                        }
                                                        placement_defect_json = json.dumps(placement_defect)
                                                        no_def_segs += 1
                                                        placement_def.append([i, test_seg])
                                                else:
                                                        #Detect and Compare Color
                                                        ret1, gr_test_seg_thresh = cv2.threshold(gr_test_seg, 5, 255, cv2.THRESH_BINARY)
                                                        cv2.imshow("thresholded seg", gr_test_seg_thresh)
                                                        cv2.waitKey(0)
                                                        test_seg_color_measures = detect_color(test_seg, gr_test_seg_thresh)
                                                        # print(test_seg_color_measures)
                                                        all_roi_deviations = []
                                                        seg_color_measures = ref_features[i][7]

                                                        for k in range(len(ref_features[i][7])):
                                                                roi_deviations = []
                                                                for l in range(len(ref_features[i][7][k])):
                                                                        rm = ref_features[i][7][k]
                                                                        tm = test_seg_color_measures[k]
                                                                        # print(rm[l][0][0])
                                                                        # print(tm[l][0][0])
                                                                        channel_dev = (rm[l][0][0] - tm[l][0][0])**2 + (rm[l][0][1] - tm[l][0][1])**2 + (rm[l][0][2] - tm[l][0][2])**2
                                                                        channel_dev = channel_dev/ (rm[l][0][0]**2 + rm[l][0][1]**2 + rm[l][0][2]**2)
                                                                        # channel_dev = rm[l][0][0] + tm[l][0][0]
                                                                        roi_deviations.append(channel_dev)
                                                                # print(roi_deviations)
                                                                all_roi_deviations.append(roi_deviations)
                                                        # print(len(all_roi_deviations))
                                                        print(all_roi_deviations)

                                                        Tcol = 0.09
                                                        # Tcol = 10000000
                                                        color_def_roi = []
                                                        for b in range(len(all_roi_deviations)):
                                                                for bi in range(len(all_roi_deviations[b])):
                                                                        if all_roi_deviations[b][bi] >= Tcol:
                                                                                color_def_roi.append(b)
                                                        if len(color_def_roi) != 0:
                                                                color_defect = {
                                                                        "type": "Color Defect",
                                                                        "status": "Color deviation occurred in rois : " + str(color_def_roi)

                                                                }

                                                                no_def_segs += 1
                                                                color_def.append([i, test_seg])
                                                                shape_defect_json = json.dumps(color_defect)
                                                                print(color_defect)
                                                        else:
                                                                #Detect and Compare Minima and Maxima
                                                                defected_contours, ref_th_seg = detMinMax2(reference_thresh_segs, gr_test_seg_thresh, ref_dimensions, segmentContourArea, i)



                                                                minmax_img = test_seg
                                                                if len(defected_contours)!=0:
                                                                        diff_image = ref_th_seg - gr_test_seg_thresh
                                                                        cv2.imshow("Diff imageX", diff_image)
                                                                        cv2.waitKey(0)
                                                                        no_def_segs += 1
                                                                        minmax_def.append([i, test_seg])

                                                                # ref_curvatures = ref_features[i][8]
                                                                # print("Ref curvature values ",i,":", ref_curvatures)
                                                                # print("Test curvature values ",i,":", test_seg_curvs)
                                                                # cvals = []
                                                                #
                                                                # for k in range(len(test_seg_curvs)):
                                                                #         tlen = len(test_seg_curvs[k])
                                                                #         rlen = len(ref_curvatures[k])
                                                                #
                                                                #         if tlen >= rlen:
                                                                #                 ct_cvals = []
                                                                #                 for m in range(len(test_seg_curvs[k])):
                                                                #                         val = test_seg_curvs[k][m]
                                                                #                         for l in range(rlen):
                                                                #                                 if val == ref_curvatures[k][l]:
                                                                #                                         ct_cvals.append([m, l])
                                                                #                 cvals.append(ct_cvals)

        print("Number of defected segemnts : ",no_def_segs)
        return shape_def, size_def, placement_def, rotation_def, color_def, minmax_def

#MinMax2
def detMinMax2(ref_thresh_segs, tseg_thresh, ref_dimensions, segmentArea, n):
        print("MinMax2...", n)

        #For the ref seg
        # ref_dimensions = ref_thresh_segs[n].shape
        print(ref_dimensions)
        ref = resize_segments(ref_thresh_segs[n], ref_dimensions)
        cv2.imshow("ref",ref)
        cv2.waitKey(0)

        #For the test seg

        # cv2.imshow("original tseg ", tseg_thresh)
        # cv2.waitKey(0)
        test_dimensions = tseg_thresh.shape
        # print(test_dimensions)
        test = resize_segments(tseg_thresh, ref_dimensions)
        # cv2.imshow("test", test)
        # cv2.waitKey(0)

        diff_image = ref - test

        diff_image = cv2.merge((diff_image, diff_image, diff_image))
        diff_image = cv2.cvtColor(diff_image, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Diff Image ", diff_image)
        cv2.waitKey(0)
        gauss_diff = cv2.GaussianBlur(diff_image,(5,13), cv2.BORDER_DEFAULT)
        # cv2.imshow("Gauss Diff Image ", gauss_diff)
        # cv2.waitKey(0)

        _, thresh_gdiff= cv2.threshold(gauss_diff, 254, 255, cv2.THRESH_BINARY)
        cv2.imshow("Gauss Diff Image Thresh", thresh_gdiff)
        cv2.waitKey(0)

        cont_arr, hierarchy = cv2.findContours(thresh_gdiff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        def_contours = []
        for cnt in cont_arr:
                cnt_area = cv2.contourArea(cnt)
                if cnt_area/segmentArea >=0.006:
                        def_contours.append(cnt)

        return def_contours, ref_thresh_segs[n]


def resize_segments(thr_seg, ref_dimensions):
        nz_locs = np.argwhere(thr_seg)
        # print(nz_locs)
        x_arr = []
        y_arr = []
        for loc in nz_locs:
                y_arr.append(loc[0])
                x_arr.append(loc[1])
        x_arr = sorted(x_arr)
        y_arr = sorted(y_arr)

        x_min = x_arr[0]
        x_max = x_arr[len(x_arr)-1]
        y_min = y_arr[0]
        y_max = y_arr[len(y_arr)-1]

        crop_seg = thr_seg[ y_min:y_max , x_min:x_max ]
        cv2.imshow("Cropped ", crop_seg)
        cv2.waitKey(0)
        # cv2.destroyAllWindows()
        bg = np.zeros((ref_dimensions[0], ref_dimensions[1]), np.uint8) * 255
        x_offset = y_offset = 20

        bg[y_offset: y_offset + crop_seg.shape[0], x_offset: x_offset + crop_seg.shape[1]] = crop_seg
        # cv2.imshow("Prepped ", bg)
        # cv2.waitKey(0)

        return bg





#Shape Detection
def detect_shape(gray_seg):
        print("Shape Detection...")
        _, thresh_seg = cv2.threshold(gray_seg, 20, 255, cv2.THRESH_BINARY)

        #Moments
        moments = cv2.moments(thresh_seg)
        #Hu Moments
        huMoments = cv2.HuMoments(moments)
        #LogScale Hu Moments
        for i in range(0,7):
                if huMoments[i] !=0 :
                        huMoments[i] = -1*math.copysign(1.0, huMoments[i])* math.log10(abs(huMoments[i]))
                else:
                        huMoments[i] = 0
        huMoments = huMoments.flatten()
        return huMoments, moments, thresh_seg

#Size Detection
def detect_size(thresholded_seg):
        print("Size Detection...")
        contours, hierarchy = cv2.findContours(thresholded_seg, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        print(len(contours))
        #Calculating Total Contour Area in a segment
        segmentContourArea = 0
        no_of_contours = len(contours)
        for i in range(no_of_contours):
                segmentConotourArea = segmentContourArea + cv2.contourArea(contours[i])
        return segmentConotourArea

#Rotation Detection
def detect_rotation(moments):
        print("Rotation Detection")
        #central moments
        c_moments = [moments['mu20'], moments['mu11'], moments['mu02'], moments['mu30'], moments['mu21'], moments['mu12'], moments['mu03']]
        average_c_moments = (moments['mu20'] + moments['mu11'] + moments['mu02'] + moments['mu30'] + moments['mu21'] + moments['mu12'] + moments['mu03'])/7
        return average_c_moments

#Placement Detection
def detect_placement(moments, garment_center):
        print("Placement Detection...")
        xg = garment_center[0]
        yg = garment_center[1]

        #Segment Center of Mass
        xc = int(moments['m10']/moments['m00'])
        yc = int(moments['m01']/moments['m00'])
        segment_com = [xc,yc]

        #Distance between segment's center of mass and the center of the garment
        l = np.sqrt((xg-xc)**2+(yg-yc)**2)

        #Angle between the center of mass of the segment and the center of garment

        if (xc-xg) != 0 and (yc-yg) != 0:
                tan_value = abs((yc - yg) / (xc - xg))
                theta_rad = math.atan(tan_value)
                theta_deg = math.degrees(theta_rad)
                if(xc<xg) and (yc<yg):
                        theta = 180 - theta_deg
                elif (xc>xg) and (yc<yg):
                        theta = theta_deg
                elif (xc<xg) and (yc>yg):
                        theta = 180 + theta_deg
                elif (xc>xg) and (yc>yg):
                        theta = 360 - theta_deg
        else:
                if (xc==xg) and (yc<yg):
                        theta = 90
                elif (xc==xg) and (yc>yg):
                        theta = 270
                elif (xg>xg) and (yc==yg):
                        theta = 0
                elif (xc<xg) and (yc==yg):
                        theta = 180
                elif (xc==xg) and (yc==yg):
                        theta = -1

        segment_placement_measures = [l,theta]
        return segment_placement_measures

#Color Detection
def detect_color(segment, thresholded_segment):
        print("Color Detection...")
        # cv2.imshow('thresh',thresholded_segment)
        # cv2.waitKey(0)
        non_zero_indexes = np.argwhere(thresholded_segment)
        # print(non_zero_indexes)
        l = len(non_zero_indexes)
        min_x = non_zero_indexes[0][1]
        max_x = non_zero_indexes[0][1]
        min_y = non_zero_indexes[0][0]
        max_y = non_zero_indexes[0][0]
        for i in range(l):
                if non_zero_indexes[i][0] < min_y:
                        min_y = non_zero_indexes[i][0]
                if non_zero_indexes[i][0] > max_y:
                        max_y = non_zero_indexes[i][0]
                if non_zero_indexes[i][1] < min_x:
                        min_x = non_zero_indexes[i][1]
                if non_zero_indexes[i][1] > max_x:
                        max_x = non_zero_indexes[i][1]
        print(min_x , max_x, min_y, max_y)

        # cv2.line(segment,(min_x,min_y), (max_x,max_y),(255,0,0),3)


        segment_section = segment[min_y:max_y, min_x:max_x]

        #Converting segment section color to hsv
        seg_section_hsv = cv2.cvtColor(segment_section, cv2.COLOR_BGR2HSV)

        #Splitting the image in to 4 ROI s
        seg_section_dimensions = segment_section.shape
        split_index_v = int(seg_section_dimensions[0]/2)
        slice_1 = np.split(seg_section_hsv,[split_index_v],0)
        slice_2 = []
        for i in range(len(slice_1)):
                split_index_h = int(slice_1[i].shape[1] / 2)
                slice_2.append(np.split(slice_1[i],[split_index_h],1))

        rois = [slice_2[0][0], slice_2[0][1], slice_2[1][0],slice_2[1][1]]
        segment_color_measures = []
        for j in range(len(rois)):
                #Splitting into Color Channels
                channels = cv2.split(rois[j])
                # print(channels)
                roi_measures = []

                for i in range(len(channels)):
                        channel_measures = []
                        sorted_channel = np.sort(channels[i])
                        color_median = np.median(sorted_channel)
                        #Color channel Standard Deviation
                        color_std_dev = np.std(channels[i])
                        color_mean = np.mean(channels[i])
                        color_skewness = 3*(color_mean-color_median)/color_std_dev
                        color_variance = np.var(channels[i])
                        channel_measures.append([color_mean, color_variance,color_skewness])
                        # print(channel_measures)
                        roi_measures.append(channel_measures)
                segment_color_measures.append(roi_measures)
        # print(segment_color_measures)
        return segment_color_measures


        # cv2.imshow('ori',segment)
        # cv2.waitKey(0)
        # cv2.imshow('Hi',segment_section)
        # cv2.waitKey(0)
        #
        # cv2.imshow('hsv',seg_section_hsv)
        # cv2.waitKey(0)

        # for j in range(len(rois)):
        #         cv2.imshow('roi'+str(j), rois[j])
        #         cv2.waitKey(0)
        #
        # cv2.destroyAllWindows()




def display_arr(arr, type):
        def_count = []
        for a in arr:
                def_count.append(a[0])
        print(str(type) + " defect segments : ", def_count)


#Function Usage

# check_artwork_position(ref_artwork_loc, ref_or_cloth_loc, test_artwork_mask_loc, test_or_cloth_loc)
match_segments(nonmatching_ref_loc, nonmatching_test_loc, matching_ref_loc, matching_test_loc)
ref_features, ref_thresholded_segs, ref_dimensions, ref_segs = detect_features(no_of_matching_ref_segs, 1)
shape_def, size_def, placement_def, rotation_def, color_def, minmax_def = detect_and_compare_matching_segments(no_of_matching_test_segs, ref_features, 1, ref_thresholded_segs, ref_dimensions, ref_segs)

display_arr(shape_def, "Shape")
display_arr(size_def, "Size")
display_arr(placement_def,"Placement")
display_arr(rotation_def,"Rotation")
display_arr(color_def,"Color")
display_arr(minmax_def,"Minima Maxima")
