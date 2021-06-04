import numpy as np
import cv2
import math
import statistics as stat
from matplotlib import pyplot as plt
import os
import json

# matching_ref_loc ="./Assets/Seg_Module/Output/tex_girlpig/defect_1/matching_segments/reference/"
# matching_test_loc = "./Assets/Seg_Module/Output/tex_girlpig/defect_1/matching_segments/defect/"
# nonmatching_ref_loc = "./Assets/Seg_Module/Output/tex_girlpig/defect_1/none_matching_segments/reference/"
# nonmatching_test_loc = "./Assets/Seg_Module/Output/tex_girlpig/defect_1/none_matching_segments/test/"
# nonmatching_ref_conflict = "./Assets/Seg_Module/Output/tex_girlpig/defect_1/conflict/ref/"
# nonmatching_test_conflict = "./Assets/Seg_Module/Output/tex_girlpig/defect_1/conflict/test/"

# #ref artwork & cloth loc
# ref_artwork_mask_loc = "./Assets/BR_Module/Output/mask/ref/artwork/"
# ref_or_cloth_loc = "./Assets/BR_Module/Output/mask/ref/cloth/"  #outer removed
# ref_or_cloth_loc = "./Assets/BR_Module/Output/tex_rainbow.jpg"
#
# #test artwork &cloth loc
# test_artwork_mask_loc = "Assets/BR_Module/Output/mask/test/artwork/"
# test_or_cloth_loc = "Assets/BR_Module/Output/mask/test/cloth/"  #outer removed
#
# #ref isolated artwork loc
# ref_artwork_loc = "./Assets/BR_Module/Output/ref/isolated_artwork/"
#
# #might need to be adjusted as per segment rois
# ref_seg_roi_loc = "./Assets/QA_Module/Output/rois"
# test_seg_roi_loc = "./Assets/QA_Module/Output/rois"
#
# # matching_ref
# mr_file_list = os.listdir(matching_ref_loc)
# no_of_matching_ref_segs = len(mr_file_list)
#
# #matching_test
# mt_file_list = os.listdir(matching_test_loc)
# no_of_matching_test_segs = len(mt_file_list)
#
# #nonmatching_ref
# if os.path.exists(nonmatching_ref_loc):
#         nmr_file_list = os.listdir(nonmatching_ref_loc)
#         no_of_nonmatching_ref_segs = len(nmr_file_list)
# else:
#         no_of_nonmatching_ref_segs = 0
#
# #non_matching_test
# if os.path.exists(nonmatching_test_loc):
#         nmt_file_list = os.listdir(nonmatching_test_loc)
#         no_of_nonmatching_test_segs = len(nmt_file_list)
# else:
#         no_of_nonmatching_test_segs = 0
#
#
# #non_matching_ref_conflict
# if os.path.exists(nonmatching_ref_conflict):
#         nmr_conflict_file_list = os.listdir(nonmatching_ref_conflict)
#         no_of_ref_conflict_segs = len(nmr_conflict_file_list)
# else:
#         no_of_ref_conflict_segs = 0
#
# #non_matching_test_conflict
# if os.path.exists(nonmatching_test_conflict):
#         nmt_conflict_file_list = os.listdir(nonmatching_test_conflict)
#         no_of_test_conflict_segs = len(nmt_conflict_file_list)
# else:
#         no_of_test_conflict_segs = 0


def match_segments(nm_ref_loc, nm_test_loc, m_ref_loc, m_test_loc, no_of_nonmatching_ref_segs, no_of_test_conflict_segs, nmr_file_list, no_of_nonmatching_test_segs, nmt_file_list,
                   ref_or_cloth_loc, no_of_ref_conflict_segs, nmr_conflict_file_list, nmt_conflict_file_list, nonmatching_ref_conflict, nonmatching_test_conflict, matching_ref_loc, matching_test_loc, ref_artwork_loc, test_artwork_loc):
        # To be displayed in ui
        common_def_image = cv2.imread(test_artwork_loc)  # Add the test segment location path here ----------------------------------------------------------------
        print("Conflict Segment Matching Started...........................................................")
        print(nm_test_loc)
        if no_of_nonmatching_ref_segs != 0 and no_of_test_conflict_segs == 0:
                print("Missing segment in test artwork")
                for segf in nmr_file_list:
                        # if segf.endswith('.jpg') or segf.endswith('.jpeg'):
                                print("File", segf)
                                def_seg = cv2.imread(nm_ref_loc + segf)
                                def_seg_gr = cv2.cvtColor(def_seg, cv2.COLOR_BGR2GRAY)
                                def_seg_thresh = cv2.threshold(def_seg_gr, 10, 255, cv2.THRESH_BINARY)
                                print("def thresh seg",def_seg_thresh)
                                # if def_seg is not None:
                                # print("segf",def_seg)
                                def_seg_disp = cv2.resize(def_seg, (1200,900))
                                #--------------------------
                                # common_def_image = mark_defect(common_def_image, cv2.bitwise_not(def_seg_thresh))
                                #--------------------------
                                cv2.imshow("Missing segments", common_def_image)
                                cv2.waitKey(0)
                cv2.destroyAllWindows()
        elif no_of_nonmatching_test_segs != 0:
                print("Checking for Color Patch/ Fade or Damaged printwork")
                for segf in nmt_file_list:
                                print("Color patch inspection..........................")
                        # if segf.endswith('.jpg') or segf.endswith('.jpeg'):
                                print(segf)
                                def_seg = cv2.imread(nm_test_loc + segf) #------------------------------------------------------------
                                def_seg_nz = np.argwhere(def_seg)
                                defx = []
                                defy = []
                                for pt in def_seg_nz:
                                        defx.append(pt[1])
                                        defy.append(pt[0])
                                defx = sorted(defx)
                                defy = sorted(defy)
                                defx_min = defx[0]
                                defx_max = defx[len(defx) - 1]
                                defy_min = defy[0]
                                defy_max = defy[len(defy) - 1]
                                def_seg_section = def_seg[defy_min:defy_max, defx_min:defx_max]
                                ref_image = cv2.imread(ref_artwork_loc)
                                ref_dims = ref_image.shape

                                #Image for the comparison with the ref artwork
                                comp_image = np.zeros((ref_dims[0], ref_dims[1], 1), np.uint8)*255
                                comp_image = cv2.merge((comp_image, comp_image, comp_image))

                                for pos in def_seg_nz:
                                        comp_image[pos[0]][pos[1]] = ref_image[pos[0]][pos[1]]

                                comp_disp = cv2.resize(comp_image, (900,1200))
                                cv2.imshow("Comp_image init", comp_disp)
                                cv2.waitKey(0)

                                # ref_image_section = ref_image[defy_min:defy_max, defx_min:defx_max]
                                # cv2.imshow("ref_image_section", ref_image_section)

                                #Average color of segf
                                (B, G, R) = cv2.split(def_seg)
                                segf_mean = (np.mean(B) + np.mean(G) + np.mean(R))/3
                                print("Defect area color mean :", segf_mean)

                                #Average color of ref_image
                                (Bc, Gc, Rc) = cv2.split(comp_image)
                                comp_image_mean = (np.mean(Bc) + np.mean(Gc) + np.mean(Rc))/3
                                print("Reference area color mean :", comp_image_mean)

                                mean_deviation_measure = abs(segf_mean - comp_image_mean)/segf_mean
                                print("Mean deviation measure ", mean_deviation_measure)

                                if mean_deviation_measure >= 0.07:
                                        print("Color patch detected...")
                                        for pos in def_seg_nz:
                                                common_def_image[pos[0]][pos[1]] = [0,255,0]
                                        for pos in def_seg_nz:
                                                comp_image[pos[0]][pos[1]] = [0,255,0]

                                        common_def_disp = cv2.resize(common_def_image,(900,1200))
                                        cv2.imshow("Comp_image detected", common_def_disp)
                                        cv2.waitKey(0)
                                else:
                                        print("Damaged Printwork detected...")
                                        for pos in def_seg_nz:
                                                common_def_image[pos[0]][pos[1]] = [0,255,0]
                                        common_def_disp = cv2.resize(common_def_image, (900, 1200))
                                        cv2.imshow("Common def image", common_def_disp)
                                        cv2.waitKey(0)

                                def_seg_disp = cv2.resize(def_seg, (900,1200))
                                cv2.imshow(segf , def_seg)
                                cv2.waitKey(0)
                                cv2.destroyAllWindows()
        elif no_of_ref_conflict_segs != 0:
                print("Hit....................................")
                for i in range(len(nmr_conflict_file_list)):
                        nmrc_seg = cv2.imread(nonmatching_ref_conflict + "C_" + str(i))
                        nmrc_seg_gr = cv2.cvtColor(nmrc_seg, cv2.COLOR_BGR2GRAY)

                        #Reference keypoints
                        rc_kp = []
                        matching_kp_list = []
                        sift = cv2.SIFT_create()
                        kprc, desr = sift.detectAndCompute(nmrc_seg_gr, None)
                        tc_segs = []
                        for j in range(len(nmt_conflict_file_list)):
                                nmtc_seg = cv2.imread(nonmatching_test_conflict + "N_"+str(i)+str(j))
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
                        cv2.imwrite(matching_ref_loc+"M_"+str(curr_files), nmrc_seg)
                        os.remove(nonmatching_ref_conflict+"C_"+str(i))
                        curr_matching_test_list = os.listdir(matching_test_loc)
                        cv2.imwrite(matching_test_loc+"M_"+str(curr_files), tc_segs[best_match])
                        os.remove(nonmatching_test_conflict + "C_" + str(i) + str(x))
        print("Conflict Segment Matching Done...............................................")





def detect_features(no_of_matching_ref_segs, ref_img_check, matching_ref_loc, matching_test_loc, ref_or_cloth_loc):
        print("Reference Image Feature Extraction Started.........................................")

        ref_features=[]
        thresholded_segments = []
        ref_segs = []

        for i in range(no_of_matching_ref_segs):
                print("Reference Image Segment ", str(i))
                seg_features = []
                if ref_img_check == 1:
                        seg = cv2.imread(matching_ref_loc+"M"+"_"+str(i)+".jpg")
                else:
                        seg = cv2.imread(matching_test_loc + "M" + "_" + str(i) + ".jpg")
                ref_segs.append(seg)
                gr_seg = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)

                #Detect Shape
                huMoments, moments, thresh_seg = detect_shape(gr_seg)
                print("Hu 7", huMoments[6])
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

                outer_removed_img = cv2.imread(ref_or_cloth_loc)
                segment_placement_measures = detect_placement(moments, garment_center, outer_removed_img)
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
        print("Reference Features :",len(ref_features))
        return ref_features, thresholded_segments, dimensions, ref_segs

def detect_and_compare_matching_segments(no_of_segments,ref_features,test_img_check, reference_thresh_segs, ref_dimensions, ref_segs, no_of_matching_test_segs, matching_test_loc, test_or_cloth_loc, test_artwork_loc):
        print("Detect and Match test image stage reached...........................")

        no_def_segs = 0

        shape_def = []
        color_def = []
        placement_def = []
        rotation_def = []
        size_def = []
        minmax_def = []

        #To be displayed in ui
        common_def_image = cv2.imread(test_artwork_loc)  # Add the test segment location path here ----------------------------------------------------------------

        shape_def_image = cv2.imread(test_artwork_loc)
        size_def_image = cv2.imread(test_artwork_loc)
        rotation_def_image = cv2.imread(test_artwork_loc)
        placement_def_image = cv2.imread(test_artwork_loc)
        color_def_image = cv2.imread(test_artwork_loc)
        minmax_def_image = cv2.imread(test_artwork_loc)

        def_dict = {
                "shape": shape_def_image,
                "size": size_def_image,
                "rotation": rotation_def_image,
                "placement": placement_def_image,
                "boundary": minmax_def_image,
                "color": color_def_image,
                "common": common_def_image
        }


        for i in range(no_of_matching_test_segs):
                print(no_of_matching_test_segs)

                print("Matching Test Segment :", str(i))
                if test_img_check == 1:
                        test_seg = cv2.imread(matching_test_loc+"M"+"_"+str(i)+".jpg")

                gr_test_seg = cv2.cvtColor(test_seg, cv2.COLOR_BGR2GRAY)

                # Detect and Compare Shape
                huMoments, moments, thresh_seg = detect_shape(gr_test_seg)
                print("Hu7", huMoments[6])


                if ref_features[i][0]*huMoments[6] < 0 and abs(ref_features[i][0]/huMoments[6]) <= 1.05 and abs(ref_features[i][0]/huMoments[6]) >= 0.95:
                        shape_defect = {
                                "type": "Shape Error",
                                "status": "Mirrored segment"
                        }
                        print(shape_defect)
                        shape_defect_json = json.dumps(shape_defect)
                        shape_defect_image = mark_defect(test_seg, thresh_seg)
                        common_def_image = mark_defect(common_def_image, thresh_seg)
                        no_def_segs += 1
                        shape_def.append([i, test_seg])

                        # -----------------displaying the defected segment in the test artwork
                        shape_def_image = mark_defect(shape_def_image, thresh_seg)
                        def_dict["shape"] = shape_def_image
                        shape_def_disp = cv2.resize(shape_def_image, (750, 1000))
                        cv2.imshow("Shape Defect View", shape_def_disp )
                        cv2.waitKey(0)

                        shape_saved = cv2.imwrite("./Assets/QA_Module/Output/Shape/shape.jpg", shape_def_disp)
                        cv2.destroyAllWindows()
                        # ---------------------------------------------------------------------
                else:
                        #shape_deviation_measure
                        shape_deviation = ((ref_features[i][1]-huMoments[0])**2 + (ref_features[i][2]-huMoments[1])**2
                                           + (ref_features[i][3]-huMoments[2])**2 )/(ref_features[i][1]+ref_features[i][2]+ref_features[i][3])
                        print("Shape Deviation : ",shape_deviation)
                        Ts = 0.15       #Shape Deviation Threshold - Experimental

                        if shape_deviation >= Ts:
                                shape_defect = {
                                        "type": "Shape Defect",
                                        "status" : "Shape deviation = "+str(shape_deviation)+". Exceeded threshold."
                                }
                                no_def_segs += 1
                                shape_def.append([i, test_seg])

                                shape_defect_image = mark_defect(test_seg, thresh_seg)
                                shape_defect_json = json.dumps(shape_defect)
                                common_def_image = mark_defect(common_def_image, thresh_seg)

                                # -----------------displaying the defected segment in the test artwork
                                shape_def_image = mark_defect(shape_def_image, thresh_seg)
                                def_dict["shape"] = shape_def_image
                                shape_def_disp = cv2.resize(shape_def_image, (750, 1000))
                                cv2.imshow("Shape Defect View", shape_def_disp)
                                shape_saved = cv2.imwrite("./Assets/QA_Module/Output/Shape/shape.jpg", shape_def_disp)

                                cv2.waitKey(0)
                                cv2.destroyAllWindows()
                                # ---------------------------------------------------------------------
                        else:
                                #Detect and Compare Size
                                segmentContourArea = detect_size(thresh_seg)
                                size_deviation = abs(ref_features[i][4] - segmentContourArea)/ ref_features[i][4]
                                print("Size Deviation",size_deviation)
                                Ta = 0.12       #Size Deviation Threshold- Experimental
                                if size_deviation >= Ta:
                                        size_defect = {
                                                "type": "Size Defect",
                                                "status" : "Size Deviation = " + str(size_deviation)+ ". Exceeded threshold. "
                                        }
                                        no_def_segs += 1
                                        size_def.append([i, test_seg])

                                        size_defect_image = mark_defect(test_seg, thresh_seg)
                                        size_defect_json = json.dumps(size_defect)
                                        common_def_image = mark_defect(common_def_image, thresh_seg)

                                        # -----------------displaying the defected segment in the test artwork
                                        size_def_image = mark_defect(size_def_image, thresh_seg)
                                        def_dict["size"] = size_def_image
                                        size_def_disp = cv2.resize(size_def_image, (750, 1000))

                                        size_saved = cv2.imwrite("./Assets/QA_Module/Output/Size/size.jpg", size_def_disp)

                                        cv2.imshow("Size Defect View", size_def_disp)

                                        cv2.waitKey(0)
                                        cv2.destroyAllWindows()
                                        # ---------------------------------------------------------------------
                                else:
                                        #Detect and Compare Rotation
                                        rotation_measure = detect_rotation(moments)
                                        rotation_deviation = abs(ref_features[i][5] - rotation_measure)/(1+ ref_features[i][5] + rotation_measure)
                                        print("Rotation Deviation : ", rotation_deviation)
                                        Tr = 0.34

                                        if rotation_deviation >= Tr:
                                                rotation_defect = {
                                                        "type": "Rotation Defect",
                                                        "status": "Size Deviation " + str(rotation_deviation) + ". Exceeded threshold. "
                                                }
                                                no_def_segs += 1
                                                rotation_defect_json = json.dumps(rotation_defect)
                                                rotation_def.append([i, test_seg])
                                                rotation_defect_image = mark_defect(test_seg, thresh_seg)
                                                common_def_image = mark_defect(common_def_image, thresh_seg)
                                                # -----------------displaying the defected segment in the test artwork
                                                rotation_def_image = mark_defect(rotation_def_image, thresh_seg)
                                                def_dict["rotation"] = rotation_def_image
                                                rotation_def_disp = cv2.resize(rotation_def_image, (750, 1000))
                                                cv2.imshow("Rotation Defect View", rotation_def_disp)
                                                rotation_saved = cv2.imwrite("./Assets/QA_Module/Output/Rotation/rotation.jpg",rotation_def_disp)
                                                cv2.waitKey(0)
                                                cv2.destroyAllWindows()
                                                # ---------------------------------------------------------------------

                                        else:
                                                #Detect and Compare Placement
                                                test_img_dimensions = test_seg.shape
                                                xg = int(test_img_dimensions[1]/2)
                                                yg = int(test_img_dimensions[0]/2)
                                                test_seg_center = [xg, yg]
                                                test_img_or = cv2.imread(test_or_cloth_loc)
                                                testseg_placement_measures = detect_placement(moments,test_seg_center, test_img_or)
                                                angle_deviation = ((ref_features[i][6][1]-testseg_placement_measures[1])**2)/ref_features[i][6][1]
                                                distance_deviation = ((ref_features[i][6][0] - testseg_placement_measures[0])**2)/ref_features[i][6][0]
                                                total_deviation = np.sqrt(abs(angle_deviation - distance_deviation))
                                                print("Placement Deviation Measure : " ,total_deviation)
                                                Tp = 0.3

                                                if total_deviation >= Tp:
                                                        placement_defect = {
                                                                "type" : "Placement Defect",
                                                                "status" : "Placement measure deviation "+ str(total_deviation) + " . Exceeded threshold"
                                                        }
                                                        placement_defect_json = json.dumps(placement_defect)
                                                        no_def_segs += 1
                                                        # placement_defect_image = mark_defect(test_seg, thresh_seg)
                                                        placement_def.append([i, test_seg])
                                                        common_def_image = mark_defect(common_def_image, thresh_seg)

                                                        # -----------------displaying the defected segment in the test artwork
                                                        placement_def_image = mark_defect(placement_def_image, thresh_seg)
                                                        def_dict["placement"] = placement_def_image
                                                        placement_def_disp = cv2.resize(placement_def_image, (750, 1000))

                                                        placement_saved = cv2.imwrite("./Assets/QA_Module/Output/Placement/placement.jpg", placement_def_disp)
                                                        cv2.imshow("Placement Defect View", placement_def_disp)
                                                        cv2.waitKey(0)
                                                        cv2.destroyAllWindows()
                                                        #---------------------------------------------------------------------


                                                else:
                                                        # Detect and Compare Minima and Maxima
                                                        defected_contours, ref_th_seg, seg_thresh_gdiff, test_translate, prep_diff, det_diff = detMinMax2(
                                                                reference_thresh_segs, thresh_seg, ref_dimensions,
                                                                segmentContourArea, i)

                                                        minmax_img = test_seg
                                                        if len(defected_contours) != 0:
                                                                # diff_image = ref_th_seg - gr_test_seg_thresh
                                                                # cv2.imshow("Diff imageX", diff_image)
                                                                cv2.waitKey(0)
                                                                # To draw the test seg's minima maxima defect---------------
                                                                disp_img_minmax = test_seg
                                                                cv2.drawContours(seg_thresh_gdiff, defected_contours,
                                                                                 -1, (0, 0, 255), 2)
                                                                # cv2.drawContours(disp_img_minmax, defected_contours, -1,
                                                                #                  (0, 255, 0), cv2.FILLED)

                                                                for ct in defected_contours:
                                                                        xd, yd, wd, hd = cv2.boundingRect(ct)
                                                                        cv2.rectangle(minmax_def_image, (xd+test_translate[2]-10 ,yd+test_translate[0]-10), (xd+wd+test_translate[2]+10,yd+hd+test_translate[0]+10), (0,0,255),3)

                                                                print("Prepdiff shape ", prep_diff.shape)
                                                                # cv2.imshow("prepdiff from dnm", prep_diff) -------------
                                                                #
                                                                # cv2.waitKey(0)
                                                                # prept_diff = cv2.merge((prept_diff,prept_diff,prept_diff))

                                                                # temp_image = common_def_image


                                                                # -------------------Displaying defect-------------
                                                                print("Test translate ", test_translate)
                                                                prep_diff_nz = np.argwhere(prep_diff)
                                                                # print(prep_diff_nz)
                                                                for pt in prep_diff_nz:
                                                                        pt[0] = pt[0] + test_translate[0]
                                                                        pt[1] = pt[1] + test_translate[2]

                                                                new_pos = prep_diff_nz

                                                                for px_pos in new_pos:
                                                                        common_def_image[px_pos[0]][px_pos[1]] = [0,0,255]
                                                                        # minmax_def_image[px_pos[0]][px_pos[1]] = [0,0,255]

                                                                def_dict["boundary"] = minmax_def_image

                                                                minmax_def_disp = cv2.resize(minmax_def_image, (750,1000))

                                                                boundary_saved = cv2.imwrite("./Assets/QA_Module/Output/Boundary/boundary.jpg",minmax_def_disp)
                                                                cv2.imshow("Minmax Defect View", minmax_def_disp)
                                                                cv2.waitKey(0)
                                                                cv2.destroyAllWindows()

                                                                no_def_segs += 1
                                                                minmax_def.append([i, test_seg])


                                                        else:
                                                                #Detect and Compare Color
                                                                ret1, gr_test_seg_thresh = cv2.threshold(gr_test_seg, 5, 255, cv2.THRESH_BINARY)
                                                                # cv2.imshow("thresholded seg", gr_test_seg_thresh)
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
                                                                                # channel_dev = abs(rm[l][0][0] - tm[l][0][0])**2 + abs(rm[l][0][1] - tm[l][0][1]) + abs(rm[l][0][2] - tm[l][0][2])**2
                                                                                channel_dev = abs(rm[l][0][0] - tm[l][0][0]) + abs(rm[l][0][1] - tm[l][0][1]) + abs(rm[l][0][2] - tm[l][0][2])
                                                                                channel_dev = channel_dev/ (rm[l][0][0]**2 + rm[l][0][1]**2 + rm[l][0][2]**2)
                                                                                # channel_dev = rm[l][0][0] + tm[l][0][0]
                                                                                roi_deviations.append(channel_dev)
                                                                        # print(roi_deviations)
                                                                        all_roi_deviations.append(roi_deviations)
                                                                # print(len(all_roi_deviations))
                                                                print(all_roi_deviations)

                                                                Tcol = 4.3
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
                                                                        color_def_image = mark_defect(color_def_image, thresh_seg)
                                                                        common_def_image = mark_defect(common_def_image, thresh_seg)
                                                                        print(color_defect)

                                                                        #------------------------------------------
                                                                        color_def_image = mark_defect(color_def_image, thresh_seg)

                                                                        def_dict["color"] = color_def_image
                                                                        color_def_disp = cv2.resize(color_def_image,(750,1000))

                                                                        color_saved = cv2.imwrite("./Assets/QA_Module/Output/Color/color.jpg", color_def_disp)
                                                                        cv2.imshow("Color Defect View", color_def_disp)
                                                                        cv2.waitKey(0)
                                                                        cv2.destroyAllWindows()
                                                                        #------------------------------------------

        print("Number of defected segemnts : ",no_def_segs)

        def_dict["common"] = color_def_image
        common_def_disp = cv2.resize(common_def_image, (1008,1344))
        cv2.imshow("Common image ", common_def_disp)
        cv2.waitKey(0)

        shape_saved = cv2.imwrite("./Assets/QA_Module/Output/Size/shape.jpg", cv2.resize(def_dict["shape"],(750,1000)))
        size_saved = cv2.imwrite("./Assets/QA_Module/Output/Size/size.jpg", cv2.resize(def_dict["size"],(750,1000)))
        rotation_saved = cv2.imwrite("./Assets/QA_Module/Output/Rotation/rotation.jpg",cv2.resize(def_dict["rotation"],(750,1000)))
        placement_saved = cv2.imwrite("./Assets/QA_Module/Output/Placement/placement.jpg",cv2.resize(def_dict["placement"],(750,1000)))
        boundary_saved = cv2.imwrite("./Assets/QA_Module/Output/Boundary/boundary.jpg",cv2.resize(def_dict["boundary"],(750,1000)))
        color_saved = cv2.imwrite("./Assets/QA_Module/Output/Color/color.jpg",cv2.resize(def_dict["color"],(750,1000)))
        common_saved = cv2.imwrite("./Assets/QA_Module/Output/Common/common.jpg",cv2.resize(def_dict["common"],(750,1000)))



        return shape_def, size_def, placement_def, rotation_def, color_def, minmax_def

#MinMax2
def detMinMax2(ref_thresh_segs, tseg_thresh, ref_dimensions, segmentArea, n):
        print("MinMax2...", n)

        #For the ref seg
        # ref_dimensions = ref_thresh_segs[n].shape
        print(ref_dimensions)
        ref, ref_translate, rcrop_seg = resize_segments(ref_thresh_segs[n], ref_dimensions)
        # cv2.imshow("ref",ref)
        # cv2.waitKey(0)

        #For the test seg

        # cv2.imshow("original tseg ", tseg_thresh)
        # cv2.waitKey(0)
        test_dimensions = tseg_thresh.shape
        # print(test_dimensions)
        test, test_translate, tcrop_seg = resize_segments(tseg_thresh, ref_dimensions)
        # cv2.imshow("test", test)
        # cv2.waitKey(0)

        diff_image = ref- test

        diff_image = cv2.merge((diff_image, diff_image, diff_image))
        diff_image = cv2.cvtColor(diff_image, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("Diff Image ", diff_image)-------------------------------------------------
        # cv2.waitKey(0)
        # cv2.imshow("Diff Image ", diff_image)
        # cv2.waitKey(0)
        gauss_diff = cv2.GaussianBlur(diff_image,(3,3), cv2.BORDER_DEFAULT)
        # cv2.imshow("Gauss Diff Image ", gauss_diff)
        # cv2.waitKey(0)

        _, thresh_gdiff= cv2.threshold(gauss_diff, 128, 255, cv2.THRESH_BINARY)
        # cv2.imshow("Gauss Diff Image Thresh", thresh_gdiff)-----------------------------------
        # cv2.waitKey(0)
        # cv2.imshow("Gauss Diff Image Thresh", thresh_gdiff)-----------------
        # cv2.waitKey(0)

        cont_arr, hierarchy = cv2.findContours(thresh_gdiff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # cont_arr, hierarchy = cv2.findContours(diff_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        #----------------------------------------------------------------------------------------------

        rcrop_seg = cv2.merge((rcrop_seg, rcrop_seg, rcrop_seg))
        tcrop_seg = cv2.merge((tcrop_seg, tcrop_seg, tcrop_seg))

        prepr_diff = np.zeros((rcrop_seg.shape[0] + 20, rcrop_seg.shape[1] + 20, 3), np.uint8) * 255
        prept_diff = np.zeros((rcrop_seg.shape[0] + 20, rcrop_seg.shape[1] + 20, 3), np.uint8) * 255

        prepr_diff[0:rcrop_seg.shape[0], 0:rcrop_seg.shape[1]] = rcrop_seg
        prept_diff[0:tcrop_seg.shape[0], 0:tcrop_seg.shape[1]] = tcrop_seg

        prep_diff = prepr_diff - prept_diff
        #----------------------------------------------------------------------------------------------
        if len(cont_arr) == 0:
                diff_image = test - ref

                diff_image = cv2.merge((diff_image, diff_image, diff_image))
                diff_image = cv2.cvtColor(diff_image, cv2.COLOR_BGR2GRAY)
                # cv2.imshow("Diff Image ", diff_image)-------------------------------------------------
                # cv2.waitKey(0)
                # cv2.imshow("Diff Image ", diff_image)------------------------------------
                # cv2.waitKey(0)
                gauss_diff = cv2.GaussianBlur(diff_image, (3, 3), cv2.BORDER_DEFAULT)
                # cv2.imshow("Gauss Diff Image ", gauss_diff)
                # cv2.waitKey(0)

                _, thresh_gdiff = cv2.threshold(gauss_diff, 128, 255, cv2.THRESH_BINARY)
                # cv2.imshow("Gauss Diff Image Thresh", thresh_gdiff)-----------------------------------
                # cv2.waitKey(0)
                # cv2.imshow("Gauss Diff Image Thresh", thresh_gdiff)
                # cv2.waitKey(0)

                cont_arr, hierarchy = cv2.findContours(thresh_gdiff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                prep_diff = prept_diff - prepr_diff


        print(n,"    ", len(cont_arr))

#Extra Operator-----------------------------------------------------------------------------------------------------------------
        ult_ctrs = []
        # for a in range(len(cont_arr)):
        #         if(len(cont_arr) != 1 or len(cont_arr) != 0):
        #                 for c in range(len(cont_arr[a]) - 1):
        #                         for j in range(-2,3):
        #                                 for k in range(-2,3):
        #                                         pt = cont_arr[a][c].flatten()
        #                                         if thresh_gdiff[pt[0]+j][pt[1]+k] == 255:
        #                                                 if a+1 != (len(cont_arr) - 1):
        #                                                         if [[pt[0]+j,pt[1]+k]] in cont_arr[a+1]:
        #                                                                 ult_ctrs.append([cont_arr[a] ,cont_arr[a+1]])
        #         else:
        #                 ult_ctrs.append([cont_arr[a]])
        #
        # def_ult_contours = []
        # for u in ult_ctrs:
        #         if len(u) != 1:
        #                 ctr_area = cv2.contourArea(u[0]) + cv2.contourArea(u[1])
        #         else:
        #                 ctr_area = cv2.contourArea(u[0])
        #         if ctr_area/segmentArea >= 0.008:
        #                 def_ult_contours.append(u)
#Extra Operator End--------------------------------------------------------------------------------------------------------------

        def_contours = []
        for cnt in cont_arr:
                cnt_area = cv2.contourArea(cnt)
                area_percentage = cnt_area/segmentArea
                if area_percentage >= 0.0053 :
                        def_contours.append(cnt)
                        print("area% ", area_percentage)

        # rcrop_seg = cv2.merge((rcrop_seg, rcrop_seg, rcrop_seg))
        # tcrop_seg = cv2.merge((tcrop_seg, tcrop_seg, tcrop_seg))
        #
        #
        # prepr_diff = np.zeros((rcrop_seg.shape[0] + 20, rcrop_seg.shape[1] + 20, 3), np.uint8) * 255
        # prept_diff = np.zeros((rcrop_seg.shape[0] + 20, rcrop_seg.shape[1] + 20, 3), np.uint8) * 255
        #
        # prepr_diff[0:rcrop_seg.shape[0], 0:rcrop_seg.shape[1]] = rcrop_seg
        # prept_diff[0:tcrop_seg.shape[0], 0:tcrop_seg.shape[1]] = tcrop_seg

        # det_diff = prepr_diff - prept_diff
        _, det_diff = cv2.threshold(gauss_diff, 254, 255, cv2.THRESH_BINARY)


        # crop_diff = rcrop_seg - tcrop_seg
        # prep_diff = prept_diff - prepr_diff
        # cv2.imshow("Prep diff",prep_diff)
        # cv2.waitKey(0)
        # crop_diff = 0
        return def_contours, ref_thresh_segs[n], thresh_gdiff, test_translate, prep_diff, det_diff


def resize_segments(thr_seg, ref_dimensions):
        nz_locs = np.argwhere(thr_seg)
        thr_disp = cv2.resize(thr_seg, (900,1200))
        # cv2.imshow("thresh seg to check test translate", thr_disp)---
        # cv2.waitKey(0)
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
        translate_coords = [y_min, y_max, x_min, x_max]

        crop_seg = thr_seg[ y_min:y_max , x_min:x_max ]
        # cv2.imshow("Cropped ", crop_seg)-------------------------
        # cv2.waitKey(0)
        cr_dim = crop_seg.shape
        # cr_w = int(cr_dim[1] + 2)
        # cr_h = int(cr_dim[0] + 2)
        # cv2.destroyAllWindows()
        bg = np.zeros((ref_dimensions[0], ref_dimensions[1]), np.uint8) * 255
        # bg = np.zeros((cr_h, cr_w), np.uint8) * 255
        # bg_crop = np.zeros((cr_h, cr_w), np.uint8) * 255
        x_offset = y_offset = 20

        bg[y_offset: y_offset + crop_seg.shape[0], x_offset: x_offset + crop_seg.shape[1]] = crop_seg
        # cv2.imshow("Prepped ", bg)
        # cv2.waitKey(0)

        return bg, translate_coords, crop_seg

def drawDefect(def_image, test_seg, gr_test_seg):
        img1 = def_image
        img2 = test_seg
        test_seg_nz = np.argwhere(test_seg)
        for nz in test_seg_nz:
                img2[nz[0]][nz[1]] = [0,255,0]

        rows, cols, channels = img2.shape
        roi = img1[0:rows, 0:cols]

        img2_gr = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        pret, pmask = cv2.threshold(img2_gr, 10, 255, cv2.THRESH_BINARY)
        pmask_inv = cv2.bitwise_not(pmask)
        img1_bg = cv2.bitwise_and(roi, roi, mask = pmask_inv)
        img2_fg = cv2.bitwise_and(img2, img2, mask = pmask)

        dst = cv2.add(img1_bg, img2_fg)
        img1[0:rows, 0:cols] = dst

        return img1

#Shape Detection
def detect_shape(gray_seg):
        print("Shape Detection...")
        _, thresh_seg = cv2.threshold(gray_seg, 10, 255, cv2.THRESH_BINARY)

        #Moments
        moments = cv2.moments(thresh_seg)
        #Hu Moments
        huMoments = cv2.HuMoments(moments)
        #LogScale Hu Moments
        # for i in range(0,7):
        #         if huMoments[i] !=0 :
        #                 huMoments[i] = -1*math.copysign(1.0, huMoments[i])* math.log10(abs(huMoments[i]))
        #         else:
        #                 huMoments[i] = 0
        # huMoments = huMoments.flatten()
        return huMoments, moments, thresh_seg

#Size Detection
def detect_size(thresholded_seg):
        print("Size Detection...")
        # contours, hierarchy = cv2.findContours(thresholded_seg, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        nz_locs = np.argwhere(thresholded_seg)
        #Calculating Total Contour Area in a segment
        segmentContourArea = len(nz_locs)
        print(len(nz_locs))
        # no_of_contours = len(contours)
        # for i in range(no_of_contours):
        #         segmentConotourArea = segmentContourArea + cv2.contourArea(contours[i])
        return segmentContourArea

#Rotation Detection
def detect_rotation(moments):
        print("Rotation Detection")
        #central moments
        c_moments = [moments['mu20'], moments['mu11'], moments['mu02'], moments['mu30'],
                     moments['mu21'], moments['mu12'], moments['mu03']]
        average_c_moments = (moments['mu20'] + moments['mu11'] + moments['mu02']
                             + moments['mu21'] + moments['mu12'])/5
        return average_c_moments

#Placement Detection
def detect_placement(moments, garment_center, outer_removed_seg):
        print("Placement Detection...")
        # xg = garment_center[0]
        # yg = garment_center[1]

        # Finding the garment's center
        # nz_pos = np.argwhere(outer_removed_seg)
        # x = []
        # y = []
        # for a in nz_pos:
        #         x.append(a[1])
        #         y.append(a[0])
        # x = sorted(x)
        # y = sorted(y)
        #
        # xg = (x[0] + x[len(x)-1])/2
        # yg = (y[0] + y[len(y)-1])/2

        or_gr = cv2.cvtColor(outer_removed_seg, cv2.COLOR_BGR2GRAY)
        ret_or, thresh_or_seg = cv2.threshold(or_gr, 20, 255, cv2.THRESH_BINARY)

        # cv2.imshow("thresh_or", thresh_or_seg)
        # cv2.waitKey(0)

        #Find garment center 2
        # ctrs, h = cv2.findContours(thresh_or_seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        ctrs, h = cv2.findContours(thresh_or_seg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        # ctrs_len = []
        ctrs_len = []
        # for c in ctrs:
        #         ctrs_len.append(len(c))
        # print("Contours 1 length list :", ctrs_len)


        # sample = outer_removed_seg

        for k in ctrs:
                ctrs_len.append(len(k))
        print("Contours 2 length list:", ctrs_len)

        ctrlen_sorted = sorted(ctrs_len)
        max_len = ctrlen_sorted[len(ctrlen_sorted)-1]

        max_idx = 0
        for j in range(len(ctrs_len)):
                if ctrs_len[j] == max_len:
                        max_idx = j

        max_ctr = ctrs[max_idx]
        print("Cloth ctr length :", len(max_ctr))
        x = []
        y = []
        for pt in max_ctr:
                x.append(pt.flatten()[1])
                y.append(pt.flatten()[0])
        x = sorted(x)
        y = sorted(y)

        xg = (x[0] + x[len(x)-1])/2
        yg = (y[0] + y[len(y)-1])/2
        print("Garment Center :", xg," ",yg)

        # cv2.drawContours(sample, ctrs2, len(ctrs2) - 1,(0,255,0), 3)
        # sample = cv2.resize(sample, (750,1000 ))
        # cv2.imshow("Contours ", sample)
        # cv2.waitKey(0)

        #Segment Center of Mass
        xc = int(moments['m10']/moments['m00'])
        yc = int(moments['m01']/moments['m00'])
        segment_com = [xc,yc]

        #Distance between segment's center of mass and the center of the garment
        l = np.sqrt((xg-xc)**2+(yg-yc)**2)

        #Angle between the center of mass of the segment and the center of garment
        PI = 3.141592
        if (xc-xg) != 0 and (yc-yg) != 0:
                tan_value = abs((yc - yg) / (xc - xg))
                theta_rad = math.atan(tan_value)
                theta_deg = math.degrees(theta_rad)
                if(xc<xg) and (yc<yg):
                        theta = PI - theta_rad
                elif (xc>xg) and (yc<yg):
                        theta = theta_rad
                elif (xc<xg) and (yc>yg):
                        theta = PI + theta_rad
                elif (xc>xg) and (yc>yg):
                        theta = 2*PI - theta_rad
        else:
                if (xc==xg) and (yc<yg):
                        theta = PI/2
                elif (xc==xg) and (yc>yg):
                        theta = 3*(PI/2)
                elif (xc>xg) and (yc==yg):
                        theta = 0
                elif (xc<xg) and (yc==yg):
                        theta = 2*PI
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


def mark_defect(test_img, thresh_seg):
        img_def = test_img
        # img_def = cv2.resize(img_def,(900,1200))
        # thresh_seg = cv2.resize(thresh_seg,(900, 1200))
        thresh_pos = np.argwhere(thresh_seg)
        for pt in thresh_pos:
                img_def[pt[0]][pt[1]] = [0, 255, 0]
        # cv2.imshow("Color def image", img_color_def)

        return img_def




def display_arr(arr, type):
        def_count = []
        for a in arr:
                def_count.append(a[0])
        print(str(type) + " defect segments : ", def_count)


#Function Usage

# check_artwork_position(ref_artwork_loc, ref_or_cloth_loc, test_artwork_mask_loc, test_or_cloth_loc)
# match_segments(nonmatching_ref_loc, nonmatching_test_loc, matching_ref_loc, matching_test_loc)
# ref_features, ref_thresholded_segs, ref_dimensions, ref_segs = detect_features(no_of_matching_ref_segs, 1)
# shape_def, size_def, placement_def, rotation_def, color_def, minmax_def = detect_and_compare_matching_segments(no_of_matching_test_segs, ref_features, 1, ref_thresholded_segs, ref_dimensions, ref_segs)
#
# display_arr(shape_def, "Shape")
# display_arr(size_def, "Size")
# display_arr(placement_def,"Placement")
# display_arr(rotation_def,"Rotation")
# display_arr(color_def,"Color")
# display_arr(minmax_def,"Minima Maxima")
