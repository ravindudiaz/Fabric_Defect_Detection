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

#might need to be adjusted as per segment rois
ref_seg_roi_loc = "./Assets/QA_Module/Output/rois"
test_seg_roi_loc = "./Assets/QA_Module/Output/rois"

# matching_ref_loc
mr_file_list = os.listdir(matching_ref_loc)
no_of_matching_ref_segs = len(mr_file_list)

#matching_test_loc
mt_file_list = os.listdir(matching_test_loc)
no_of_matching_test_segs = len(mt_file_list)

#nonmatching_ref_loc
nmr_file_list = os.listdir(nonmatching_ref_loc)
no_of_nonmatching_ref_segs = len(nmr_file_list)

#non_matching_test_loc
nmt_file_list = os.listdir(nonmatching_test_loc)
no_of_nonmatching_test_segs = len(nmt_file_list)

def match_segments(nm_ref_loc, nm_test_loc, m_ref_loc, m_test_loc):
        print("Segment Matching Started...........................................................")
        print(nm_test_loc)

        for i in range(no_of_nonmatching_test_segs):

                #Get non matching segment
                print("Non Matching Test Segment "+str(i)+ " :")
                seg_nmt =  cv2.imread(nm_test_loc+"nm_"+str(i)+".jpg")
                seg_nmtg = cv2.cvtColor(seg_nmt, cv2.COLOR_BGR2GRAY)
                # cv2.imshow("nmt gray", seg_nmtg)
                # cv2.waitKey(0)
                ret1, nmt_thresh = cv2.threshold(seg_nmtg , 4, 255, cv2.THRESH_BINARY)

                x = []
                y = []

                nz = np.argwhere(nmt_thresh != 0)
                for pt in nz:
                        x.append(pt[0])
                        y.append(pt[1])

                template = seg_nmtg[min(x): max(x), min(y):max(y)]
                # print(min(y), max(y), min(x),max(x))
                # cv2.imshow("Template", template)
                # cv2.waitKey(0)

                #SIFT for nonmatching test seg
                sift = cv2.SIFT_create()
                kpt, dest = sift.detectAndCompute(template, None)

                max_inlier_points = 0
                best_match = 0

                #For Template Matching
                min_sqdiff_sum = 0
                best_tmatch = 0
                min_sqdiff = 0
                sqdiff = []

                #Get Matching  Ref segment
                for j in range(no_of_matching_ref_segs):
                        print("Matching Ref Segment "+str(j)+" :")
                        seg_ref =  cv2.imread(m_ref_loc+"m_"+str(j)+".jpg")
                        seg_refg = cv2.cvtColor(seg_ref, cv2.COLOR_BGR2GRAY)
                        # cv2.imshow("ref gray", seg_refg)
                        # cv2.waitKey(0)
                        ret2, ref_thresh = cv2.threshold(seg_refg, 4, 255, cv2.THRESH_BINARY)

                #         # SIFT and FLANN Based
                #         #SIFT for ref seg
                #         kpr, desr = sift.detectAndCompute(seg_refg, None)
                #         print(len(kpr), len(kpt))
                #         # for d in dest:
                #         #         print(dest)
                #
                #
                #
                #         if(len(kpr) >= len(kpt)):
                #                 # bf = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
                #                 flann = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
                #                 matches = flann.match(dest, desr)
                #                 print(len(matches))
                #
                #                 matches = sorted(matches, key=lambda x: x.distance)
                #                 # matches = matches[:int(len(matches)*4/5)]
                #                 print("Threshold clue : ",matches[len(matches)-1].distance)
                #                 # matches = list(filter(lambda x: x.distance <= matches[len(matches)-1].distance*4/5, matches))
                #                 matches = list(filter(lambda x: x.distance <= 250, matches))
                #                 print(type(matches))
                #                 total_dist = 0
                #                 for m in matches:
                #                         # print(m.distance)
                #                         total_dist += m.distance
                #
                #                 print("Match Distance : ",total_dist)
                #                 points1 = np.zeros((len(matches), 2), dtype=np.float32)
                #                 points2 = np.zeros((len(matches), 2), dtype=np.float32)
                #
                #                 for l,match in enumerate(matches):
                #                         points1[l, :] = kpt[match.queryIdx].pt
                #                         points2[l, :] = kpr[match.queryIdx].pt
                #
                #                 h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
                #                 # print(mask)
                #                 # seg_reg = cv2.warpPerspective()
                #
                #                 matching_result = cv2.drawMatches(template, kpt, seg_ref, kpr, matches[:20], None)
                #
                #                 cv2.imshow("Matching result", matching_result)
                #                 cv2.waitKey(0)
                #
                #                 inl = 0
                #                 for x in mask:
                #                         if x == [1]:
                #                                 inl += 1
                #                 print("Inliers" , inl)
                #                 if inl >= max_inlier_points:
                #                         max_inlier_points = inl
                #                         best_match = j
                # print("Max Inlier Points :" + str(max_inlier_points))
                # print("Best Match :" + str(best_match)+"  **********************************")

                        #Template
                        res = cv2.matchTemplate(seg_refg, template, cv2.TM_SQDIFF)
                        # print(res)
                        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                        th, tw = template.shape[::]
                        top_left = min_loc
                        bottom_right = (top_left[0]+tw, top_left[1]+th)
                        cv2.rectangle(seg_refg, top_left, bottom_right, 255, 2)
                        # plt.imshow(res, cmap='gray')

                        res_np = np.array(res)
                        res_pos = np.argwhere(seg_refg)
                        # print(res_pos)
                        # sqdiff_sum = sum(sum(res))
                        min_res = np.min(res)
                        print("Minimum distance value in res : ", min_res)
                        # print("Segment sqdiff sum : " ,sqdiff_sum)

                        # if min_sqdiff > min_res:
                        #         # min_sqdiff_sum = min_res
                        #         min_sqdiff = min_res
                        #         best_tmatch = j

                        sqdiff.append(min_res)

                        # res_selected = res_np[res_np != (1.1370468e+09 or 1.1370469e+09)]
                        # print(res_selected)
                        # res_selected = list(res_selected)
                        # sumList = sum(res_selected)



                        # print("Matched Sum :", sumList,"********************************")
                        # sum_prev = sum(sum(res))
                        # print("sum : ", sum_prev)
                        # cv2.imshow("ref gray template matched", seg_refg)
                        # cv2.waitKey(0)

                        sh = seg_ref.shape[0]
                        sw = seg_ref.shape[1]
                        # print(str(sh*sw))
                        # print(len(res[0]) * len(res))
                sorted_sqdiff = sorted(sqdiff)
                min_sqdiff = sorted_sqdiff[0]
                for k in range(len(sqdiff)):
                        if sqdiff[k] == min_sqdiff:
                                best_tmatch = k
                print(min_sqdiff)
                print("Best Match : ", best_tmatch, " ***********************************************")




def detect_features(no_of_matching_ref_segs, ref_img_check):
        print("Reference Image Feature Extraction Started.........................................")

        ref_features=[]

        for i in range(no_of_matching_ref_segs):
                print("Reference Image Segment ", str(i))
                seg_features = []
                if ref_img_check == 1:
                        seg = cv2.imread(matching_ref_loc+"m"+"_"+str(i)+".jpg")
                else:
                        seg = cv2.imread(matching_test_loc + "m" + "_" + str(i) + ".jpg")

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
                xg = int(dimensions[1] / 2)
                yg = int(dimensions[0] / 2)
                garment_center = [xg, yg]
                segment_placement_measures = detect_placement(moments, garment_center)
                print(segment_placement_measures)

                #Detect Color
                segment_color_measures = detect_color(seg, thresh_seg)

                #Detect Minima Maxima
                # seg_zero_crossings, seg_point_measures = detect_minima_maxima(thresh_seg, seg)

                detMinMax(thresh_seg, seg)

                # seg_features = [huMoments[6],huMoments[0],huMoments[1],huMoments[2],segmentContourArea, ref_avg_comment_measure, segment_placement_measures, segment_color_measures, seg_zero_crossings, seg_point_measures]
                seg_features = [huMoments[6], huMoments[0], huMoments[1], huMoments[2], segmentContourArea, ref_avg_comment_measure, segment_placement_measures, segment_color_measures]
                # cv2.imshow("Init Contours", thresh_seg)
                # cv2.waitKey(0)

                ref_features.append(seg_features)
        # print(ref_features)
        return ref_features

def detect_and_compare_matching_segments(no_of_segments,ref_features,test_img_check):
        print("Detect and Match test image stage reached...........................")

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
                                shape_defect_json = json.dumps(shape_defect)
                        else:
                                #Detect and Compare Size
                                segmentContourArea = detect_size(thresh_seg)
                                size_deviation = abs(ref_features[i][4] - segmentContourArea)/ ref_features[i][4]
                                print(size_deviation)
                                # Ta = 0.15       #Size Deviation Threshold- Experimental
                                Ta = 100000
                                if size_deviation >= Ta:
                                        size_defect = {
                                                "type": "Size Defect",
                                                "status" : "Size Deviation = " + str(size_deviation)+ ". Exceeded threshold. "
                                        }
                                        size_defect_json = json.dumps(size_defect)
                                else:
                                        #Detect and Compare Rotation
                                        rotation_measure = detect_rotation(moments)
                                        rotation_deviation = abs((ref_features[i][5] - rotation_measure)/ref_features[i][5])
                                        print("Rotation Deviation : ", rotation_deviation)
                                        # Tr = 0.056
                                        Tr = 100000

                                        if rotation_deviation >= Tr:
                                                rotation_defect = {
                                                        "type": "Rotation Defect",
                                                        "status": "Size Deviation " + str(rotation_deviation) + ". Exceeded threshold. "
                                                }
                                                rotation_defect_json = json.dumps(rotation_defect)
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
                                                else:
                                                        #Detect and Compare Color
                                                        ret1, gr_test_seg_thresh = cv2.threshold(gr_test_seg, 40, 255, cv2.THRESH_BINARY)
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
                                                                        # channel_dev = rm[l][0][0] + tm[l][0][0]
                                                                        roi_deviations.append(channel_dev)
                                                                # print(roi_deviations)
                                                                all_roi_deviations.append(roi_deviations)
                                                        # print(len(all_roi_deviations))
                                                        print(all_roi_deviations)

                                                        # Tcol = 0.015
                                                        Tcol = 10000000
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
                                                                shape_defect_json = json.dumps(color_defect)
                                                                print(color_defect)
                                                        else:
                                                                #Detect and Compare Minima and Maxima
                                                                test_zero_crossings, test_point_measures = detect_minima_maxima(gr_test_seg_thresh, test_seg)









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


#Minima Maxima Detection 2
def detMinMax(thresholded_segment, segment):
        print("Minima Maxima Detection")
        contours_all, hierarchy = cv2.findContours(thresholded_segment, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cntAreas = []
        contours = []
        # print(contours_all)
        print("Total contours discovered :", len(contours_all))
        for cnt in contours_all:
                cntAreas.append(cv2.contourArea(cnt))
        cntAreas = sorted(cntAreas)
        cntAreaThreshold = max(cntAreas) * 0.0008
        # print(cntAreas)
        for cnt in contours_all:
                # print(len(cnt))
                if (cv2.contourArea(cnt) >= cntAreaThreshold) and len(cnt) >= 20:
                        contours.append(cnt)
        print("Selected Contours : ", len(contours))
        dimensions = segment.shape
        img_cont = np.zeros((dimensions[0], dimensions[1], 1), np.uint8) * 255
        cv2.drawContours(img_cont, contours, -1, 255, 1)

        cv2.imshow("Init Contours", img_cont)
        cv2.waitKey(0)

        #Parameters to be passed to x zero crossings
        diffSeqx = []
        diffY = []
        xAxis = []

        approxCont = []

        for k in range(len(contours)):
                epsilon = 0.001 * cv2.arcLength(contours[k], True)
                appcnt = cv2.approxPolyDP(contours[k], epsilon, True)
                approxCont.append(appcnt)
                contours[k] = appcnt

        diffSeqx, diffY, xAxis = detect_x_zerocrossings(contours, diffSeqx, diffY, xAxis)

        #Parameters to be passed to y zero crossings
        diffSeqy = []
        diffX = []
        yAxis = []

        diffSeqy, diffX, yAxis = detect_y_zerocrossings(contours, diffSeqy, diffX, yAxis)

def detect_x_zerocrossings(contours, diffSeq, diffY, xAxis):
        # Get Zero Crossings over X axis
        for n in range(len(contours)):
                pst = 0
                ngt = 0
                diffY_ct = []
                diffSeq_ct = []
                for i in range(len(contours[n]) - 1):
                        d = contours[n][i + 1].flatten()[1] - contours[n][i].flatten()[1]
                        diffY_ct.append(d)
                        if (d >= 0):
                                pst += 1
                                if ngt != 0:
                                        diffSeq_ct.append(ngt)
                                        ngt = 0
                        else:
                                ngt += 1
                                if pst != 0:
                                        diffSeq_ct.append(pst)

                if ngt != 0:
                        diffSeq_ct.append(ngt)
                        ngt = 0
                elif pst != 0:
                        diffSeq_ct.append(pst)
                        pst = 0
                diffSeq.append(diffSeq_ct)
                # print("diffSeq ct :", diffSeq_ct)
                diffY.append(diffY_ct)
                # print("diffY_ct :", diffY_ct)
        print("diffSeq", diffSeq, "******************************************")
        print("diffY", diffY, "******************************************")

        for n in range(len(contours)):
                xAxisct = []
                # if n == 0:
                cntr = 0
                temp = []
                for i in range(len(diffSeq[n]) - 1):
                        if diffSeq[n][i] >= 1 and diffSeq[n][i + 1] >= 1:
                                temp.append(cntr + diffSeq[n][i])
                        cntr += diffSeq[n][i]
                xAxis.append(temp)
        # xAxis.append(xAxisct)

        print("xAxis : ",xAxis)
        for m in range(len(xAxis)):
                print("Contour ", m," xz points: ", len(xAxis[m]))
        return diffSeq, diffY, xAxis

def detect_y_zerocrossings(contours, diffSeq, diffX, yAxis):
        # Get Zero Crossings over X axis
        for n in range(len(contours)):
                pst = 0
                ngt = 0
                diffX_ct = []
                diffSeq_ct = []
                for i in range(len(contours[n]) - 1):
                        d = contours[n][i + 1].flatten()[0] - contours[n][i].flatten()[0]
                        diffX_ct.append(d)
                        if (d >= 0):
                                pst += 1
                                if ngt != 0:
                                        diffSeq_ct.append(ngt)
                                        ngt = 0
                        else:
                                ngt += 1
                                if pst != 0:
                                        diffSeq_ct.append(pst)

                if ngt != 0:
                        diffSeq_ct.append(ngt)
                        ngt = 0
                elif pst != 0:
                        diffSeq_ct.append(pst)
                        pst = 0
                diffSeq.append(diffSeq_ct)
                diffX.append(diffX_ct)
        print("diffSeq", diffSeq, "******************************************")
        print("diffX", diffX, "******************************************")

        for n in range(len(contours)):
                # if n == 0:
                        cntr = 0
                        temp = []
                        for i in range(len(diffSeq[n]) - 1):
                                if diffSeq[n][i] >= 1 and diffSeq[n][i + 1] >= 1:
                                        temp.append(cntr + diffSeq[n][i])
                                cntr += diffSeq[n][i]
                        yAxis.append(temp)
        print("yAxis ", yAxis)

        # print("No of y points : ", len(yAxis))
        for m in range(len(yAxis)):
                print("Contour ", m," yz points: ", len(yAxis[m]))
        return diffSeq, diffX, yAxis

def sortXYarrays(xAxis, yAxis, contours):
        position_List = []

        for n in range(len(contours)):
                xyList_ct = xAxis[n] + yAxis[n]



#Minima Maxima Detection
def detect_minima_maxima(thresholded_segment,segment):
        print("Minima Maxima Detection")
        contours_all, hierarchy = cv2.findContours(thresholded_segment, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cntAreas = []
        contours = []
        # print(contours_all)
        print("Total contours discovered :",len(contours_all))
        for cnt in contours_all:
                cntAreas.append(cv2.contourArea(cnt))
        cntAreas = sorted(cntAreas)
        cntAreaThreshold = max(cntAreas)*0.0008
        # print(cntAreas)
        for cnt in contours_all:
                # print(len(cnt))
                if (cv2.contourArea(cnt) >= cntAreaThreshold) and len(cnt) >= 20:
                        contours.append(cnt)
        print("Selected Contours : ", len(contours))
        dimensions = segment.shape
        img_cont = np.zeros((dimensions[0],dimensions[1],1),np.uint8)*255
        cv2.drawContours(img_cont, contours, -1, 255, 1)

        cv2.imshow("Init Contours", img_cont)
        cv2.waitKey(0)

        #Appoximating the contour to the closest polygon
        approx = []
        smoothed_curves = []
        all_seg_point_measures = []
        all_seg_zero_crossing_pts = []

        for i in range(len(contours)):
                        epsilon = 0.001*cv2.arcLength(contours[i], True)
                        approx.append(cv2.approxPolyDP(contours[i], epsilon, True))
                        # print("Reached", len(contours[i]))
                        x = []
                        y = []

                        #Gaussian Kernel
                        gKernel = cv2.getGaussianKernel(5, 8)
                        G = cv2.transpose(gKernel)
                        s_kernel = np.array([1, -1])
                        ss_kernel = np.array([1, -2, 1])

                        G_s = np.convolve(G.flatten(), s_kernel.flatten(), "same")
                        G_ss = np.convolve(G.flatten(), ss_kernel.flatten(), "same")

                        for j in range(len(approx[i])):
                                        # print("Cont"+str(j))
                                        x.append(approx[i][j].flatten()[0])
                                        y.append(approx[i][j].flatten()[1])

                        smoothed_curve_x = np.convolve(x, G.flatten(), "same")
                        smoothed_curve_y = np.convolve(y, G.flatten(), "same")
                        # print("x array",smoothed_curve_x)
                        # print("y array",smoothed_curve_y)


                        convol_len = len(G.flatten())
                        x_len = len(smoothed_curve_x)
                        # print("smoothed curve x length : ",x_len)
                        y_len = len(smoothed_curve_y)
                        for a in range(int(convol_len/2)):
                                smoothed_curve_x[a] = approx[i][a].flatten()[0]
                                smoothed_curve_y[a] = approx[i][a].flatten()[1]
                                smoothed_curve_x[x_len - a - 1] = approx[i][x_len - a - 1].flatten()[0]
                                smoothed_curve_y[x_len - a - 1] = approx[i][x_len - a - 1].flatten()[1]

                        for u in range(len(smoothed_curve_x)):
                                smoothed_curve_x[u] = int(smoothed_curve_x[u])
                                smoothed_curve_y[u] = int(smoothed_curve_y[u])

                        #Partial Derivative Values
                        smoothed_curve_xs = np.convolve(smoothed_curve_x, s_kernel.flatten(), "same")
                        smoothed_curve_ys = np.convolve(smoothed_curve_y, s_kernel.flatten(), "same")
                        smoothed_curve_xss = np.convolve(smoothed_curve_x, ss_kernel.flatten(), "same")
                        smoothed_curve_yss = np.convolve(smoothed_curve_y, ss_kernel.flatten(), "same")

                        smoothed_curve = []
                        for j in range(len(smoothed_curve_x)):
                                smoothed_curve.append([[smoothed_curve_x[j], smoothed_curve_y[j]]])

                        smoothed_curve = np.array(smoothed_curve)
                        # print(smoothed_curve)
                        smoothed_curves.append(smoothed_curve)


                        #Applying Curvature Scale Space Derived Method
                        zero_crossing_pts = []
                        kappa = ((smoothed_curve_xs*smoothed_curve_yss)-(smoothed_curve_xss*smoothed_curve_ys))/(smoothed_curve_xs**2 + smoothed_curve_ys**2)**1.5

                        seg_point_measures = []
                        for r in range(len(kappa)):
                                # if kappa[r] <= 0.0005 and kappa[r] >= -0.0005:
                                if kappa[r] == 0:
                                        z_point = [int(smoothed_curve[r].flatten()[0]), int(smoothed_curve[r].flatten()[1])]
                                        zero_crossing_pts.append(z_point)
                                        if (smoothed_curve_xs[r] + smoothed_curve_ys[r])**3 != 0 :
                                                seg_point_measures.append((smoothed_curve_xs - smoothed_curve_ys)**2/(smoothed_curve_xs[r] + smoothed_curve_ys[r])**3)

                                        else:
                                                seg_point_measures.append((smoothed_curve_xs - smoothed_curve_ys)**2)
                                        cv2.line(segment, (z_point[0],z_point[1]), (z_point[0],z_point[1]), (255,0,0), 5)


                        all_seg_zero_crossing_pts.append(zero_crossing_pts)
                        all_seg_point_measures.append(seg_point_measures)
        # print(len(smoothed_curves))






        #Visualizing smoothed approximated contours

        smoothed_cont = np.zeros((dimensions[0], dimensions[1], 1), np.uint8) * 255
        # cv2.drawContours(smoothed_cont, smoothed_curves, -1, 255, 1)

        # Visualizing approximated contours
        approx_cont_visual = np.zeros((dimensions[0], dimensions[1], 1), np.uint8) * 255
        cv2.drawContours(approx_cont_visual, approx, -1, 255, 1)

        # Display Original Contours
        cv2.imshow("Original segment contours", img_cont)
        cv2.waitKey(0)

        #Display Approximated Contours
        cv2.imshow("Approximated Contours", approx_cont_visual)
        cv2.waitKey(0)

        # #Display Smoothed Contours
        # cv2.imshow("Smoothed Contours", smoothed_cont)
        # cv2.waitKey(0)

        #Display Points
        cv2.imshow("Detected Points", segment)
        cv2.waitKey(0)

        return all_seg_zero_crossing_pts, all_seg_point_measures




#Function Usage
# match_segments(nonmatching_ref_loc, nonmatching_test_loc, matching_ref_loc, matching_test_loc)
ref_features = detect_features(no_of_matching_ref_segs, 1)
detect_and_compare_matching_segments(no_of_matching_test_segs, ref_features, 1)



