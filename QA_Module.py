
import numpy as np
import cv2
import math
import statistics as stat
from matplotlib import pyplot as plt
import os

        #printed artwork Location
ref_artwork_loc = "./image1/ref_artwork_mask/"
test_artwork_loc = "./image1/test_artwork_mask/"
ref_seg_loc = "./image1/ref_img_segs/"
test_seg_loc = "./image1/test_img_segs/"
roi_loc = "./image1/color_rois/"



matching_ref_loc ="./Assets/Seg_Module/Output/Matching/ref/"
matching_test_loc = "./Assets/Seg_Module/Output/Matchihng/test/"
nonmatching_ref_loc = "./Assets/Seg_Module/Output/Non_Matchihng/ref/"
nommatching_test_loc="./Assets/Seg_Module/Output/Non_Matchihng/test/"

# matching_ref_loc
file_list = os.listdir(matching_ref_loc)
no_of_segments = len(file_list)


#Test Image background mask and reference image outer background mask is needed for this
#The relevant pictures of the Reference Image or Test Image are needed when using this funtion.
#These images will be used to extract placement measures in Placement detection
#Following are possibly to be implemented as a function
#We might possibly have to give the file location as a parameter for this function
#(Possibly as a variable in cv2.imread() at the beginning which reads the segments
#Reference Image and Test Image segments will be stored in a different folders

#Begin function here
def detect_features(no_of_segmnts, ref_img_check):
        features = []
        for i in range(no_of_segments):
                seg_features = []
                if ref_img_check == 1 :
                        seg = cv2.imread(matching_ref_loc+"m"+"_"+str(i)+".jpg")
                else :
                        seg = cv2.imread(matching_test_loc + "m" + "_" + str(i) + ".jpg")

                # seg = cv2.resize(seg, (800,800))
                # cv2.imshow("segment_"+str(i), seg)
                gr_seg = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)

                #Shape Detection
                _, thresh_seg = cv2.threshold(gr_seg, 80, 255, cv2.THRESH_BINARY)
                # cv2.imshow("Thresh seg"+str(i),thresh_seg)
                # cv2.waitKey(0)

                        #Moments
                moments = cv2.moments(thresh_seg)
                        #HuMoments
                huMoments = cv2.HuMoments(moments)
                        #LogScale Moments
                for i in range(0,7):
                        if huMoments[i] != 0 :
                                huMoments[i] = -1*math.copysign(1.0, huMoments[i])* math.log10(abs(huMoments[i]))
                        else:
                                huMoments[i] = 0

                huMoments = huMoments.flatten()

                # Size Detection
                contours, hierarchy = cv2.findContours(thresh_seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                # cv2.drawContours(seg, contours, -1, (0,255,0), 1)
                print(len(contours))





                        #Calculating Contour area
                contourArea = 0
                for i in range(len(contours)):
                        contourArea = contourArea + cv2.contourArea(contours[i])
                        contour_com = [int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00'])]

                # print("Contour center of mass :",contour_com)

                #Rotation Detection
                c_moments = [moments['mu20'], moments['mu11'], moments['mu02']]

                        #11,12,13,14,15,16,17 Central  values in moments

                #Placement Detection - should be done using the image prepared by using the masks
                dimensions = seg.shape
                        # segment center
                segment_center = [int(dimensions[1]/2), int(dimensions[0]/2)]
                        #Contour center of mass is calculated in the loop in size detection section
                        #eucledian distance to center of mass from the segment center
                euc_distance = int(math.sqrt(((contour_com[0]-segment_center[0])**2+(contour_com[1]-segment_center[0])**2)))

                        #Angle between the center of mass and the center of segment
                if (contour_com[0]-segment_center[0]) !=0 :
                        tan_value = (contour_com[1]-segment_center[1])/(contour_com[0]-segment_center[0])
                        theta = math.atan(tan_value)
                else :
                        tan_value = np.Infinity


                #Minima Maxima Detection

                blurred_seg = cv2.GaussianBlur(seg, (3,3), 0)
                # cv2.imshow("XO Gaussian", blurred_seg)
                # cv2.waitKey(0)
                blurred_seg_gr = cv2.cvtColor(blurred_seg, cv2.COLOR_BGR2GRAY)
                # cv2.imshow("blurred seg gray",blurred_seg_gr)
                # cv2.waitKey(0)
                laplacian_seg = cv2.Laplacian(blurred_seg_gr, cv2.CV_16S, 4)
                # laplacian_seg = cv2.Laplacian(gr_seg, cv2.CV_16S, 3)
                laplacian_seg = cv2.convertScaleAbs(laplacian_seg)
                ret2, thresh_lap_seg = cv2.threshold(laplacian_seg, 30, 255, cv2.THRESH_BINARY)
                # cv2.imshow("XO - Laplacian of Gaussian", laplacian_seg)
                # cv2.waitKey(0)
                # cv2.imshow("Thresh log seg", thresh_lap_seg)
                # cv2.waitKey(0)
                #A method should be verified to decide this threshold value. Consider the difference between the colors
                #in the segment boundaries. Consider Doing this using the mask given by Ishara. The contour covering the
                #boundary of the entire print will be used. The color distances between the segment boundaries might have
                #to be considered as well. Use mean color or mean intensity as well

                ret, thresh_blurred_seg = cv2.threshold(blurred_seg_gr, 80, 255, cv2.THRESH_BINARY)
                # cv2.imshow("XO - Thresholded ", thresh_blurred_seg)
                # cv2.waitKey(0)

                blur_contours, hierarchy2 = cv2.findContours(thresh_blurred_seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                approxContours = []
                # for cont in blur_contours:
                for cont in blur_contours:
                        epsilon = 0.0001*cv2.arcLength(cont, True)
                        approx = cv2.approxPolyDP(cont, epsilon, True)
                        no_of_points = len(approx)
                        print(f'number of points in the contour : {no_of_points}')

                        if no_of_points > 200:
                                curve_sections = 10
                        elif no_of_points > 150:
                                curve_sections = 8
                        elif no_of_points > 100:
                                curve_sections = 6
                        elif no_of_points > 75:
                                curve_sections = 5
                        elif no_of_points > 50:
                                curve_sections = 3
                        elif no_of_points > 25:
                                curve_sections = 2
                        else :
                                curve_sections = 1
                        # print(approx[0][0])

                        splitted_curve = np.array_split(approx, curve_sections)
                        # print(splitted_curve)

                        for k in range(len(splitted_curve)):
                                x = []
                                y = []
                                for j in range(len(splitted_curve[k])):
                                        # print(f'curve section {k} point {j} : {splitted_curve[k][j]}')
                                        appending_coord = tuple(splitted_curve[k][j][0])
                                        # print(f'appending coord: {appending_coord}')
                                        x.append(appending_coord[0])
                                        y.append(appending_coord[1])
                                plt.figure(k)
                                plt.plot(x, y)
                                # plt.show()


                        s_coord = tuple(approx[0][0].flatten())
                        # cv2.line(seg, (s_coord[0], s_coord[1]), (s_coord[0],s_coord[1]), (0, 0, 255), 5)
                        approxContours.append(approx)

                cv2.drawContours(seg, approxContours,-1, (0, 255, 0), 1)

                cv2.imshow("Segment" + str(i), seg)
                cv2.waitKey(0)

                seg_features = [huMoments[0], huMoments[1], huMoments[2], huMoments[6], contourArea, c_moments[0], c_moments[1], c_moments[2], theta ]


                # print(seg_features)
                features.append(seg_features)
                # cv2.imshow("segment_"+str(i), seg)
                # cv2.imshow("segment grThresh_"+str(i), thresh_seg)
                # cv2.waitKey(0)


        #Color Detection

                #Extract the printwork from the image or take the Output from Ishara
                #Should possible give this as a parameter for the function
                #Take the printwork from Piyumika with black for the background

        if ref_img_check == 1:
                artwork = cv2.imread(ref_artwork_loc + "ref_artwork.jpg")
        else :
                artwork = cv2.imread(test_artwork_loc + "test_artwork.jpg")

        artwork_hsv = cv2.cvtColor(artwork, cv2.COLOR_BGR2HSV)



        #Getting the contours corresponding to each segment
        # artwork_contours = cv2.findContours()
        # print(artwork_hsv.shape)
                #Breaking down the image into Regions of Interest (4 regions)

        artwork_dimensions = artwork.shape
        # print(artwork_hsv)
        split_index = int(artwork_dimensions[0]/2)
        slice_1 = np.split(artwork_hsv, [split_index], 0)
        print(len(slice_1))
        slice_2 = []
        # print("Slice 1 :", slice_1)
        for i in range(len(slice_1)):
                split_index_2 = int(slice_1[i].shape[1]/2)
                slice_2.append(np.split(slice_1[i], [split_index_2] , 1))

        roi = [slice_2[0][0], slice_2[0][1], slice_2[1][0], slice_2[1][1]]

                #Calculating Color measures
        color_measures = []

        for j in range(len(roi)):
                channels = cv2.split(roi[j])
                channel_measures = []
                # print(channels)
                for i in range(len(channels)):
                        color_mean = 0
                        sorted_channel = np.sort(channels[i])
                        color_median = np.median(sorted_channel)
                        color_std_dev = np.std(channels[i])                 #Standard Deviation
                        color_skewness = 3*(color_mean - color_median)/color_std_dev
                        color_mean = np.mean(channels[i])
                        color_variance = np.var(channels[i])
                        channel_measures.append([color_mean, color_variance, color_skewness])

                # cv2.imshow("roi", roi[j])
                # cv2.waitKey(0)
                color_measures.append(channel_measures)
                features.append(channel_measures)

        print("Color Measures : ",color_measures)
        # channel_means = cv2.mean(artwork_hsv)
        # print("Color channel_means :",channel_means)

        for l in range(4):
                cv2.imshow("roi"+str(l), roi[l])
                cv2.waitKey(0)
                if ref_img_check == 1:
                        cv2.imwrite(f'{roi_loc}roi_ref_{l}.jpg', roi[l])
                else :
                        cv2.imwrite(f'{roi_loc}roi_test_{l}.jpg', roi[l])

        # cv2.waitKey(0)
        cv2.destroyAllWindows()
        return features

#End function here

#Feature List Description


#Fundtion usage
ref_features = detect_features(no_of_segments, 1)
print("Reference Image features : ", ref_features)
print("Feature extraction finished")

#Function usage
test_features = detect_features(no_of_segments, 0)
print("Test Image features : ", test_features)
print("Feature Extraction Finished")

#Comparison

#Shape Comparison, Scale Comparison, Rotation Comparison
def compare_ssr(no_of_segments, ref_features, test_features ) :
        ssr_distance = []   #Shape Scale Rotation distance

        for i in range(no_of_segments):
                seg_shape_distance = []
                if ref_features[i][3] != -1*test_features[i][3]:  #Checking Mirroring using 7th Hu Moment
                        for j in range(3):
                                seg_hu_distance = ref_features[i][j] - test_features[i][j]
                                seg_shape_distance.append(seg_hu_distance)
                        ssr_distance.append(seg_shape_distance)

                        # Segment Scaling Check
                        k_downscale = 100 / 98
                        k_upscale = 100 / 102
                        if ref_features[i][4] > int(k_downscale * test_features[i][4]):
                                ssr_distance[i].append([ref_features[i][4] - test_features[i][4], "Down Scaled"])
                        elif ref_features[i][4] < k_upscale * test_features[i][4]:
                                ssr_distance[i].append([ref_features[i][4] - test_features[i][4], "Up Scaled"])
                        else :
                                ssr_distance[i].append([ref_features[i][4] - test_features[i][4], "Not Considerably Scaled - Accepted"])

                        # Rotatation Check using Central Moments
                        rotation_distance = []
                        for k in range(5, 8):
                                rotation_distance.append(ref_features[i][k] - test_features[i][k])
                        rotation_distance_average = (rotation_distance[0] + rotation_distance[1] + rotation_distance[2])/len(rotation_distance)
                        rotation_distance.append(rotation_distance_average)

                        if(rotation_distance_average != 0):
                                rot_status = "Rotated"
                        else :
                                rot_status = "Not Rotated"

                        ssr_comparison_status = "Done"

                        ssr_distance[i].append([rotation_distance_average, rot_status])


                else :
                        print("Mirrored Segment Detected : Rejected")
                        break
        return ssr_distance, ssr_comparison_status
#End of function

#Color Comparison by Regions of Interest (roi)

def compare_color(no_of_segments, ref_features, test_features) :
        color_distance = []
        for i in range(no_of_segments, no_of_segments+4) :             #Going through ROI s
                roi_color_distance = []
                for j in range(len(ref_features[i])):
                        measure_distance = []
                        for k in range(3):
                                distance = ref_features[i][j][k]-test_features[i][j][k]
                                measure_distance.append(distance)
                        roi_color_distance.append(measure_distance)
                color_distance.append(roi_color_distance)
        # print(color_distance)
        return color_distance

# print(ref_features[no_of_segments])
# print(ref_features[no_of_segments+1])
# print(ref_features[no_of_segments+2])
# print(ref_features[no_of_segments+3])

#Function usage
ssr_distance, ssr_comparison_status = compare_ssr(no_of_segments, ref_features, test_features)

color_distance = compare_color(no_of_segments, ref_features, test_features)

print("Shape Scale Rotation Distance: ", ssr_distance)
print("ROI Color Distance : ", color_distance)

# print(ssr_distance " ",ssr_comparison_status)