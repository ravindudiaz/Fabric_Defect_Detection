import os
from os import walk
import cv2
import csv
import matplotlib.pyplot as plt

# base_path = "H:/FYP/Piu/work"
# img_name = "patagonia_defect"
# file_name = '3'
# sub_folder = 'reference'

class FeatureExtract():
    def __init__(self,dir):

        self.dir = dir
        #self.img_name = img_name


        self.ref_path = os.path.join(dir, 'segments')
        self.ref_csv_path = os.path.join(dir, 'features.csv')
        self.ref_colors_csv_path = os.path.join(dir, 'color.csv')

    def write_to_csv(self,fieldnames,data):
        # print(self.ref_csv_path)
        try:
            with open(self.ref_csv_path, mode='w', newline='') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                for segment_data in data:
                    writer.writerow(segment_data)
            return (True)
        except:
            return (False)
    def read_from_csv(self):
        with open(self.ref_colors_csv_path, mode='r', newline='') as csv_file:
            reader = csv.DictReader(csv_file)
            dic_colors = []
            for r in reader:
                dic_colors.append(r)

            return dic_colors

    def get_image_names(self):
        image_names = []
        for (dirpath, dirnames, filenames) in walk(self.ref_path):
            image_names .extend(filenames)
            break
        return image_names

    def extract_features(self,image_list, color_list):
        data = []
        # print(image_list)
        for image in image_list:

            imagename, image_extension = os.path.splitext(image )
            img = cv2.imread(os.path.join(self.ref_path,image))
            imgray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
            # #ret, thresh = cv2.threshold(imgray, 10, 255, 0)

            # Otsu's thresholding after Gaussian filtering
            blur = cv2.GaussianBlur(imgray , (5, 5), 0)
            ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            contours, hierarchy = cv2.findContours(th3, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
            area = cv2.contourArea(contours[0])
            # print('area:')
            # print(area)

            M = cv2.moments(contours[0])
            try:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            except:
                # cv2.imshow('gray', blur)
                # cv2.imshow('thresh', th3)
                cv2.waitKey()


            # print('centers:')
            # print(cX)
            # print(cY)
            # image = cv2.circle(img, (cX, cY), 30, (255, 0, 0), 2)
            #cv2.imshow('center', image )
            # imgplot = plt.imshow(image)
            # plt.show()
            # cv2.waitKey()

            # print(img[cY,cX])
            center = {"X": cX, "Y": cY}

            has_child = False
            for i,cnt in enumerate(contours):
                if hierarchy[0][i][2] != -1:
                    has_child = True
                    break

            # print(has_child)
            segmentData = {'id': imagename, 'color': '', 'area': area, 'center': center, 'has_child': str(has_child)}
            for item in color_list:
                if(item['id'] == segmentData['id']):
                    segmentData['color'] = item['color']
            data.append(segmentData)
        return data

# ref_segments = get_image_names(ref_path)
#
# color_features = read_from_csv(ref_colors_csv_path)
#
# full_features = extract_features(ref_segments,color_features)
#
# print(color_features)
# print(full_features)
#
# write_to_csv(ref_csv_path,['id','color', 'area', 'center','has_child'],full_features)
