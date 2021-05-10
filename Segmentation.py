import numpy as np
import cv2
import os
import csv
from feature_extract import FeatureExtract


### create folder structire
# base_path = "H:/FYP/Piu/images"
# work_path = "H:/FYP/Piu/work"
# img_name = "uni_nike_dark_def3_nm2.jpg"
# sub_folder_name = 'defect_3'
# filename = 'uni_nike_dark'


img_path = ""
#new_dir_sub = ""
new_dir = ""
main_color_path = ""
segment_path = ""


# files = os.listdir(work_path)
# if(len(files) == 0):
#     filename = '1'
# else:
#     files.sort(reverse=True)
#     filename = int(files[0]) + 1
#     filename = str(filename)



#filename, file_extension = os.path.splitext(img_name )


def setFolderNames(image_path, work_path, sub_folder_name):
    global new_dir,main_color_path,segment_path,img_path

    img_path = image_path
    #new_dir_sub = os.path.join(work_path, filename)
    new_dir = os.path.join(work_path, sub_folder_name )
    main_color_path = os.path.join(new_dir, "colors")
    segment_path = os.path.join(new_dir, "segments")
    #out_path = os.path.join(base_path, "segmented")

def createFolders(work_path):
    global new_dir,main_color_path,segment_path

    # try:
    #     os.mkdir(new_dir_sub)
    # except OSError:
    #     print("Creation of the directory failed")
    # else:
    #     print("Successfully created the directory")
    try:
        os.mkdir(new_dir)
    except OSError:
        print("Creation of the directory failed")
        files = os.listdir(work_path)
        if (len(files) == 0):
            print("Rename failed")
        else:
            files.sort(reverse=True)
            try:
                files.remove('reference')
            except:
                pass
            filename = files[0]

            filename_number = filename[-(len(filename) - 7):]
            print(filename_number)
            new_file_number = int(filename_number) + 1
            new_filename = 'defect_' + str(new_file_number)

            new_dir = os.path.join(work_path, new_filename)
            main_color_path = os.path.join(new_dir, "colors")
            segment_path = os.path.join(new_dir, "segments")

            try:
                os.mkdir(new_dir)
            except:
                print("Final Rename failed")
    try:
        os.mkdir(main_color_path)
        os.mkdir(segment_path)
    except OSError:
        print("Creation of the directory failed")

    else:
        print("Successfully created the directory")

def createFoldersReference():
    try:
        os.mkdir(new_dir)
        os.mkdir(main_color_path)
        os.mkdir(segment_path)

    except OSError:
        print("Creation of the directory failed")

    else:
        print("Successfully created the directory")

global color_dic
color_dic = []

def calculateKValue(img2):
    grey = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY);
    cv2.imshow('grey', grey)

    grey_array = []
    print(grey_array)

    for i in range(256):
        grey_array.append(0)

    width, height = grey.shape
    for i in range(width):
        for j in range(height):
            k = grey[i, j]
            print(k)
            grey_array[k] = grey_array[k] + 1

    print(grey_array)
    sorted_grey_array = np.sort(grey_array)
    sorted_grey_array = sorted_grey_array[::-1]
    max_pixels = sorted_grey_array[1]
    print(max_pixels)
    k_list = []

    for i in grey_array:
        if ((i / max_pixels) > 0.8):
            k_list.append(i)
    print(k_list)
    return len(k_list)

def remove_small_contous(img):

        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]

        cnt_areas = []
        for contour in contours:
            cnt_areas.append(cv2.contourArea(contour))
        cnt_areas_sort = sorted(cnt_areas, reverse=True)

        print("this is cnt areas")
        print(cnt_areas_sort)
        print(cnt_areas)
        print(hierarchy)
        #print(contours)
        items_to_be_removed = []
        for i, cnt in enumerate(contours):
            if hierarchy[0][i][3] == -1 and hierarchy[0][i][2] == -1:
            #if hierarchy[0][i][3] == -1:
                print('remove ready' + str(i))
                if cv2.contourArea(cnt)*200 < cnt_areas_sort[0]:
                    print('removed' + str(i))
                    items_to_be_removed.append(i)
#[a,b,c,d,e,f]#[1,2,4]
#[a,c,d,e,f]#[1,1,4]
#[a,d,e,f]

        print(items_to_be_removed)
        for i,item in enumerate(items_to_be_removed):
            contours.pop(item)
            #hierarchy.pop(item)
            try:
                items_to_be_removed[i+1] = items_to_be_removed[i+1]-(i+1)
            except:
                pass

        mask = np.zeros_like(img)
        cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)
        cv2.imshow("image after small patch removal", mask)
        cv2.waitKey()

        return mask

def write_to_csv(path,fieldnames,data):
    with open(path, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for segment_data in data:
            writer.writerow(segment_data)

def doSegmentation(k_value):

    img = cv2.imread(img_path)
    # img = cv2.resize(img, (1000, 1000))

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    out_image_path = os.path.join(main_color_path, 'morph_' + ".jpg")
    cv2.imwrite(out_image_path, opening)


    # blur = cv2.blur(opening,(2,2))

    #### mean shift algorithm
    spatialRad = 10
    colorRad = 30

    img2 = cv2.pyrMeanShiftFiltering(opening, spatialRad, colorRad)
    out_image_path = os.path.join(main_color_path, 'meanShift_' + ".jpg")
    cv2.imwrite(out_image_path, img2)

    cv2.imshow('meanshift', img2)

    ### apply Kmeans
    img2 = img2.reshape((-1, 3))
    img2 = np.float32(img2)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # k = int(input("K value: "))
    k = k_value
    attempts = 20;

    ret, lable, center = cv2.kmeans(img2, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    print(center)
    arrayTemp = np.zeros((k, 3))

    # looping each color
    for i, element in enumerate(arrayTemp):

        arrayTemp[i] = [255, 255, 255]
        print(i)
        arrayTemp = np.uint8(arrayTemp)
        print(arrayTemp)
        res = arrayTemp[lable.flatten()]
        res3 = res.reshape((img.shape))

        out1 = np.zeros_like(img)  # Extract out the object and place into output image
        out1[res3 == 255] = img[res3 == 255]
        cv2.imshow('color_' + str(i), out1)
        out_image_path = os.path.join(main_color_path, 'color_' + str(i) + ".jpg")
        cv2.imwrite(out_image_path, out1)

        imgray = cv2.cvtColor(res3, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 127, 255, 0)

        kernel = np.ones((2, 2), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        cv2.imshow('image_after_morph_' + str(i), opening)

        contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]

        x, y, w, h = cv2.boundingRect(contours[0]) ## to check if extracted color is background
        print("location of cnt is" + str(x) + "_" + str(y))
        if (x != 0 and y != 0): ## if not background color

            color = {"b": center[i][0],"r": center[i][1], "g": center[i][2] } #to add to csv]

            #rmv noise patches
            patch_rmvd_img = remove_small_contous(opening)

            contours, hierarchy = cv2.findContours(patch_rmvd_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]

            print("hierachy =")
            print(hierarchy)
            print("contour len =" + str(len(contours)))
            stage = 0;
            parent = -1
            lst = []
            for x, cnt in enumerate(contours):
                if (hierarchy[0][x][3] == -1):
                    print(str(x) + " is parent")
                    stage = 0
                    lst.append(stage)
                    print("stage is " + str(stage))
                    stage = stage + 1
                    parent = x
                else:
                    if (hierarchy[0][x][3] == parent):
                        if (hierarchy[0][x][2] == -1):
                            print(str(x) + " is parent without child")
                            lst.append(stage)
                            print("stage is " + str(stage))
                        else:
                            print(str(x) + " is parent with child")
                            lst.append(stage)
                            print("stage is " + str(stage))
                            stage = stage + 1
                            parent = x

                    else:
                        parent = hierarchy[0][x][3]
                        stage = lst[parent] + 1
                        if (hierarchy[0][x][2] == -1):
                            print(str(x) + " is parent without child")
                            lst.append(stage)
                            print("stage is " + str(stage))
                        else:
                            print(str(x) + " is parent with child")
                            lst.append(stage)
                            print("stage is " + str(stage))
                            stage = stage + 1
                            parent = x
            print(lst)
            save_pending = False
            max_level = max(lst)
            print('max of list is ' + str(max_level))
            search_no = 0
            if max_level == 0:

                for a, cnt in enumerate(contours):
                    mask = np.zeros_like(img)
                    cv2.drawContours(mask, contours, a, (255, 255, 255), -1)
                    cv2.imshow("to_draw_new_" + str(i) + "_" + str(a), mask)
                    id =  str(i) + "" + str(search_no) + "" + str(a)
                    out_image_path = os.path.join(segment_path,
                                              str(id) + ".jpg")
                    out = np.zeros_like(img)  # Extract out the object and place into output image
                    out[mask == 255] = img[mask == 255]
                    cv2.imwrite(out_image_path, out)
                    color_dic.append({'id': id, 'color': color })
            else:
                while search_no <= max_level:
                    save_pending = False
                    print("came inside while")
                    for a, cnt in enumerate(contours):
                        print("search no is " + str(search_no))
                        if search_no == lst[a]:
                            if save_pending == True:
                                print("saving image for " + str(i) + "" + str(search_no) + "" + str(a - 1))
                                out = np.zeros_like(img)  # Extract out the object and place into output image
                                out[mask == 255] = img[mask == 255]

                                cv2.imshow("to_draw_new_" + str(i) + "" + str(search_no) + "" + str(a - 1), out)
                                id = str(i) + "" + str(search_no) + "" + str(a - 1)
                                out_image_path = os.path.join(segment_path,
                                                              str(id )  + ".jpg")
                                cv2.imwrite(out_image_path, out)
                                color_dic.append({'id': id, 'color': color })
                                save_pending = False

                            mask = np.zeros_like(img)
                            print("drawing image for " + str(i) + "" + str(search_no) + "" + str(a))
                            cv2.drawContours(mask, contours, a, (255, 255, 255), -1)
                            save_pending = True

                        if lst[a] == search_no + 1:
                            print("drawing child image for " + str(i) + "" + str(search_no) + "" + str(a))
                            cv2.drawContours(mask, contours, a, (0, 0, 0), -1)

                        if a == len(contours) - 1:
                            if save_pending == True:
                                out = np.zeros_like(img)  # Extract out the object and place into output image
                                out[mask == 255] = img[mask == 255]

                                cv2.imshow("to_draw_new_" + str(i) + "" + str(search_no) + "" + str(a), out)
                                id = str(i) + "" + str(search_no) + "" + str(a)
                                out_image_path = os.path.join(segment_path,
                                                              str(id) + ".jpg")
                                color_dic.append({'id': id, 'color': color })
                                cv2.imwrite(out_image_path, out)

                    search_no = search_no + 2

            arrayTemp[i] = [0, 0, 0]
        else:
            arrayTemp[i] = [0, 0, 0]

    write_to_csv(os.path.join(new_dir,'color.csv'),['id','color'],color_dic)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    feature_extract = FeatureExtract(
            dir = new_dir
    )
    ref_segments = feature_extract.get_image_names()

    color_features = feature_extract.read_from_csv()

    full_features = feature_extract.extract_features(ref_segments,color_features)
    print
    print(color_features)
    print(full_features)

    if feature_extract.write_to_csv(['id','color', 'area', 'center','has_child'],full_features):
        return (True, feature_extract.ref_csv_path)
    else:
        return (False, '')


