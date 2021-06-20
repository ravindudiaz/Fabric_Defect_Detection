import numpy as np
import cv2
import os
import csv
from feature_extract import FeatureExtract
import shutil
import matplotlib.pyplot as plt


### create folder structire
img_path = ""
img_mask_path = ""
new_dir = ""
main_color_path = ""
segment_path = ""
mask_path = ""
isReference = False
isTextured = False

def setFolderNames( work_path, sub_folder_name):
    global new_dir,main_color_path,segment_path,img_path
    work_path = './Assets/Seg_Module/Output/'

    new_dir = os.path.join(work_path, sub_folder_name )
    main_color_path = os.path.join(new_dir, "colors")
    segment_path = os.path.join(new_dir, "segments")



def setFolderNames_reference():
    global new_dir, main_color_path, segment_path, img_path,mask_path, isReference,img_mask_path,isTextured

    for dirpath, dirnames, filenames in os.walk('Assets/BR_Module/Output/artworks_ref'):
        for file in filenames:
            if 'uni_' in file:
                isTextured = False
                print('image is uniform')
            else:
                isTextured = True
                print('image is textured')
            file_path = os.path.join(dirpath,file)
            img_path = file_path

    for dirpath, dirnames, filenames in os.walk('Assets/BR_Module/Output/artwork_masks_ref'):
        for file in filenames:
            file_path = os.path.join(dirpath,file)
            img_mask_path = file_path

    new_dir='Assets/Seg_Module/Output/reference'

    main_color_path = os.path.join(new_dir, "colors")
    segment_path = os.path.join(new_dir, "segments")
    mask_path = os.path.join(new_dir, "masks")
    isReference = True
    return createFolders()



def setFolderNames_defect():
    global new_dir, main_color_path, segment_path, img_path,mask_path,isReference,img_mask_path,isTextured

    for dirpath, dirnames, filenames in os.walk('Assets/BR_Module/Output/artworks_test'):
        for file in filenames:
            if 'uni_' in file:
                isTextured = False
                print('image is uniform')
            else:
                isTextured = True
                print('image is textured')
            file_path = os.path.join(dirpath, file)
            img_path = file_path

    for dirpath, dirnames, filenames in os.walk('Assets/BR_Module/Output/artwork_masks_test'):
        for file in filenames:
            file_path = os.path.join(dirpath, file)
            img_mask_path = file_path

    new_dir='Assets/Seg_Module/Output/defect'

    main_color_path = os.path.join(new_dir, "colors")
    segment_path = os.path.join(new_dir, "segments")
    mask_path = os.path.join(new_dir, "masks")
    isReference = False
    return createFolders()


def createFolders():
    if os.path.exists(new_dir):
        shutil.rmtree(new_dir, ignore_errors=True)
    try:
        os.mkdir(new_dir)
        os.mkdir(main_color_path)
        os.mkdir(segment_path)
        os.mkdir(mask_path)


    except OSError:
        print("Creation of directory failed")
        return False

    else:
        print("Successfully created the directory")
        return True

global color_dic
color_dic = []

###k value calculation

def save_k_value(data):
    path = os.path.join(new_dir, 'config.csv')
    with open(path, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['L','A','B'])
        writer.writeheader()
        for d in data:
            writer.writerow(d)

def remove_small_contous(img):

        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]

        cnt_areas = []
        for contour in contours:
            cnt_areas.append(cv2.contourArea(contour))
        cnt_areas_sort = sorted(cnt_areas, reverse=True)
        print(cnt_areas_sort )
        items_to_be_removed = []
        for i, cnt in enumerate(contours):
            if cv2.contourArea(cnt) < 130:
                items_to_be_removed.append(i)
            else:
                if hierarchy[0][i][3] == -1 and hierarchy[0][i][2] == -1:
                #if hierarchy[0][i][3] == -1:
                    # print('remove ready' + str(i))
                    # if cv2.contourArea(cnt)*500 < cnt_areas_sort[0]:
                    if cv2.contourArea(cnt) * 500 < cnt_areas_sort[0]:
                        print('removed' + str(i))
                        items_to_be_removed.append(i)
#[a,b,c,d,e,f]#[1,2,4]
#[a,c,d,e,f]#[1,1,4]
#[a,d,e,f]

        # print(items_to_be_removed)
        for i,item in enumerate(items_to_be_removed):
            contours.pop(item)
            #hierarchy.pop(item)
            try:
                items_to_be_removed[i+1] = items_to_be_removed[i+1]-(i+1)
            except:
                pass

        mask = np.zeros_like(img)
        if(len(contours)>0):
            cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)
        mask_resized = cv2.resize(mask, (500, 500))
        # cv2.imshow("image after small patch removal", mask_resized)
        # cv2.waitKey()

        return mask

def write_to_csv(path,fieldnames,data):
    with open(path, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for segment_data in data:
            writer.writerow(segment_data)
####new approach

def unique_count(a):
    img = cv2.resize(a, (700, 700))

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    #### mean shift algorithm
    spatialRad = 150
    colorRad = 50

    img2 = cv2.pyrMeanShiftFiltering(opening, spatialRad, colorRad)
    img_lab = cv2.cvtColor(img2.astype(np.float32) / 255, cv2.COLOR_RGB2Lab)
    # cv2.imshow('mean',img2)
    # cv2.waitKey()
    sorted_colors = []
    sorted_count = []
    colors, count = np.unique(img_lab.reshape(-1,a.shape[-1]), axis=0, return_counts=True)

    for i,value in enumerate(count):
        if value>100:
            sorted_count.append(value)
            sorted_colors.append(colors[i])

    color_check = np.zeros_like(sorted_count)
    # print(sorted_colors)
    unique_color = []
    for index1,color in enumerate(sorted_colors):
        if color_check[index1] != -1:
            unique_color.append(color)

            for index2,color2 in enumerate(sorted_colors):
                if (index1 != index2):
                    delta_E = np.sqrt(np.sum((color - color2) ** 2, axis=-1)) / 255.
                    delta_E = round(delta_E * 100, 10)

                    if(delta_E<5):
                        color_check[index2] = -1

    print('No of unique colors found: ',len(unique_color))
    print(unique_color)
    return unique_color

def get_most_matching(color1,clr_list):
    matching = None
    dif = None
    index = None
    for i,clr in enumerate(clr_list):
        delta_E = np.sqrt(np.sum((color1 - clr) ** 2, axis=-1)) / 255.
        delta_E = round(delta_E * 100, 2)
        if dif == None:
            dif = delta_E
            matching = clr
            index = i
        else:
            if delta_E < dif:
                dif = delta_E
                matching = clr
                index = i
    return matching,index
def refine_mask(mask):

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
    for c in contours:
        cv2.drawContours(mask, c, -1, [0, 0, 0], 2)
    return mask

def fill_colors(img,img_mask,colors):
    global isTextured
    print('Segmenting main colors....')
    # imgray = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(img_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]

    mask2 = img_mask.copy()
    thickness = 3
    if isTextured:
        thickness = 0
    boxes = []
    print('thickness is',thickness)
    print('isTextured: ',isTextured)
    for c in contours:
        cv2.drawContours(mask2, c, -1, [0, 0, 0], thickness)
        (x, y, w, h) = cv2.boundingRect(c)
        boxes.append([x, y, x + w, y + h])

    boxes = np.asarray(boxes)
    left, top = np.min(boxes, axis=0)[:2]
    right, bottom = np.max(boxes, axis=0)[2:]
    print(left,top,right,bottom)

    mask_ls = []
    for clr in colors:
        out = np.zeros_like(img)
        mask_ls.append(out)

    for h1 in range(top,bottom):
        for w1 in range(left,right):
            # new_clr, index = get_most_matching(img[h1][w1], colors)
            # img[h1][w1] = new_clr
            # # print(new_clr)
            # mask_ls[index][h1][w1] = [255, 255, 255]
            if (img_mask[h1][w1] == [0]).any() :
                # print('in mask')
                img[h1][w1] = [0,0,0]
            else:
                new_clr,index = get_most_matching(img[h1][w1],colors)
                if(mask2[h1][w1] == [0]).any() :
                    pass
                else:
                    mask_ls[index][h1][w1] = [255, 255, 255]

                # print(img[h1][w1])
                # if (np.array_equiv(new_clr,np.array([0,0,0]))):
                #     if(mask2[h1][w1] == [0]).any() :
                #         pass
                #     else:
                #         mask_ls[index][h1][w1] = [255, 255, 255]
                # else:
                #     # print('front')
                #     mask_ls[index][h1][w1] = [255, 255, 255]
                img[h1][w1] = new_clr

    return img,mask_ls

def get_dim(img):
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
    largest = contours[0]
    for contour in contours:
        if cv2.contourArea(largest)<cv2.contourArea(contour):
            largest = contour
    x, y, w, h = cv2.boundingRect(largest)

    return(x, y, w, h)

def doSegmentation():

    # path = 'Assets/Output/BR_Module/Input/'
    # path_mask = 'H:/FYP/Piu/dataset-20210522T160245Z-001/k_value_test/tex_girlheart_mask.jpg'

    img_mask = cv2.imread(img_mask_path)
    width_1, height_1, depth_1 = img_mask.shape
    ratio = 0.5
    img_mask = cv2.resize(img_mask, (int(height_1 * ratio), int(width_1 * ratio)))
    # x, y, w, h = get_dim(img_mask)
    img_mask_gray = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_mask_gray, 127, 255, 0)
    img_mask = remove_small_contous(thresh)
    # cv2.imshow('img mask',img_mask)
    if not isTextured:
        img_mask = refine_mask(img_mask)
    # cv2.imshow('refined_img mask', img_mask)
    # cv2.waitKey()
    img = cv2.imread(img_path)
    img_original = img.copy()

    img = cv2.imread(img_path)

    img = cv2.resize(img, (int(height_1 * ratio), int(width_1 * ratio)))

    clr_lst = []
    if isReference:
        print('Finding unique color values...')
        data_ls = []
        clr_lst = unique_count(img)
        for clr in clr_lst:
            data = {'L': clr[0], 'A' : clr[1], 'B': clr[2]}
            data_ls.append(data)
        save_k_value(data_ls)

    else:
        print('Reading unique color values...')
        with open('Assets/Seg_Module/Output/reference/config.csv', mode='r', newline='') as csv_file:
            reader = csv.DictReader(csv_file)
            for r in reader:
                L = float(r['L'])
                A = float(r['A'])
                B = float(r['B'])
                clr_lst.append([L,A,B])
        print('Unique values: ',clr_lst)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    out_image_path = os.path.join(main_color_path, 'morph_' + ".jpg")
    cv2.imwrite(out_image_path, opening)

    #### mean shift algorithm
    spatialRad = 20
    colorRad = 30

    img = cv2.pyrMeanShiftFiltering(opening, spatialRad, colorRad)
    out_image_path = os.path.join(main_color_path, 'meanShift_' + ".jpg")
    cv2.imwrite(out_image_path, img)

    img = cv2.medianBlur(img, 5)

    img_lab = cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_RGB2Lab)
    img_lab = cv2.resize(img_lab, (int(height_1 * ratio), int(width_1 * ratio)))

    new_img, masks = fill_colors(img_lab, img_mask, clr_lst)
    print('main color segmentationg complete..')
    # new_img = cv2.cvtColor(new_img, cv2.COLOR_Lab2BGR)
    # new_img = cv2.resize(new_img, (height_1, width_1))
    # median = cv2.medianBlur(new_img, 5)


    new_masks = []
    for i,mask in enumerate(masks):
        mask = cv2.resize(mask, (height_1, width_1))
        mask_show = cv2.resize(mask, (500, 500))
        # cv2.imshow('mask', mask_show)
        # cv2.waitKey()
        out = np.zeros_like(img_original)  # Extract out the object and place into output image
        out[mask == 255] = img_original[mask == 255]
        out_image_path = os.path.join(main_color_path, 'color_' + str(i) + ".jpg")
        cv2.imwrite(out_image_path, out)
        imgray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

        # new = plt.imshow(mask )
        # plt.show()
        blur = cv2.GaussianBlur(imgray, (5, 5), 0)
        ret3, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        new_masks.append(thresh)
    # print("--- %s seconds ---" % (time.time() - start_time))

    for i,mask in enumerate(new_masks):

        color = {"L": clr_lst[i][0], "A": clr_lst[i][1], "B": clr_lst[i][2]}  # to add to csv]
        print('Segmenting children of color: ', color )
        kernel = np.ones((2, 2), np.uint8)
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # rmv noise patches
        patch_rmvd_img = remove_small_contous(opening)
        # cv2.imshow('patch_rmvd_img',patch_rmvd_img)
        # cv2.waitKey()
        contours, hierarchy = cv2.findContours(patch_rmvd_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]

        # print("hierachy =")
        # print(hierarchy)
        # print("contour len =" + str(len(contours)))
        stage = 0;
        parent = -1
        lst = []
        # cv2.imshow('mask', mask)
        if len(contours) != 0 :
            for x, cnt in enumerate(contours):
                if (hierarchy[0][x][3] == -1):
                    # print(str(x) + " is parent")
                    stage = 0
                    lst.append(stage)
                    # print("stage is " + str(stage))
                    stage = stage + 1
                    parent = x
                else:
                    if (hierarchy[0][x][3] == parent):
                        if (hierarchy[0][x][2] == -1):
                            # print(str(x) + " is parent without child")
                            lst.append(stage)
                            # print("stage is " + str(stage))
                        else:
                            # print(str(x) + " is parent with child")
                            lst.append(stage)
                            # print("stage is " + str(stage))
                            stage = stage + 1
                            parent = x

                    else:
                        parent = hierarchy[0][x][3]
                        stage = lst[parent] + 1
                        if (hierarchy[0][x][2] == -1):
                            # print(str(x) + " is parent without child")
                            lst.append(stage)
                            # print("stage is " + str(stage))
                        else:
                            # print(str(x) + " is parent with child")
                            lst.append(stage)
                            # print("stage is " + str(stage))
                            stage = stage + 1
                            parent = x
            # print(lst)
            save_pending = False
            max_level = max(lst)
            # print('max of list is ' + str(max_level))
            search_no = 0
            if max_level == 0:

                for a, cnt in enumerate(contours):
                    mask = np.zeros_like(img_original)
                    cv2.drawContours(mask, contours, a, (255, 255, 255), -1)

                    id = str(i) + "" + str(search_no) + "" + str(a)
                    out_image_path = os.path.join(segment_path,
                                                  str(id) + ".jpg")
                    out_mask_path = os.path.join(mask_path,
                                                 str(id) + ".jpg")
                    out = np.zeros_like(img_original)  # Extract out the object and place into output image
                    out[mask == 255] = img_original[mask == 255]
                    cv2.imwrite(out_image_path, out)
                    cv2.imwrite(out_mask_path, mask)
                    color_dic.append({'id': id, 'color': color})
            else:
                while search_no <= max_level:
                    save_pending = False
                    # print("came inside while")
                    for a, cnt in enumerate(contours):
                        # print("search no is " + str(search_no))
                        if search_no == lst[a]:
                            if save_pending == True:
                                # print("saving image for " + str(i) + "" + str(search_no) + "" + str(a - 1))
                                out = np.zeros_like(img_original)  # Extract out the object and place into output image
                                out[mask == 255] = img_original[mask == 255]

                                # cv2.imshow("to_draw_new_" + str(i) + "" + str(search_no) + "" + str(a - 1), out)
                                id = str(i) + "" + str(search_no) + "" + str(a - 1)
                                out_image_path = os.path.join(segment_path,
                                                              str(id) + ".jpg")
                                out_mask_path = os.path.join(mask_path,
                                                             str(id) + ".jpg")
                                cv2.imwrite(out_image_path, out)
                                cv2.imwrite(out_mask_path, mask)
                                color_dic.append({'id': id, 'color': color})
                                save_pending = False

                            mask = np.zeros_like(img_original)
                            # print("drawing image for " + str(i) + "" + str(search_no) + "" + str(a))
                            cv2.drawContours(mask, contours, a, (255, 255, 255), -1)
                            save_pending = True

                        if lst[a] == search_no + 1:
                            # print("drawing child image for " + str(i) + "" + str(search_no) + "" + str(a))
                            cv2.drawContours(mask, contours, a, (0, 0, 0), -1)

                        if a == len(contours) - 1:
                            if save_pending == True:
                                out = np.zeros_like(img_original)  # Extract out the object and place into output image
                                out[mask == 255] = img_original[mask == 255]

                                # cv2.imshow("to_draw_new_" + str(i) + "" + str(search_no) + "" + str(a), out)
                                id = str(i) + "" + str(search_no) + "" + str(a)
                                out_image_path = os.path.join(segment_path,
                                                              str(id) + ".jpg")
                                out_mask_path = os.path.join(mask_path,
                                                             str(id) + ".jpg")
                                color_dic.append({'id': id, 'color': color})
                                cv2.imwrite(out_image_path, out)
                                cv2.imwrite(out_mask_path, mask)

                    search_no = search_no + 2


    write_to_csv(os.path.join(new_dir,'color.csv'),['id','color'],color_dic)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    feature_extract = FeatureExtract(
            dir = new_dir
    )
    ref_segments = feature_extract.get_image_names()

    color_features = feature_extract.read_from_csv()

    print('Extracting features..')
    full_features = feature_extract.extract_features(ref_segments,color_features)

    # print(color_features)
    # print(full_features)
    print('Writing features..')
    if feature_extract.write_to_csv(['id','color', 'area', 'center','has_child'],full_features):
        print('Segmentation complete!')
        return (True, feature_extract.ref_csv_path)
    else:
        return (False, '')


