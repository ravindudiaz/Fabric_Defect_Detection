import os
from os import walk
import csv
import math
import ast
from pprint import pprint
import shutil
import numpy as np
import cv2

# base_path = "H:/FYP/Piu/work"
# img_name = "3"

def get_center_details(img):
    # img = cv2.resize(img,(1000,1000))
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
    largest = contours[0]
    for contour in contours:
        if cv2.contourArea(largest) < cv2.contourArea(contour):
            largest = contour

    M = cv2.moments(largest)

    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    # cv2.line(img, (cX, y), (cX,cY), (0, 255, 0), 4)
    # cv2.circle(img, (cX, cY), 3, (0, 0, 255), -1)

    rect = cv2.minAreaRect(largest)

    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

    # cv2.imshow('image',img)
    # cv2.waitKey()
    return cX,cY, abs(rect[2])

def check_is_offset(ref,test):
    print('Checking for placement errors..')
    cXR, cYR, angleR = get_center_details(ref)
    cXT, cYT, angleT = get_center_details(test)
    if abs(cXR - cXT) > 5 or abs(cYR-cYT) > 5 or abs(angleR - angleT) > 5:
        print('offset found')
        return True
    else:
        print('offset not found')
        return False


def doSegmentMatching(reference_csv,defect_csv):
    # reference_csv = os.path.join(base_path,img_name,'reference','features.csv')
    # defect_csv = os.path.join(base_path,img_name,'defect','features.csv')
    referece_features = []
    defect_features = []
    isOffset = False
    ref_mask = None
    test_mask = None
    for dirpath, dirnames, filenames in os.walk('Assets/BR_Module/Output/artwork_masks_ref'):
        for file in filenames:
            file_path = os.path.join(dirpath,file)
            ref_mask = cv2.imread(file_path)
    for dirpath, dirnames, filenames in os.walk('Assets/BR_Module/Output/artwork_masks_test'):
        for file in filenames:
            file_path = os.path.join(dirpath,file)
            test_mask = cv2.imread(file_path)

    with open(reference_csv, mode='r', newline='') as csv_file:
        reader = csv.DictReader(csv_file)
        for r in reader:
            referece_features.append(r)


    with open(defect_csv, mode='r', newline='') as csv_file:
        reader = csv.DictReader(csv_file)
        for r in reader:
            defect_features.append(r)


    # print(referece_features)
    # print(defect_features)

    def rgb_dif(clr1, clr2)->list:

        delta_E = np.sqrt(np.sum((clr1 - clr2) ** 2, axis=-1)) / 255.
        return (round(delta_E*100,2))

    isOffset = check_is_offset(ref_mask, test_mask)
    matching_id_ls = []
    min_weighted_diff_ls = []
    matching_conflicts_defect = []

    for ref_feature in referece_features:
        total_area_dif = 0
        total_coord_dif = 0
        total_match_ratio = 0
        feature_temp = []
        final_features = []

        print('Feature matching started..')
        for defect_feature in defect_features:

            area_dif = abs(float(ref_feature['area']) - float(defect_feature['area']))

            res1 = ast.literal_eval(ref_feature['center'])
            x1 = res1['X']
            y1 = res1['Y']

            res2 = ast.literal_eval(defect_feature['center'])
            x2 = res2['X']
            y2 = res2['Y']
            # print('x1:'+str(x1)+ ' y1:'+str(y1)+ ' x2:'+str(x2)+ ' y2:'+str(y2) )
            center_X_dif = abs(x1-x2)/x1
            # print('center_X_dif :'+str(center_X_dif) )

            center_Y_dif = abs(y1-y2)/y1
            # print('center_Y_dif :'+str(center_Y_dif))

            center_dif_avg = (center_X_dif + center_Y_dif) /2


            child_dif = 0
            if(ref_feature['has_child'] == defect_feature['has_child']):
                child_dif = 0
            else:
                child_dif = 100

            ref_color = ast.literal_eval(ref_feature['color'])
            ref_r = ref_color['L']
            ref_g = ref_color['A']
            ref_b = ref_color['B']
            ref_color_full = np.array([ref_r,ref_g,ref_b])

            def_color = ast.literal_eval(defect_feature['color'])
            def_r = def_color['L']
            def_g = def_color['A']
            def_b = def_color['B']
            def_color_full = np.array([def_r,def_g,def_b])

            color_dif = rgb_dif(ref_color_full,def_color_full)



            if (isOffset):
                match_ratio = matchShape(reference_csv, ref_feature['id'], defect_csv, defect_feature['id'])
                data = {'id': defect_feature['id'],
                        'color_diff': color_dif,
                        'area_dif' : area_dif,
                        'center_dif' : center_dif_avg,
                        'child_dif' : child_dif,
                        'match_ratio' : match_ratio
                        }

                feature_temp.append(data)
                total_area_dif = total_area_dif + area_dif
                total_coord_dif = total_coord_dif + center_dif_avg
                total_match_ratio = total_match_ratio + match_ratio
            else:

                data = {'id': defect_feature['id'],
                        'color_diff': color_dif,
                        'area_dif': area_dif,
                        'center_dif': center_dif_avg,
                        'child_dif': child_dif,
                        }

                feature_temp.append(data)
                total_area_dif = total_area_dif + area_dif
                total_coord_dif = total_coord_dif + center_dif_avg

        min_weighted_diff = None
        matching_id = None
        initial_conflict = []


        for item in feature_temp:
            if(total_area_dif == 0):
                avg_area_diff = 0
            else:
                avg_area_diff = (item['area_dif']/total_area_dif)*100

            if(total_coord_dif==0):
                avg_center_diff = 0
            else:
                avg_center_diff = (item['center_dif']/total_coord_dif)*100

            if isOffset:
                if(total_match_ratio == 0):
                    avg_match_ratio = 0
                else:
                    avg_match_ratio = (item['match_ratio']/total_match_ratio)*100

            color_dif = item['color_diff']
            if isOffset:
                weighted_diff = (avg_area_diff * 0.3) + (color_dif * 0.4) + (avg_match_ratio*0.3)
                # print('avg_match_ratio ', avg_match_ratio)
            else:
                weighted_diff = (avg_area_diff * 0.3) + (color_dif * 0.3) +(avg_center_diff*0.4)
            # print('id', item['id'])
            # print('color_dif ', color_dif)
            # print('avg_center_diff ', avg_center_diff)
            # print('avg_area_diff: ', avg_area_diff)
            # print('avg_center_diff: ', avg_center_diff)
            # print('weighted_diff ', weighted_diff)


            if min_weighted_diff == weighted_diff:
                if len(initial_conflict) == 0:
                    initial_conflict.append(matching_id)
                    initial_conflict.append(item['id'])
                else:
                    initial_conflict.append(item['id'])


            # if(weighted_diff<10)
            if min_weighted_diff == None:
                min_weighted_diff = weighted_diff
                matching_id = item['id']
                initial_conflict.clear()


            if min_weighted_diff>weighted_diff:
                min_weighted_diff = weighted_diff
                matching_id = item['id']
                initial_conflict.clear()


            #print(avg_area_diff)
        if len(initial_conflict) == 0:
            matching_id_ls.append(matching_id)
            min_weighted_diff_ls.append(min_weighted_diff)

        else:
            data = {'ref_id': ref_feature['id'],
                    'defect_id': initial_conflict
                    }
            matching_conflicts_defect.append(data)


    # print(matching_conflicts_defect)
    # print(matching_id_ls)
    # print(min_weighted_diff_ls)
    count_array = []
    for i in defect_features:
        data = {'id' : i['id'],
                'count' : 0
        }
        count_array.append(data)


    for i , defect_feature in enumerate(defect_features):

        for id in matching_id_ls:
            if defect_feature['id'] == id:
                count_array[i]['count'] = count_array[i]['count'] + 1

        for item in matching_conflicts_defect:
            conflict_ids = item['defect_id']
            for c_id in conflict_ids:
                if c_id == defect_feature['id']:
                    count_array[i]['count'] = -1

    # print(count_array)

    matching_segments = []
    none_matching_defect_segments = []
    none_matching_reference_segments = []
    matching_conflicts_refference = []

    print('Saving matching segments..')
    for i,element in enumerate(count_array):
        if element['count'] > 1:
            least_diff = None
            track_id = None
            save_pending = True
            save_pending_ref_conflicts = False
            ref_conflicts_ids = []

            for x, id in enumerate(matching_id_ls):

                if(id == element['id']):
                    if least_diff == None:
                        least_diff = min_weighted_diff_ls[x]
                        track_id = x
                    else:
                        if least_diff == min_weighted_diff_ls[x]:

                            ref_conflicts_ids.append(referece_features[x]['id'])

                            save_pending_ref_conflicts = True
                            save_pending = False

                        elif least_diff > min_weighted_diff_ls[x]:

                            none_matching_reference_segments.append(referece_features[track_id]['id'])
                            least_diff = min_weighted_diff_ls[x]
                            track_id = x
                            save_pending = True

                        else:

                            none_matching_reference_segments.append(referece_features[x]['id'])

            if save_pending:
                data = {'ref_id': referece_features[track_id] ['id'],
                        'defect_id': element['id']
                        }
                matching_segments.append(data)
            if save_pending_ref_conflicts:
                ref_conflicts_ids.append(referece_features[track_id]['id'])
                data = {'ref_id': ref_conflicts_ids,
                        'defect_id': element['id']
                        }
                matching_conflicts_refference.append(data)
                save_pending_ref_conflicts = False

            # for x,id in enumerate(matching_id_ls):
            pass
        if element['count'] == 1:
            # print(element ['id'])
            for x,id in enumerate(matching_id_ls):
                if id == element ['id']:
                    data = {'ref_id' : referece_features[x]['id'],
                            'defect_id': id
                            }
                    matching_segments.append(data)

        if element['count'] == 0:
            none_matching_defect_segments.append(element ['id'])

    all_matches_ls = [matching_segments,none_matching_reference_segments,none_matching_defect_segments,matching_conflicts_refference,matching_conflicts_defect]
    print('matching_segments')
    pprint(matching_segments)
    print('none_matching_reference_segments')
    print(none_matching_reference_segments)
    print('none_matching_defect_segments')
    print(none_matching_defect_segments)
    print('matching_conflicts_refference')
    print(matching_conflicts_refference)
    print('matching_conflicts_defect')
    print(matching_conflicts_defect)
    print('Segment matching completed with above results..')
    return all_matches_ls

def matchShape(reference_csv,ref_id,defect_csv,defect_id):
    print('shape matching for Ref: ',ref_id+' Test: ',defect_id)
    ref_path = os.path.dirname(reference_csv)
    ref_path = os.path.join(ref_path,'masks',ref_id+'.jpg')
    defect_path = os.path.dirname(defect_csv)
    defect_path = os.path.join(defect_path, 'masks',defect_id+'.jpg')

    print('temp ref path:',ref_path )
    ref_segment = cv2.imread(ref_path)
    ref_segment = cv2.cvtColor(ref_segment, cv2.COLOR_BGR2GRAY)
    ret, ref_segment = cv2.threshold(ref_segment, 127, 255, 0)
    im,ref_segment_contours, hierarchy = cv2.findContours(ref_segment, 2,1)


    defect_segment = cv2.imread(defect_path)
    defect_segment = cv2.cvtColor(defect_segment, cv2.COLOR_BGR2GRAY)
    ret, defect_segment = cv2.threshold(defect_segment, 127, 255, 0)
    im,defect_segment_contours, hierarchy = cv2.findContours(defect_segment, 2,1)


    match_ratio = cv2.matchShapes(ref_segment_contours[0],defect_segment_contours[0], 1, 0.0)
    # print('match ratio:', match_ratio)
    #cv2.waitKey()
    return match_ratio


def copyAndRename(src_file,dest_dir,newName,rename):

    shutil.copy(src_file, dest_dir)  # copy the file to destination dir
    if(rename):
        oldfilename = os.path.basename(src_file)
        dst_file = os.path.join(dest_dir, oldfilename)
        new_dst_file_name = os.path.join(dest_dir, newName)

        os.rename(dst_file, new_dst_file_name)  # rename


def saveMatchingSegments(save_path, all_matches_ls):
    matching_segments = all_matches_ls[0]
    none_matching_reference_segments = all_matches_ls[1]
    none_matching_defect_segments = all_matches_ls[2]
    matching_conflicts_refference = all_matches_ls[3]
    matching_conflicts_defect = all_matches_ls[4]
    base_path_1 = os.path.split(save_path)
    reference_path = os.path.join(base_path_1[0],'reference')
    reference_img_ext = ''
    defect_img_ext = ''

    for (root,dirs,files) in os.walk(os.path.join(reference_path, 'segments')):
        for file in files:
            filename, file_extension = os.path.splitext(file)
            reference_img_ext = file_extension

    for (root,dirs,files) in os.walk(os.path.join(save_path, 'segments')):
        for file in files:
            filename, file_extension = os.path.splitext(file)
            defect_img_ext = file_extension


    matching_segments_path = os.path.join(save_path,'matching_segments' )
    # print(matching_segments_path)
    try:
        os.makedirs(matching_segments_path)
    except:
        print('folder exist')

    matching_reference_segments_path = os.path.join(matching_segments_path,'reference' )
    try:
        os.makedirs(matching_reference_segments_path)
    except:
        print('folder exist')

    matching_defect_segments_path = os.path.join(matching_segments_path,'defect' )
    try:
        os.makedirs(matching_defect_segments_path)
    except:
        print('folder exist')

    for i,item in enumerate(matching_segments):

        defect_src_file = os.path.join(save_path,'segments',item['defect_id'] + defect_img_ext)
        new_defect_name = 'M_'+str(i)+defect_img_ext
        # print(defect_src_file)
        copyAndRename(defect_src_file, matching_defect_segments_path, new_defect_name,True)

        ref_src_file = os.path.join(reference_path, 'segments',item['ref_id'] + reference_img_ext)
        new_ref_name = 'M_' + str(i) +reference_img_ext
        copyAndRename(ref_src_file, matching_reference_segments_path, new_ref_name,True)

    if (len(none_matching_defect_segments) != 0) or len(none_matching_reference_segments):
        try:
            none_matching_segments_path = os.path.join(save_path,'none_matching_segments' )
            os.makedirs(none_matching_segments_path)
        except:
            print('folder exist')

        if len(none_matching_reference_segments) != 0:
            none_matching_reference_segments_path = os.path.join(none_matching_segments_path,'reference' )
            try:
                os.makedirs(none_matching_reference_segments_path)
            except:
                print('folder exist')

            for i, item in enumerate(none_matching_reference_segments):

                ref_src_file = os.path.join(reference_path, 'segments',str(item) + reference_img_ext)
                new_ref_name = 'N_' + str(i) + reference_img_ext
                copyAndRename(ref_src_file, none_matching_reference_segments_path, new_ref_name,True)

        if len(none_matching_defect_segments) != 0:
            none_matching_defect_segments_path = os.path.join(none_matching_segments_path, 'defect')
            try:
                os.makedirs(none_matching_defect_segments_path)
            except:
                print('folder exist')

            for i, item in enumerate(none_matching_defect_segments):
                defect_src_file = os.path.join(save_path,'segments', str(item) + defect_img_ext)
                new_defect_name = 'N_' + str(i) + defect_img_ext
                copyAndRename(defect_src_file, none_matching_defect_segments_path, new_defect_name, True)

    if (len(matching_conflicts_refference) != 0) or (len(matching_conflicts_defect) != 0):
        try:
            matching_conflicts_path = os.path.join(save_path,'conflict_segments' )
            os.makedirs(matching_conflicts_path)
        except:
            print('folder exist')

        try:
            matching_conflicts_refference_path = os.path.join(matching_conflicts_path, 'reference')
            os.makedirs(matching_conflicts_refference_path)
        except:
            print('folder exist')

        try:
            matching_conflicts_defect_path = os.path.join(matching_conflicts_path, 'defect')
            os.makedirs(matching_conflicts_defect_path)
        except:
            print('folder exist')

        if(len(matching_conflicts_refference) != 0):


            for i, item in enumerate(matching_conflicts_refference):

                ref_ids = item['ref_id']

                for count,id in enumerate(ref_ids):
                    ref_src_file = os.path.join(reference_path, 'segments', str(id) + reference_img_ext)
                    new_ref_name = 'CR_' + str(i)+'_'+str(count) + reference_img_ext
                    copyAndRename(ref_src_file, matching_conflicts_refference_path, new_ref_name, True)


                defect_src_file = os.path.join(save_path, 'segments', item['defect_id'] + defect_img_ext)
                new_defect_name = 'CR_' + str(i) + defect_img_ext
                # print(defect_src_file)
                copyAndRename(defect_src_file, matching_conflicts_defect_path, new_defect_name, True)


        if (len(matching_conflicts_defect) != 0):


            for i, item in enumerate(matching_conflicts_defect):

                defect_ids = item['defect_id']


                ref_src_file = os.path.join(reference_path, 'segments', item['ref_id'] + reference_img_ext)
                new_ref_name = 'CD_' + str(i) + reference_img_ext
                copyAndRename(ref_src_file, matching_conflicts_refference_path, new_ref_name, True)

                for count,id in enumerate(defect_ids):
                    defect_src_file = os.path.join(save_path, 'segments', str(id) + defect_img_ext)
                    new_defect_name = 'CD_' + str(i)+'_'+str(count) + defect_img_ext
                    # print(defect_src_file)
                    copyAndRename(defect_src_file, matching_conflicts_defect_path, new_defect_name, True)
