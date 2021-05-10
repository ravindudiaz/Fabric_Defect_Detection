import os
from os import walk
import csv
import math
import ast
from pprint import pprint
import shutil

# base_path = "H:/FYP/Piu/work"
# img_name = "3"

def doSegmentMatching(reference_csv,defect_csv):
    # reference_csv = os.path.join(base_path,img_name,'reference','features.csv')
    # defect_csv = os.path.join(base_path,img_name,'defect','features.csv')
    referece_features = []
    defect_features = []

    with open(reference_csv, mode='r', newline='') as csv_file:
        reader = csv.DictReader(csv_file)
        for r in reader:
            referece_features.append(r)


    with open(defect_csv, mode='r', newline='') as csv_file:
        reader = csv.DictReader(csv_file)
        for r in reader:
            defect_features.append(r)


    print(referece_features)
    print(defect_features)

    def rgb_dif(clr1, clr2)->list:
        rmean = (clr1[1] + clr2[1])/2
        r = clr1[1] - clr2[1]
        g = clr1[2] - clr2[2]
        b = clr1[0] - clr2[0]
        return math.sqrt((((512+rmean)*r*r)>>8) + 4*g*g + (((767-rmean)*b*b)>>8))


    matching_id_ls = []
    min_weighted_diff_ls = []

    # print(rgb_dif([255,255,255],[255,255,255]))
    for ref_feature in referece_features:
        total_area_dif = 0
        total_coord_dif = 0
        feature_temp = []
        final_features = []

        print('...................')
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

            # print('area_dif:')
            # print(round(area_dif, 2))
            # print('center_dif:')
            # print(round(center_dif_avg, 2) )

            data = {'id': defect_feature['id'],
                    'area_dif' : area_dif,
                    'center_dif' : center_dif_avg,
                    'child_dif' : child_dif
                    }

            feature_temp.append(data)
            total_area_dif = total_area_dif + area_dif
            total_coord_dif = total_coord_dif + center_dif_avg

            #total_avg = (area_dif + center_dif_avg +center_dif_avg) /3
            #print(total_avg)

        # print(feature_temp)
        # print(total_area_dif)
        # print(total_coord_dif)
        min_weighted_diff = None
        matching_id = None
        for item in feature_temp:
            avg_area_diff = (item['area_dif']/total_area_dif)*100
            avg_center_diff = (item['center_dif']/total_coord_dif)*100
            child_dif = item['child_dif']
            weighted_diff = (avg_area_diff+avg_center_diff+child_dif)/3

            # if(weighted_diff<10)
            if min_weighted_diff == None:
                min_weighted_diff = weighted_diff
                matching_id = item['id']
            if min_weighted_diff>weighted_diff:
                min_weighted_diff = weighted_diff
                matching_id = item['id']

            #print(avg_area_diff)
        matching_id_ls.append(matching_id)
        min_weighted_diff_ls.append(min_weighted_diff)

    print(matching_id_ls)
    print(min_weighted_diff_ls)
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

    # print(defect_features)
    # print(count_array)

    matching_segments = []
    none_matching_defect_segments = []
    none_matching_reference_segments = []
    matching_conflicts = []

    for i,element in enumerate(count_array):
        if element['count'] > 1:
            least_diff = None
            track_id = None
            save_pending = True

            for x, id in enumerate(matching_id_ls):

                if(id == element['id']):
                    if least_diff == None:
                        least_diff = min_weighted_diff_ls[x]
                        track_id = x
                    else:
                        if least_diff == min_weighted_diff_ls[x]:
                            matching_conflicts.append([track_id,x])
                            save_pending = False

                        if least_diff > min_weighted_diff_ls[x]:
                            print('new match')
                            print(id)
                            none_matching_reference_segments.append(referece_features[track_id]['id'])
                            least_diff = min_weighted_diff_ls[x]
                            track_id = x
                            save_pending = True

                        else:
                            print('old match')
                            print(id)
                            none_matching_reference_segments.append(referece_features[x]['id'])

            if save_pending:
                data = {'ref_id': referece_features[track_id] ['id'],
                        'defect_id': element['id']
                        }
                matching_segments.append(data)

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

    all_matches_ls = [matching_segments,none_matching_reference_segments,none_matching_defect_segments,matching_conflicts]
    print('matching_segments')
    pprint(matching_segments)
    print('none_matching_reference_segments')
    print(none_matching_reference_segments)
    print('none_matching_defect_segments')
    print(none_matching_defect_segments)
    print('matching_conflicts')
    print(matching_conflicts)

    return all_matches_ls

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
    matching_conflicts = all_matches_ls[3]
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
    print(matching_segments_path)
    try:
        os.makedirs(matching_segments_path)
    except:
        print('folder exist')

    for i,item in enumerate(matching_segments):

        defect_src_file = os.path.join(save_path,'segments',item['defect_id'] + defect_img_ext)
        new_defect_name = 'set_'+str(i)+'_defect'+defect_img_ext
        print(defect_src_file)
        copyAndRename(defect_src_file, matching_segments_path, new_defect_name,True)

        ref_src_file = os.path.join(reference_path, 'segments',item['ref_id'] + reference_img_ext)
        new_ref_name = 'set_' + str(i) + '_reference'+reference_img_ext
        copyAndRename(ref_src_file, matching_segments_path, new_ref_name,True)

    if len(none_matching_reference_segments) != 0:
        none_matching_reference_segments_path = os.path.join(save_path,'none_matching_reference_segments' )
        try:
            os.makedirs(none_matching_reference_segments_path)
        except:
            print('folder exist')

        for i, item in enumerate(none_matching_reference_segments):

            ref_src_file = os.path.join(reference_path, 'segments',str(item) + reference_img_ext)
            copyAndRename(ref_src_file, none_matching_reference_segments_path, '',False)

    if len(none_matching_defect_segments) != 0:
        none_matching_defect_segments_path = os.path.join(save_path, 'none_matching_defect_segments')
        try:
            os.makedirs(none_matching_defect_segments_path)
        except:
            print('folder exist')

        for i, item in enumerate(none_matching_defect_segments):
            ref_src_file = os.path.join(save_path,'segments', str(item) + defect_img_ext)
            copyAndRename(ref_src_file, none_matching_defect_segments_path, '', False)

