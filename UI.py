import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog
from tkinter import messagebox
import os
import Segmentation_v2 as Segmentation
import segment_matching
from PIL import ImageTk, Image
import QA_Module as qa
import  cv2
import BR_Module_For_UI as br

#location paths-----------------------------------------


matching_ref_loc ="./Assets/Seg_Module/Output/defect/matching_segments/reference/"
matching_test_loc = "./Assets/Seg_Module/Output/defect/matching_segments/defect/"
nonmatching_ref_loc = "./Assets/Seg_Module/Output/defect/none_matching_segments/reference/"
nonmatching_test_loc= "./Assets/Seg_Module/Output/defect/none_matching_segments/defect/"
nonmatching_ref_conflict = "./Assets/Seg_Module/Output/defect/conflict_segments/reference/"
nonmatching_test_conflict = "./Assets/Seg_Module/Output/defect/conflict_segments/defect/"


#ref artwork & cloth loc
ref_artwork_mask_loc = "./Assets/BR_Module/Output/mask/ref/artwork/"
ref_or_cloth_loc = "./Assets/BR_Module/Output/ref/outer_removed_ref/"  #outer removed ref

#test artwork &cloth loc
test_artwork_mask_loc = "Assets/BR_Module/Output/mask/test/artwork/"
test_or_cloth_loc = "Assets/BR_Module/Output/test/outer_removed_test/"  #outer removed test

#ref isolated artwork loc
ref_artwork_loc = "./Assets/BR_Module/Output/artworks_ref/"

#test isolated artwork loc
test_artwork_loc = "./Assets/BR_Module/Output/test/artworks_test/"


#might need to be adjusted as per segment rois
ref_seg_roi_loc = "./Assets/QA_Module/Output/rois"
test_seg_roi_loc = "./Assets/QA_Module/Output/rois"


#reference vars
# ref_features = []
# ref_thresholded_segs = []
# ref_dimensions = []
# ref_segs = []
# #-----------------------------------------------------------------------------------------------------------------------






root = tk.Tk()
root.geometry('1600x1000')
root.title('Defect Detector')
root.configure(background='white')
s1 = ttk.Style()

segmentMatchReportWindow = None


ref_img_thumb = 'Assets/src/ref_img_thumb.jpg'

test_img_thumb = 'Assets/src/test_img_thumb.jpg'

tex_sample_img_thumb = 'Assets/src/textured_sample_thumb.jpg'

resizeVals = (200, 200)


def getRef():
    
    img = Image.open(ref_img_thumb).resize(resizeVals)

    return img

def getTest():

    img = Image.open(test_img_thumb).resize(resizeVals)

    return img

def getTexSample():

    img = Image.open(tex_sample_img_thumb).resize(resizeVals)

    return img

ref_image =ImageTk.PhotoImage(getRef())

test_image = ImageTk.PhotoImage(getTest())

tex_sample_image = ImageTk.PhotoImage(getTexSample())




def ImageBrowser_ref():
    global ref_image
    filename = filedialog.askopenfilename( title = "Select Refference File", filetypes=[("jpg files","*.jpg"), ("jpeg files","*,jpeg"), ("png files","*.png"), ("all files","*.*")])
    imageLabel_ref.config(text=filename)
    if(filename != ''):
        img = Image.open(filename)
        img = img.resize(resizeVals,Image.ANTIALIAS)
        ref_image = ImageTk.PhotoImage(img )
        ref_img_label.config(image = ref_image)
    else:
        ref_image = ImageTk.PhotoImage(getRef())
        ref_img_label.config(image=ref_image)

def ImageBrowser_test():
    global test_image
    filename = filedialog.askopenfilename(title="Select Refference File",filetypes=[("jpg files", "*.jpg"), ("jpeg files", "*,jpeg"),("png files", "*.png"), ("all files", "*.*")])
    imageLabel_test.config(text=filename)
    if (filename != ''):
        img = Image.open(filename)
        img = img.resize(resizeVals, Image.ANTIALIAS)
        test_image = ImageTk.PhotoImage(img)
        test_img_label.config(image=test_image)
    else:
        test_image = ImageTk.PhotoImage(getTest())
        test_img_label.config(image=test_image)

def ImageBrowser_TexSample():
    global tex_sample_image
    filename = filedialog.askopenfilename(title="Select Refference File",filetypes=[("jpg files", "*.jpg"), ("jpeg files", "*,jpeg"),("png files", "*.png"), ("all files", "*.*")])
    imageLabel_TexSample.config(text=filename)
    if (filename != ''):
        img = Image.open(filename)
        img = img.resize(resizeVals, Image.ANTIALIAS)
        tex_sample_image = ImageTk.PhotoImage(img)
        tex_sample_img_label.config(image=tex_sample_image)
    else:
        tex_sample_image = ImageTk.PhotoImage(getTexSample())
        tex_sample_img_label.config(image=tex_sample_image)

def segmentImage_ref():

    image_path = imageLabel_ref.cget("text")

    if (image_path == ''):
        messagebox.showerror("Invalid path", "Invalid image path.Please make a valid selection")
        return

    else:

        if Segmentation.setFolderNames_reference():

            status,csv_path = Segmentation.doSegmentation()
            if(status):
                messagebox.showinfo("Segmentation Success!", "Image segmented successfully")
                referenceFeaturesLabel.config(text=csv_path )
            else:
                messagebox.showerror("Segmentation Failed", "Something went wrong while segmenting the image")
        else:
            messagebox.showerror("Segmentation Failed", "Something went wrong while creating directory tree.Please try again")

def segmentImage_test():
    image_path = imageLabel_test.cget("text")

    if (image_path == ''):
        messagebox.showerror("Invalid path", "Invalid image path.Please make a valid selection")
        return

    else:

        if Segmentation.setFolderNames_defect():

            status,csv_path = Segmentation.doSegmentation()
            if(status):
                messagebox.showinfo("Segmentation Success!", "Image segmented successfully")
                defectFeaturesLabel.config(text=csv_path )
            else:
                messagebox.showerror("Segmentation Failed", "Something went wrong while segmenting the image")
        else:
            messagebox.showerror("Segmentation Failed", "Something went wrong while creating directory tree.Please try again")

def featuresBrowser(selection):
    filename = filedialog.askopenfilename( title = "Select Feature File", filetypes=[("csv","*.csv")])
    if selection == 'reference':
        referenceFeaturesLabel.config(text=filename)
    if selection == 'defect':
        defectFeaturesLabel.config(text=filename)

def matchSegments():
    referenceImgCSV = referenceFeaturesLabel.cget("text")
    defectImgCSV = defectFeaturesLabel.cget("text")

    if(referenceImgCSV == '' or defectImgCSV == ''):
        messagebox.showerror("Invalid paths", "Invalid segments CSV paths")

    else:
        matches = segment_matching.doSegmentMatching(referenceImgCSV,defectImgCSV)
        path = os.path.dirname(defectImgCSV)

        segment_matching.saveMatchingSegments(path,matches)
        showSegmentMatchReport(matches )
        # except:
        #     messagebox.showerror("Segment Matching Failed", "Something went wrong while segment matching")

def showSegmentMatchReport(segments):
    global segmentMatchReportWindow

    if (segmentMatchReportWindow is not None):
        try:
            if segmentMatchReportWindow.winfo_exists():
                segmentMatchReportWindow.destroy()

        except:
            pass

    segmentMatchReportWindow = tk.Toplevel(root)

    segmentMatchReportWindow.title("Segment Matching Summary")

    segmentMatchReportWindow.geometry('600x500')

    progress_frame1 = ttk.Frame(segmentMatchReportWindow)
    progress_frame1.pack(pady=20)
    label_1Txt = ttk.Label(progress_frame1, text='Segment matching completed with following results:', font=("Calibrir", 10, 'bold'))
    label_1Txt.grid(row=0, column=1)

    progress_frame2 = ttk.Frame(segmentMatchReportWindow)
    progress_frame2.pack(pady=20)
    label_2_1Txt = ttk.Label(progress_frame2, text="Matching Segments")
    label_2_1Txt.grid(row=1, column=0)
    label_2_2Txt = ttk.Label(progress_frame2, text="Count: "+str(len(segments[0])))
    label_2_2Txt.grid(row=1, column=2)

    progress_frame3 = ttk.Frame(segmentMatchReportWindow)
    progress_frame3.pack(pady=20)
    label_2_1Txt = ttk.Label(progress_frame3, text="None Matching Reference Segments")
    label_2_1Txt.grid(row=1, column=0)
    label_2_2Txt = ttk.Label(progress_frame3, text="Count: "+str(len(segments[1])))
    label_2_2Txt.grid(row=1, column=2)

    progress_frame4 = ttk.Frame(segmentMatchReportWindow)
    progress_frame4.pack(pady=20)
    label_2_1Txt = ttk.Label(progress_frame4, text="None Matching Defect Segments")
    label_2_1Txt.grid(row=1, column=0)
    label_2_2Txt = ttk.Label(progress_frame4, text="Count: "+str(len(segments[2])))
    label_2_2Txt.grid(row=1, column=2)

    progress_frame5 = ttk.Frame(segmentMatchReportWindow)
    progress_frame5.pack(pady=20)
    label_2_1Txt = ttk.Label(progress_frame5, text="Matching Conflict Refference Segments")
    label_2_1Txt.grid(row=1, column=0)
    label_2_2Txt = ttk.Label(progress_frame5, text="Count: "+str(len(segments[3])))
    label_2_2Txt.grid(row=1, column=2)

    progress_frame6 = ttk.Frame(segmentMatchReportWindow)
    progress_frame6.pack(pady=20)
    label_2_1Txt = ttk.Label(progress_frame6, text="Matching Conflict Defect Segments")
    label_2_1Txt.grid(row=1, column=0)
    label_2_2Txt = ttk.Label(progress_frame6, text="Count: "+str(len(segments[4])))
    label_2_2Txt.grid(row=1, column=2)

def removeBackground_ref():
    global ref_image
    global ref_img_label
    global imageLabel_TexSample

    image_ref_path = imageLabel_ref.cget("text")
    image_test_path = ""
    image_tex_sample_path = imageLabel_TexSample.cget("text")

    image_ref_name = br.BRModule().splitFileNames(image_ref_path)
    image_tex_sample_name = br.BRModule().splitFileNames(image_tex_sample_path)
    image_tex_sample_nameExeptExt = os.path.splitext(image_tex_sample_name)[0]

    if (image_ref_path == ''):
        messagebox.showerror("Invalid path", "Invalid reference image path. Please make a valid selection!")
        return

    if '_tex_' in image_ref_name and image_tex_sample_path == '':
        messagebox.showerror("Invalid path", "Invalid textured sample image path. Please make a valid selection!")
        return

    if '_tex_' in image_ref_name and image_tex_sample_path != '' and not image_tex_sample_nameExeptExt in image_ref_name:
        messagebox.showerror("Invalid Selection", "Please select a relavant textured sample image!")
        return

    output_path = br.BRModule().run(image_ref_path,image_test_path,image_tex_sample_path,"ref")

    imageLabel_ref.config(text=output_path)
    if(output_path != ''):
        img = Image.open(output_path)
        img = img.resize(resizeVals,Image.ANTIALIAS)
        ref_image = ImageTk.PhotoImage(img )
        ref_img_label.config(image = ref_image)
        messagebox.showinfo("Background remove Success", "Successfully remove the background of reference image!")


def removeBackground_test():
    global test_image
    global test_img_label
    global imageLabel_TexSample

    image_ref_path = imageLabel_ref.cget("text")
    image_test_path = imageLabel_test.cget("text")
    image_tex_sample_path = imageLabel_TexSample.cget("text")

    image_ref_name = br.BRModule().splitFileNames(image_ref_path)
    image_test_name = br.BRModule().splitFileNames(image_test_path)
    image_ref_nameExeptExt = os.path.splitext(image_ref_name)[0]
    image_tex_sample_name = br.BRModule().splitFileNames(image_tex_sample_path)
    image_tex_sample_nameExeptExt = os.path.splitext(image_tex_sample_name)[0]

    if (image_test_path == ''):
        messagebox.showerror("Invalid path", "Invalid test image path. Please make a valid selection!")
        return

    if (image_ref_path == ''):
        messagebox.showerror("Invalid path", "Invalid reference image path. Please make a valid selection!")
        return

    if not (image_ref_nameExeptExt in image_test_name):
        messagebox.showerror("Invalid Selection", "Please select a relavant reference image!")
        return

    if '_tex_' in image_test_name and image_tex_sample_path == '':
        messagebox.showerror("Invalid path", "Invalid textured sample image path. Please make a valid selection!")
        return

    if '_tex_' in image_test_name and image_tex_sample_path != '' and not image_tex_sample_nameExeptExt in image_test_name:
        messagebox.showerror("Invalid Selection", "Please select a relavant textured sample image!")
        return

    output_path = br.BRModule().run(image_ref_path,image_test_path,image_tex_sample_path,"test")

    if (output_path != ''):
        img = Image.open(output_path)
        img = img.resize(resizeVals, Image.ANTIALIAS)
        test_image = ImageTk.PhotoImage(img)
        test_img_label.config(image=test_image)
        messagebox.showinfo("Background remove Success", "Successfully remove the background of test image!")

#QA Module UI

def detectRefFeatures():
    # qa.check_artwork_position(ref_artwork_loc, ref_or_cloth_loc, test_artwork_mask_loc, test_or_cloth_loc, ref_or_cloth_loc, test_or_cloth_loc)
    #------------------------------------------------------------
    global mr_file_list
    global no_of_matching_ref_segs

    # matching_ref
    if os.path.exists(matching_ref_loc):
         mr_file_list = os.listdir(matching_ref_loc)
         no_of_matching_ref_segs = len(mr_file_list)
    else:
         mr_file_list = []
         no_of_matching_ref_segs = 0
    #------------------------------------------------------------
    global mt_file_list
    global no_of_matching_test_segs

    # matching_test
    mt_file_list = os.listdir(matching_test_loc)
    no_of_matching_test_segs = len(mt_file_list)

    #------------------------------------------------------------

    global nmr_file_list
    global no_of_nonmatching_ref_segs

    # nonmatching_ref
    if os.path.exists(nonmatching_ref_loc):
        nmr_file_list = os.listdir(nonmatching_ref_loc)
        no_of_nonmatching_ref_segs = len(nmr_file_list)
    else:
        nmr_file_list = []
        no_of_nonmatching_ref_segs = 0
    #------------------------------------------------------------

    global nmt_file_list
    global no_of_nonmatching_test_segs

    # non_matching_test
    if os.path.exists(nonmatching_test_loc):
        nmt_file_list = os.listdir(nonmatching_test_loc)
        no_of_nonmatching_test_segs = len(nmt_file_list)
    else:
        nmt_file_list = []
        no_of_nonmatching_test_segs = 0

    #-------------------------------------------------------------

    global nmr_conflict_file_list
    global no_of_ref_conflict_segs

    # #non_matching_ref_conflict
    if os.path.exists(nonmatching_ref_conflict):
        nmr_conflict_file_list = os.listdir(nonmatching_ref_conflict)
        no_of_ref_conflict_segs = len(nmr_conflict_file_list)
    else:
        nmr_conflict_file_list = []
        no_of_ref_conflict_segs = 0

    #--------------------------------------------------------------

    global nmt_conflict_file_list
    global no_of_test_conflict_segs

    #non_matching_test_conflict
    if os.path.exists(nonmatching_test_conflict):
        nmt_conflict_file_list = os.listdir(nonmatching_test_conflict)
        no_of_test_conflict_segs = len(nmt_conflict_file_list)
    else:
        nmt_conflict_file_list = []
        no_of_test_conflict_segs = 0
    #---------------------------------------------------------------

    global ref_features
    global ref_thresholded_segs
    global ref_dimensions
    global ref_segs
    ref_features, ref_thresholded_segs, ref_dimensions, ref_segs = qa.detect_features(no_of_matching_ref_segs, 1, matching_ref_loc, matching_test_loc, ref_or_cloth_loc)
    messagebox.showinfo("Extraction success", "Reference image features extracted successfully!")



def detectAndCompare(ref_features, ref_thresholded_segs, ref_dimensions, ref_segs):
    qa.match_segments(nonmatching_ref_loc, nonmatching_test_loc, matching_ref_loc, matching_test_loc, no_of_nonmatching_ref_segs,
                      no_of_test_conflict_segs, nmr_file_list, no_of_nonmatching_test_segs, nmt_file_list,
                   ref_or_cloth_loc, no_of_ref_conflict_segs, nmr_conflict_file_list, nmt_conflict_file_list, nonmatching_ref_conflict,
                      nonmatching_test_conflict, matching_ref_loc, matching_test_loc, ref_artwork_loc, test_artwork_loc)

    shape_def, size_def, placement_def, rotation_def, color_def, minmax_def = qa.detect_and_compare_matching_segments(
        no_of_matching_test_segs, ref_features, 1, ref_thresholded_segs, ref_dimensions, ref_segs,
        no_of_matching_test_segs, matching_test_loc, test_or_cloth_loc, test_artwork_loc)

    qa.display_arr(shape_def, "Shape")
    qa.display_arr(size_def, "Size")
    qa.display_arr(placement_def, "Placement")
    qa.display_arr(rotation_def, "Rotation")
    qa.display_arr(color_def, "Color")
    qa.display_arr(minmax_def, "Minima Maxima")

    # messagebox.showinfo("Comparison Success!", "Comparison complete")

    newWindow = tk.Toplevel(root)
    newWindow.title("Defect Report")
    newWindow.geometry("1500x1200")
    #-----------------------------------------------------------------------
    frame9 = ttk.Frame(newWindow) #main frame
    frame9.pack( fill=tk.BOTH,expand = 1)

    my_canvas = tk.Canvas(frame9) #canvas
    my_canvas.pack(side = tk.LEFT, fill=tk.BOTH, expand=1)

    my_scroll_bar = ttk.Scrollbar(frame9, orient=tk.VERTICAL, command=my_canvas.yview())
    my_scroll_bar.pack(side=tk.RIGHT, fill=tk.Y)

    my_canvas.configure(yscrollcommand = my_scroll_bar.set)
    my_canvas.bind('<Configure>', lambda e: my_canvas.configure(scrollregion = my_canvas.bbox("all")))

    second_frame = ttk.Frame(my_canvas)
    my_canvas.create_window((0,0), window=second_frame, anchor = "nw")

    #-----------------------------------------------------------------------

    # detect_ref_features_button = ttk.Button(second_frame, text="Detect Reference Features",
    #                                         command=lambda: detectRefFeatures())
    # detect_ref_features_button.grid(row=0, column=0, padx=150)
    #
    # detectAndCompare_button = ttk.Button(second_frame, text="Detect and Compare Features",
    #                                      command=lambda: detectAndCompare(ref_features, ref_thresholded_segs,
    #                                                                       ref_dimensions, ref_segs))
    # detectAndCompare_button.grid(row=0, column=1, padx=150)


    refimglist = os.listdir("./Assets/BR_Module/Output/ref/isolated_artwork/")
    print(refimglist)
    # if refimglist != []:

    ref_artwork_img = cv2.imread("./Assets/BR_Module/Output/ref/isolated_artwork/" + refimglist[0])
    img_dims = ref_artwork_img.shape
    ref_artwork_img_saved = cv2.imwrite("./Assets/QA_Module/Output/ref_artwork/ref.jpg",cv2.resize(ref_artwork_img[int(img_dims[0]/10):int(img_dims[0]*9/10), 0:img_dims[1]],(750,800)))

    ref_artwork = Image.open("./Assets/QA_Module/Output/ref_artwork/ref.jpg")
    ref_artwork = ref_artwork.resize((750,800),Image.ANTIALIAS)
    ref_img_comp = ImageTk.PhotoImage(ref_artwork)
    comp_ref_label = ttk.Label(second_frame, background = 'white', image = ref_img_comp)
    comp_ref_label.image = ref_img_comp
    comp_ref_label.grid(row = 0, column = 0)

    testimglist = os.listdir("./Assets/QA_Module/Output/Rotation/")
    print(testimglist)
    # if testimglist != []:
    marked_artwork = Image.open("./Assets/QA_Module/Output/Rotation/" + testimglist[0])
    marked_img_comp = ImageTk.PhotoImage(marked_artwork)
    comp_test_label = ttk.Label(second_frame, background='white', image=marked_img_comp)
    comp_test_label.image = marked_img_comp
    comp_test_label.grid(row=0, column=1)

    # third_frame = ttk.Frame(second_frame)
    # third_frame.pack(pady = 5)

    r = tk.StringVar()

    rb1 = tk.Radiobutton(second_frame, text="Shape", variable = r, value="shape").pack()
    rb1.grid(row = 1, column = 0)
    rb2 = tk.Radiobutton(second_frame, text="Size", variable=r, value="shape").ppack()
    rb2.grid(row=1, column=1)






##### ref image UI
frame_0 = ttk.Frame(root)
frame_0.pack(pady = 20)

ref_topic_label = ttk.Label(frame_0, text = 'Reference Image' ,font=("Calibrir", 12, 'bold'))
ref_topic_label.grid(row=0, column=0, )

frame1 = ttk.Frame(frame_0)
frame1.grid(row=1, column=0)

imageLabelTxt = ttk.Label(frame1, text='Select Image:  ')
imageLabelTxt.grid(row=0, column=0)

label_frame_1 = tk.Frame(frame1, width=200, height=20, bg="white")
label_frame_1.pack_propagate(0)  # Stops child widgets of label_frame from resizing it

imageLabel_ref = ttk.Label(label_frame_1, background='white', text='')
imageLabel_ref.pack(side='left')
label_frame_1.grid(row=0, column=1)

browse_image_button = ttk.Button(frame1, text="Browse", command=ImageBrowser_ref)
browse_image_button.grid(row=0, column=2)


ref_img_label = ttk.Label(frame1, background='white' ,image = ref_image)
ref_img_label.grid(row=2, column=1, )

frame1_1 = ttk.Frame(frame1)
frame1_1.grid(row=3, column=1)

ref_back_rmv_button = ttk.Button(frame1_1, text="Remove Background", command=removeBackground_ref)
ref_back_rmv_button.pack (pady = 5)

frame1_2 = ttk.Frame(frame1)
frame1_2.grid(row=4, column=1)

segment_ref_image_button = ttk.Button(frame1_2, text="Segment Image", command=segmentImage_ref)
segment_ref_image_button.pack (pady = 5)


frame_seperator = ttk.Frame(frame_0, width=30)
frame_seperator.grid(row=0, column=1)


##### textured sample UI

text_sample_topic_label = ttk.Label(frame_0, text = 'Textured Sample(Only Textured)',font=("Calibrir", 12, 'bold') )
text_sample_topic_label.grid(row=0, column=2, )

frame2 = ttk.Frame(frame_0)
frame2.grid(row=1, column=2)

imageLabelTxt = ttk.Label(frame2, text='Select Image:  ')
imageLabelTxt.grid(row=0, column=0)

label_frame_2 = tk.Frame(frame2, width=200, height=20, bg="white")
label_frame_2.pack_propagate(0)  # Stops child widgets of label_frame from resizing it

imageLabel_TexSample = ttk.Label(label_frame_2, background='white', text='')
imageLabel_TexSample.pack(side='left')
label_frame_2.grid(row=0, column=1)

browse_image_button = ttk.Button(frame2, text="Browse", command=ImageBrowser_TexSample)
browse_image_button.grid(row=0, column=2)

tex_sample_img_label = ttk.Label(frame2, background='white' ,image = tex_sample_image)
tex_sample_img_label.grid(row=2, column=1)

frame_seperator = ttk.Frame(frame2, height =70)
frame_seperator.grid(row=3, column=1)

frame_seperator = ttk.Frame(frame_0, width=30)
frame_seperator.grid(row=0, column=3)



##### test image UI

test_topic_label = ttk.Label(frame_0, text = 'Test Image',font=("Calibrir", 12, 'bold') )
test_topic_label.grid(row=0, column=4, )

frame3 = ttk.Frame(frame_0)
frame3.grid(row=1, column=4)

imageLabelTxt = ttk.Label(frame3, text='Select Image:  ')
imageLabelTxt.grid(row=0, column=0)

label_frame_3 = tk.Frame(frame3, width=200, height=20, bg="white")
label_frame_3.pack_propagate(0)  # Stops child widgets of label_frame from resizing it

imageLabel_test = ttk.Label(label_frame_3, background='white', text='')
imageLabel_test.pack(side='left')
label_frame_3.grid(row=0, column=1)

browse_image_button = ttk.Button(frame3, text="Browse", command=ImageBrowser_test)
browse_image_button.grid(row=0, column=2)


test_img_label = ttk.Label(frame3, background='white' ,image = test_image)
test_img_label.grid(row=2, column=1)

frame3_1 = ttk.Frame(frame3)
frame3_1.grid(row=3, column=1)

test_back_rmv_button = ttk.Button(frame3_1, text="Remove Background", command=removeBackground_test)
test_back_rmv_button.pack (pady = 5)

frame3_2 = ttk.Frame(frame3)
frame3_2.grid(row=4, column=1)

segment_test_image_button = ttk.Button(frame3_2, text="Segment Image", command=segmentImage_test)
segment_test_image_button.pack (pady = 5)





##### refference segments UI

frame4 = ttk.Frame(root)
frame4.pack(pady = 5)

referenceFeaturesTxt = ttk.Label(frame4, text='Reference Features: ')
referenceFeaturesTxt.grid(row=0, column=0)

label_frame_4 = tk.Frame(frame4, width=400, height=20, bg="white")
label_frame_4.pack_propagate(0)  # Stops child widgets of label_frame from resizing it

referenceFeaturesLabel = ttk.Label(label_frame_4, background='white', text='')
referenceFeaturesLabel.pack(side='left')
label_frame_4.grid(row=0, column=1)

browse_reference_features_button = ttk.Button(frame4, text="Browse", command=lambda: featuresBrowser('reference'))
browse_reference_features_button.grid(row=0, column=2)

##### defect segments UI
frame5 = ttk.Frame(root)
frame5.pack(pady = 5)

defectFeaturesTxt = ttk.Label(frame5, text='Defect Features:      ')
defectFeaturesTxt.grid(row=0, column=0)

label_frame_5 = tk.Frame(frame5, width=400, height=20, bg="white")
label_frame_5.pack_propagate(0)  # Stops child widgets of label_frame from resizing it

defectFeaturesLabel = ttk.Label(label_frame_5, background='white', text='')
defectFeaturesLabel.pack(side='left')
label_frame_5.grid(row=0, column=1)

browse_defect_features_button = ttk.Button(frame5, text="Browse", command=lambda: featuresBrowser('defect'))
browse_defect_features_button.grid(row=0, column=2)

frame6 = ttk.Frame(root)
frame6.pack(pady = 5)


browse_defect_features_button = ttk.Button(frame6, text="Match Segments", command=lambda: matchSegments())
browse_defect_features_button.grid(row=0, column=1)

frame6 = ttk.Frame(root)
frame6.pack(pady=5)

detect_ref_features_button = ttk.Button(frame6, text="Detect Reference Features",
                                        command=lambda: detectRefFeatures())
detect_ref_features_button.grid(row=0, column=0, padx = 150)


detectAndCompare_button = ttk.Button(frame6, text="Detect and Compare Features", command=lambda: detectAndCompare(ref_features, ref_thresholded_segs, ref_dimensions, ref_segs))
detectAndCompare_button.grid(row=0, column=1, padx = 150)

root.mainloop()