import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog
from tkinter import messagebox
import os
import Segmentation_v2 as Segmentation
import segment_matching
from PIL import ImageTk, Image

root = tk.Tk()
root.geometry('1400x1000')
root.title('Defect Detector')
root.configure(background='white')
s1 = ttk.Style()

segmentMatchReportWindow = None


ref_img_thumb = 'Assets/src/ref_img_thumb.jpg'
test_img_thumb = 'Assets/src/test_img_thumb.jpg'

ref_image =ImageTk.PhotoImage(Image.open(ref_img_thumb))

test_image = ImageTk.PhotoImage(Image.open(test_img_thumb))




def ImageBrowser_ref():
    global ref_image
    filename = filedialog.askopenfilename( title = "Select Refference File", filetypes=[("jpg files","*.jpg"), ("jpeg files","*,jpeg"), ("png files","*.png"), ("all files","*.*")])
    imageLabel_ref.config(text=filename)
    if(filename != ''):
        img = Image.open(filename)
        img = img.resize((400,400),Image.ANTIALIAS)
        ref_image = ImageTk.PhotoImage(img )
        ref_img_label.config(image = ref_image)
    else:
        ref_image = ImageTk.PhotoImage(Image.open(ref_img_thumb))
        ref_img_label.config(image=ref_image)

def ImageBrowser_test():
    global test_image
    filename = filedialog.askopenfilename(title="Select Refference File",filetypes=[("jpg files", "*.jpg"), ("jpeg files", "*,jpeg"),("png files", "*.png"), ("all files", "*.*")])
    imageLabel_test.config(text=filename)
    if (filename != ''):
        img = Image.open(filename)
        img = img.resize((400, 400), Image.ANTIALIAS)
        test_image = ImageTk.PhotoImage(img)
        test_img_label.config(image=test_image)
    else:
        test_image = ImageTk.PhotoImage(Image.open(test_img_thumb))
        test_img_label.config(image=test_image)

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
    image_path = imageLabel_ref.cget("text")
    print(image_path)
    print('removing background ref')

def removeBackground_test():
    image_path = imageLabel_test.cget("text")
    print(image_path)
    print('removing background test')


##### ref image UI
frame_0 = ttk.Frame(root)
frame_0.pack(pady = 20)

ref_topic_label = ttk.Label(frame_0, text = 'Refference Image' ,font=("Calibrir", 12, 'bold'))
ref_topic_label.grid(row=0, column=0, )

frame1 = ttk.Frame(frame_0)
frame1.grid(row=1, column=0)

imageLabelTxt = ttk.Label(frame1, text='Select Image:  ')
imageLabelTxt.grid(row=0, column=0)

label_frame_1 = tk.Frame(frame1, width=400, height=20, bg="white")
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


frame_seperator = ttk.Frame(frame_0, width=50)
frame_seperator.grid(row=0, column=1)


##### test image UI

test_topic_label = ttk.Label(frame_0, text = 'Test Image',font=("Calibrir", 12, 'bold') )
test_topic_label.grid(row=0, column=2, )

frame2 = ttk.Frame(frame_0)
frame2.grid(row=1, column=2)

imageLabelTxt = ttk.Label(frame2, text='Select Image:  ')
imageLabelTxt.grid(row=0, column=0)

label_frame_2 = tk.Frame(frame2, width=400, height=20, bg="white")
label_frame_2.pack_propagate(0)  # Stops child widgets of label_frame from resizing it

imageLabel_test = ttk.Label(label_frame_2, background='white', text='')
imageLabel_test.pack(side='left')
label_frame_2.grid(row=0, column=1)

browse_image_button = ttk.Button(frame2, text="Browse", command=ImageBrowser_test)
browse_image_button.grid(row=0, column=2)


test_img_label = ttk.Label(frame2, background='white' ,image = test_image)
test_img_label.grid(row=2, column=1)

frame2_1 = ttk.Frame(frame2)
frame2_1.grid(row=3, column=1)

test_back_rmv_button = ttk.Button(frame2_1, text="Remove Background", command=removeBackground_test)
test_back_rmv_button.pack (pady = 5)

frame2_2 = ttk.Frame(frame2)
frame2_2.grid(row=4, column=1)

segment_test_image_button = ttk.Button(frame2_2, text="Segment Image", command=segmentImage_test)
segment_test_image_button.pack (pady = 5)




##### refference segments UI

frame3 = ttk.Frame(root)
frame3.pack(pady = 20)

referenceFeaturesTxt = ttk.Label(frame3, text='Reference Features: ')
referenceFeaturesTxt.grid(row=0, column=0)

label_frame_3 = tk.Frame(frame3, width=400, height=20, bg="white")
label_frame_3.pack_propagate(0)  # Stops child widgets of label_frame from resizing it

referenceFeaturesLabel = ttk.Label(label_frame_3, background='white', text='')
referenceFeaturesLabel.pack(side='left')
label_frame_3.grid(row=0, column=1)

browse_reference_features_button = ttk.Button(frame3, text="Browse", command=lambda: featuresBrowser('reference'))
browse_reference_features_button.grid(row=0, column=2)

##### defect segments UI
frame4 = ttk.Frame(root)
frame4.pack(pady = 20)

defectFeaturesTxt = ttk.Label(frame4, text='Defect Features:      ')
defectFeaturesTxt.grid(row=0, column=0)

label_frame_4 = tk.Frame(frame4, width=400, height=20, bg="white")
label_frame_4.pack_propagate(0)  # Stops child widgets of label_frame from resizing it

defectFeaturesLabel = ttk.Label(label_frame_4, background='white', text='')
defectFeaturesLabel.pack(side='left')
label_frame_4.grid(row=0, column=1)

browse_defect_features_button = ttk.Button(frame4, text="Browse", command=lambda: featuresBrowser('defect'))
browse_defect_features_button.grid(row=0, column=2)

frame5 = ttk.Frame(root)
frame5.pack(pady = 20)


browse_defect_features_button = ttk.Button(frame5, text="Match Segments", command=lambda: matchSegments())
browse_defect_features_button.grid(row=0, column=1)

root.mainloop()