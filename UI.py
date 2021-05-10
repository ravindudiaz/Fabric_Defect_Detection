import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog
from tkinter import messagebox
import os
import Segmentation
import segment_matching

root = tk.Tk()
root.geometry('800x600')
root.title('Defect Detector')
root.configure(background='white')
s1 = ttk.Style()
segmentMatchReportWindow = None
selection = 'reference'

def setSelection(text):
    print('selected:')
    print(text)
    global selection
    selection = text



def ImageBrowser():

    filename = filedialog.askopenfilename( title = "Select Refference File", filetypes=[("jpg files","*.jpg"), ("jpeg files","*,jpeg"), ("png files","*.png"), ("all files","*.*")])
    imageLabel.config(text=filename)

# def defectImageBrowser():
#     filename = filedialog.askopenfilename( title = "Select Defect File",filetypes=[("png files","*.png"), ("jpg files","*.jpg"), ("jpeg files","*,jpeg"),("all files","*.*")])
#     defectLabel.config(text=filename)

def outPath():
    tempdir = filedialog.askdirectory(parent=root, title='Please select output directory')
    outLabel.config(text=tempdir)

# def segmentReferenceImg():
#     image_path = referenceLabel.cget("text")
#     work_path = outLabel.cget("text")
#     if (image_path == '' or work_path == ''):
#         messagebox.showerror("Invalid path", "Invalid image or output path.Please make a valid selection")
#
#     else:
#         sub_folder_name = 'reference'
#         Segmentation.setFolderNames(image_path, work_path, sub_folder_name)
#         Segmentation.createFoldersReference()
#         Segmentation.doSegmentation()

def segmentImage():
    global selection
    image_path = imageLabel.cget("text")
    work_path = outLabel.cget("text")
    kvalue = kEntry.get()
    print(kvalue )
    try:
        int(kvalue)+1
    except:
        messagebox.showerror("Invalid value", "Invalid number of clusters")
        return

    print(selection)
    if (image_path == '' or work_path == ''):
        messagebox.showerror("Invalid path", "Invalid image or output path.Please make a valid selection")

    else:
        if selection == 'reference':
            sub_folder_name = 'reference'
            Segmentation.setFolderNames(image_path, work_path, sub_folder_name)
            Segmentation.createFoldersReference()
            status,csv_path = Segmentation.doSegmentation(int(kvalue))
            if(status):
                messagebox.showinfo("Segmentation Success!", "Image segmented successfully")
                referenceFeaturesLabel.config(text=csv_path )
            else:
                messagebox.showerror("Segmentation Failed", "Something went wrong while segmenting the image")

        if selection == 'defect':
            sub_folder_name = 'defect_1'
            Segmentation.setFolderNames(image_path, work_path, sub_folder_name)
            Segmentation.createFolders(work_path)
            status,csv_path = Segmentation.doSegmentation(int(kvalue))
            if(status):
                messagebox.showinfo("Segmentation Success!", "Image segmented successfully")
                defectFeaturesLabel.config(text=csv_path )
            else:
                messagebox.showerror("Segmentation Failed", "Something went wrong while segmenting the image")

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
    # label_2_2Txt = ttk.Label(progress_frame2, text=str(segments[0]))
    # label_2_2Txt.grid(row=2, column=1)

    progress_frame3 = ttk.Frame(segmentMatchReportWindow)
    progress_frame3.pack(pady=20)
    label_2_1Txt = ttk.Label(progress_frame3, text="None Matching Reference Segments")
    label_2_1Txt.grid(row=1, column=0)
    label_2_2Txt = ttk.Label(progress_frame3, text="Count: "+str(len(segments[1])))
    label_2_2Txt.grid(row=1, column=2)
    # label_2_2Txt = ttk.Label(progress_frame3, text=str(segments[1]))
    # label_2_2Txt.grid(row=2, column=1)

    progress_frame4 = ttk.Frame(segmentMatchReportWindow)
    progress_frame4.pack(pady=20)
    label_2_1Txt = ttk.Label(progress_frame4, text="None Matching Defect Segments")
    label_2_1Txt.grid(row=1, column=0)
    label_2_2Txt = ttk.Label(progress_frame4, text="Count: "+str(len(segments[2])))
    label_2_2Txt.grid(row=1, column=2)
    # label_2_2Txt = ttk.Label(progress_frame4, text=str(segments[2]))
    # label_2_2Txt.grid(row=2, column=1)

    progress_frame5 = ttk.Frame(segmentMatchReportWindow)
    progress_frame5.pack(pady=20)
    label_2_1Txt = ttk.Label(progress_frame5, text="Matching Conflict Segments")
    label_2_1Txt.grid(row=1, column=0)
    label_2_2Txt = ttk.Label(progress_frame5, text="Count: "+str(len(segments[3])))
    label_2_2Txt.grid(row=1, column=2)
    # label_2_2Txt = ttk.Label(progress_frame4, text=str(segments[2]))
    # label_2_2Txt.grid(row=2, column=1)

##### out path UI
frame3 = ttk.Frame(root)
frame3.pack(pady = 20)

outLabelTxt = ttk.Label(frame3, text='Output folder:        ')
outLabelTxt.grid(row=0, column=0)

label_frame_3 = tk.Frame(frame3, width=400, height=20, bg="white")
label_frame_3.pack_propagate(0)  # Stops child widgets of label_frame from resizing it

outLabel = ttk.Label(label_frame_3, background='white', text='')
outLabel.pack(side='left')
label_frame_3.grid(row=0, column=1)

browse_out_button = ttk.Button(frame3, text="Browse", command=outPath)
browse_out_button.grid(row=0, column=2)

##### reference image UI

# referenceLabelTxt = ttk.Label(frame1, text='Reference image:  ')
# referenceLabelTxt.grid(row=0, column=0)
#
# label_frame_1 = tk.Frame(frame1, width=400, height=20, bg="white")
# label_frame_1.pack_propagate(0)  # Stops child widgets of label_frame from resizing it
#
# referenceLabel = ttk.Label(label_frame_1, background='white', text='')
# referenceLabel.pack(side='left')
# label_frame_1.grid(row=0, column=1)
#
# browse_source_button = ttk.Button(frame1, text="Browse", command=lambda: ImageBrowser('reference'))
# browse_source_button.grid(row=0, column=2)
#
# frame1_2 = ttk.Frame(frame1)
# frame1_2.grid(row=1, column=1)
#
# segment_ref_button = ttk.Button(frame1_2, text="Segment Refference Image", command=lambda: segmentImage('reference'))
# segment_ref_button.pack (pady = 5)

##### image UI

frame2 = ttk.Frame(root)
frame2.pack(pady = 20)

imageLabelTxt = ttk.Label(frame2, text='Select Image:        ')
imageLabelTxt.grid(row=0, column=0)

label_frame_2 = tk.Frame(frame2, width=400, height=20, bg="white")
label_frame_2.pack_propagate(0)  # Stops child widgets of label_frame from resizing it

imageLabel = ttk.Label(label_frame_2, background='white', text='')
imageLabel.pack(side='left')
label_frame_2.grid(row=0, column=1)

browse_image_button = ttk.Button(frame2, text="Browse", command=ImageBrowser)
browse_image_button.grid(row=0, column=2)

frame2_1 = ttk.Frame(frame2 )
frame2_1.grid(row=2, column=1)

optionLabelTxt = ttk.Label(frame2_1, text='Image Type:  ')
optionLabelTxt.grid(row=0, column=0)

R1 = ttk.Radiobutton(frame2_1, text="Reference", variable=selection, value='reference', command = lambda : setSelection('reference'))
R1.grid(row=0, column=1)
R2 = ttk.Radiobutton(frame2_1, text="Defect", variable=selection, value='defect', command = lambda: setSelection('defect'))
R2.grid(row=0, column=2)

frame2_3 = ttk.Frame(frame2)
frame2_3.grid(row=4, column=1)

kLabelTxt = ttk.Label(frame2_3, text='Expected color clusters:  ')
kLabelTxt.grid(row=0, column=0)

kEntry = ttk.Entry(frame2_3)
kEntry.grid(row=0, column=1)

frame2_2 = ttk.Frame(frame2)
frame2_2.grid(row=5, column=1)

segment_image_button = ttk.Button(frame2_2, text="Segment Image", command=segmentImage)
segment_image_button.pack (pady = 5)

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