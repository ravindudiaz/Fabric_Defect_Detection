import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
from PIL import Image
from os import path


class BRModule():

    rect = (0,0,1,1)
    resizeMark = 1500
    resizerVal = .3
    
    def removeOuterBackground(self,folder,saveFolder):

        for filename in os.listdir(folder):

            editedFileName = folder +'/'+ filename
            img = cv.imread(cv.samples.findFile(editedFileName))
            
            height = img.shape[0]
            width = img.shape[1]

            if height > self.resizeMark or width > self.resizeMark :
                img = cv.resize(img,None,fx=self.resizerVal,fy=self.resizerVal)

            # create copy of image
            img = img.copy()
            self.img = img
            self.img_copy = self.img.copy()

            # allocate array for output
            self.output = np.zeros(self.img.shape, np.uint8)

            # get optimal threshold value using OTSU method
            self.getOptimalThresholdVal(self.img_copy)

            # generating mask for grabcut
            self.generateFabricMask(self.img_copy,filename)

            try:
                bgdmodel = np.zeros((1, 65), np.float64)
                fgdmodel = np.zeros((1, 65), np.float64)

                cv.grabCut(self.img_copy, self.mask, self.rect, bgdmodel, fgdmodel, 1, cv.GC_INIT_WITH_MASK)

            except:
                import traceback
                traceback.print_exc()

            mask2 = np.where((self.mask==2)|(self.mask==0),0,1).astype('uint8')
            self.output = self.img_copy*mask2[:,:,np.newaxis]

            outputName = saveFolder+'/'+filename
            cv.imwrite(outputName,self.output)


    def generateRegistratedImage(self,imageRef,imageSample,imageName,folder):

        img1_color = imageSample 
        img2_color = imageRef
 
        img1 = cv.cvtColor(img1_color, cv.COLOR_BGR2GRAY) 
        img2 = cv.cvtColor(img2_color, cv.COLOR_BGR2GRAY) 
        height, width = img2.shape 

        orb_detector = cv.ORB_create(5000) 

        kp1, d1 = orb_detector.detectAndCompute(img1, None) 
        kp2, d2 = orb_detector.detectAndCompute(img2, None) 

        matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck = True) 

        matches = matcher.match(d1, d2) 
        matches.sort(key = lambda x: x.distance) 
        matches = matches[:int(len(matches)*90)]

        no_of_matches = len(matches) 

        p1 = np.zeros((no_of_matches, 2)) 
        p2 = np.zeros((no_of_matches, 2)) 

        for i in range(len(matches)): 
            p1[i, :] = kp1[matches[i].queryIdx].pt 
            p2[i, :] = kp2[matches[i].trainIdx].pt 

        homography, mask = cv.findHomography(p1, p2, cv.RANSAC) 

        transformed_img = cv.warpPerspective(img1_color, 
					homography, (width, height)) 

        savePath = folder+"/"+imageName 
        cv.imwrite(savePath, transformed_img) 

            

    def getOptimalThresholdVal(self,image):

        image = cv.GaussianBlur(image, (5, 5), 0)

        bins_num = 256

        hist, bin_edges = np.histogram(image, bins=bins_num)
        hist = np.divide(hist.ravel(), hist.max())

        bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.

        weight1 = np.cumsum(hist)
        weight2 = np.cumsum(hist[::-1])[::-1]

        mean1 = np.cumsum(hist * bin_mids) / weight1
        mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]

        inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

        index_of_max_val = np.argmax(inter_class_variance)
        
        threshold = bin_mids[:-1][index_of_max_val]

        self.minThresholdVal = threshold
        self.maxThresholdVal = threshold*2

    

    def generateFabricMask(self,image,name):
        
        img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        _,mask = cv.threshold(img,self.minThresholdVal,self.maxThresholdVal,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

        kernal = np.ones((5,5), np.uint8)
        mg = cv.morphologyEx(mask, cv.MORPH_GRADIENT, kernal)
        mg = cv.medianBlur(mg,5)

        gaussian = cv.GaussianBlur(mg,(3,3),cv.BORDER_DEFAULT)
        edges = cv.Canny(gaussian,self.minThresholdVal,self.maxThresholdVal)

        contours, hierarchy = cv.findContours(edges,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)
        
        #first sort the array by area
        sorteddata = sorted(contours, key = cv.contourArea, reverse=True)
        
        image_binary = np.zeros(image.shape[:2],np.uint8)

        for n in range(0,len(sorteddata)):

            if n == 0:
                cv.drawContours(image_binary, [sorteddata[n]],
                     -1, (255, 255, 255), -1)

        self.mask = np.zeros(image.shape[:2],np.uint8)
        newmask = image_binary

        self.mask[newmask == 0] = 0
        self.mask[newmask == 255] = 1

        # nameWithMask = "fabric_masks/Mask_"+name        
        # cv.imwrite(nameWithMask, newmask)



    def generateUniformArtWorkMask(self,folder,saveFolder):

        for filename in os.listdir(folder):

            editedFileName = folder +'/'+ filename
            img = cv.imread(cv.samples.findFile(editedFileName))

            self.getOptimalThresholdVal(img)
        
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            _,mask = cv.threshold(img,self.minThresholdVal,self.maxThresholdVal,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

            # mask = cv.fastNlMeansDenoising(mask,None,10,10,7,21)

            kernal = np.ones((5,5), np.uint8)
            mg = cv.morphologyEx(mask, cv.MORPH_GRADIENT, kernal)
            mg = cv.medianBlur(mg,5)

            gaussian = cv.GaussianBlur(mg,(3,3),cv.BORDER_DEFAULT)
            edges = cv.Canny(gaussian,self.minThresholdVal,self.maxThresholdVal)

            contours, hierarchy = cv.findContours(edges,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)
        
            #first sort the array by area
            sorteddata = sorted(contours, key = cv.contourArea, reverse=True)

            image_binary = np.zeros(img.shape[:2],np.uint8)

            for n in range(0,len(sorteddata)):

                if n > 4:
                    # x, y, w, h = cv.boundingRect(sorteddata[n])
                    cv.drawContours(image_binary, [sorteddata[n]],
                         -1, (255, 255, 255), -1)
                    # cv.rectangle(image_binary, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # mask = np.zeros(img.shape[:2],np.uint8)
            # newmask = image_binary

            # mask[newmask == 0] = 0
            # mask[newmask == 255] = 1

            nameWithMask = saveFolder+"/Mask_"+filename        
            cv.imwrite(nameWithMask, image_binary)



    def generateUniformFabricEdge(self,folder,saveFolder):

        for filename in os.listdir(folder):

            if 'uni_' in filename:

                editedFileName = folder +'/'+ filename

                img = cv.imread(cv.samples.findFile(editedFileName))
                self.getOptimalThresholdVal(img)

                gaussian = cv.GaussianBlur(img,(3,3),cv.BORDER_DEFAULT)
                edges = cv.Canny(gaussian,self.minThresholdVal,self.maxThresholdVal)

                nameWithEdge = saveFolder+"/"+filename        

                cv.imwrite(nameWithEdge, edges)


    
    def isolateUniformFabArtwork(self,folder,maskFolder,saveFolder):

        for filename in os.listdir(folder):

            if 'uni_' in filename or '_tex_' in filename:

                editedFileName = folder +'/'+ filename
                img = cv.imread(cv.samples.findFile(editedFileName))
            
                height = img.shape[0]
                width = img.shape[1]

                if height > self.resizeMark or width > self.resizeMark :
                    img = cv.resize(img,None,fx=self.resizerVal,fy=self.resizerVal)

                # create copy of image
                img = img.copy()
                self.img = img
                self.img_copy = self.img.copy()

                # allocate array for output
                self.output = np.zeros(self.img.shape, np.uint8)

                # get optimal threshold value using OTSU method
                self.getOptimalThresholdVal(self.img_copy)

                # generating mask for grabcut
                self.getUniformArtworkMask(self.img_copy,filename,maskFolder)

                try:
                    bgdmodel = np.zeros((1, 65), np.float64)
                    fgdmodel = np.zeros((1, 65), np.float64)

                    cv.grabCut(self.img_copy, self.mask, self.rect, bgdmodel, fgdmodel, 1, cv.GC_INIT_WITH_MASK)

                except:
                    import traceback
                    traceback.print_exc()

                mask2 = np.where((self.mask==2)|(self.mask==0),0,1).astype('uint8')
                self.output = self.img_copy*mask2[:,:,np.newaxis]

                outputName = saveFolder+'/'+filename
                cv.imwrite(outputName,self.output)



    def getUniformArtworkMask(self,img,filename,folder):

        maskName = "Mask_"+filename

        editedMaskFileName = folder +'/'+ maskName
        maskImg = cv.imread(cv.samples.findFile(editedMaskFileName))
        maskImg = cv.cvtColor(maskImg, cv.COLOR_BGR2GRAY)

        self.mask = np.zeros(img.shape[:2],np.uint8)
        newmask = maskImg

        self.mask[newmask == 0] = 0
        self.mask[newmask == 255] = 1
            


    def registratedMachingFiles(self,refFolder,folder,saveFolder):

        for refFilename in os.listdir(refFolder):

            refFilenameExeptExt = os.path.splitext(refFilename)[0]

            for filename in os.listdir(folder):

                if refFilenameExeptExt in filename:

                    editedRefFileName = refFolder +'/'+ refFilename

                    editedFileName = folder +'/'+ filename

                    imgRef = cv.imread(cv.samples.findFile(editedRefFileName))

                    img = cv.imread(cv.samples.findFile(editedFileName))

                    self.generateRegistratedImage(imgRef,img,filename,saveFolder)

    
    
    def deleteGeneratedFiles(self,folder):

        for filename in os.listdir(folder):

            editedFileName = folder +'/'+ filename
            os.remove(editedFileName)

    

    def generateteTexturedArtworkDarft(self,samplefolder,folder,saveFolder):

        blockSizeRows = 5
        blockSizeColumns = 5
        
        for sampleFilename in os.listdir(samplefolder):

            editedSampleFileName = samplefolder +'/'+ sampleFilename

            sampleImage = cv.imread(editedSampleFileName)
            sampleImageHSV = cv.cvtColor(sampleImage, cv.COLOR_BGR2HSV)
            sampleImageHIST = cv.calcHist([sampleImageHSV], [0,1], None, [180,256], [0,180,0,256])
            cv.normalize(sampleImageHIST, sampleImageHIST, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

            for filename in os.listdir(folder):

                sampleFilenameExeptExt = os.path.splitext(sampleFilename)[0]

                if sampleFilenameExeptExt in filename and '_tex_' in filename:

                    editedFileName = folder +'/'+ filename

                    imageToBackRmv = cv.imread(editedFileName)
                    imageToBackRmv = cv.cvtColor(imageToBackRmv, cv.COLOR_BGR2HSV)

                    outputImage = Image.open(r""+editedFileName)
                    outputImageArr = np.array(outputImage)

                    for row in range(0,imageToBackRmv.shape[0]- blockSizeRows,blockSizeRows):

                        for column in range(0,imageToBackRmv.shape[1]- blockSizeColumns,blockSizeColumns):

                            imageBlock = imageToBackRmv[row:row+blockSizeRows,column:column+blockSizeColumns]
                            imageBlockHIST = cv.calcHist([imageBlock], [0,1], None, [180,256], [0,180,0,256])
                            cv.normalize(imageBlockHIST, imageBlockHIST, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

                            value = cv.compareHist(sampleImageHIST, imageBlockHIST, cv.HISTCMP_CORREL)

                            if value > 0 :
                                outputImageArr[row:row+blockSizeRows,column:column+blockSizeColumns] = (0, 0, 0)

                    outputImage = Image.fromarray(outputImageArr)
                    outputImage.save(saveFolder+"/Mask_"+filename)



    def sharpTexturedArtworkDraft(self,folder):

        for filename in os.listdir(folder):

            editedFileName = folder +'/'+ filename
            
            image = cv.imread(editedFileName)

            kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
            erode = cv.erode(image, kernel, iterations=1)


            kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
            opening = cv.morphologyEx(erode, cv.MORPH_OPEN, kernel)

            cv.imwrite(editedFileName, opening)



    def generateteTexturedArtworkMask(self,folder,saveFolder):

        for filename in os.listdir(folder):

            editedFileName = folder +'/'+ filename
            
            img = np.array(Image.open(editedFileName).convert('L'))

            img = cv.normalize(img, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
            res, img = cv.threshold(img, 64, 255, cv.THRESH_BINARY)

            cv.floodFill(img, None, (0,0), 0)

            saveFileName = saveFolder +'/'+ filename

            Image.fromarray(img).save(saveFileName)



    def run(self):

        print('Start Background Removal Module Execution...')

        # --input folders--

        #Include both of textured and uniform reference fabric images with background
        referenceImages = 'Assets/BR_Module/Input/ref'
        #Include both of textured and uniform test fabric images with background
        testImages = 'Assets/BR_Module/Input/test'
        #Include textured fabric sample images
        texSamples = 'Assets/BR_Module/Input/tex_samples'

        # --output folders--

        #To store outer background removed fabric images(Reference/Test)
        outerRemReferenceImages = 'Assets/BR_Module/Output/outer_removed_ref'
        outerRemTestImages = 'Assets/BR_Module/Output/outer_removed_test'

        #To store registrated test images(Test)
        registratedTestImages = 'Assets/BR_Module/Output/registrated_test'

        #To store uniform artwork edge images(Reference/Test)
        edgeReferenceImages = 'Assets/BR_Module/Output/edge_ref'
        edgeTestImages = 'Assets/BR_Module/Output/edge_test'

        #To store textured artwork drafts(Reference/Test)
        artworksDraftsRef = 'Assets/BR_Module/Output/artworks_drafts_ref'
        artworksDraftsTest = 'Assets/BR_Module/Output/artworks_drafts_test'

        #To store fabric artworks masks(Reference/Test)
        artworkMasksReferenceImages = 'Assets/BR_Module/Output/artwork_masks_ref'
        artworkMasksTestImages = 'Assets/BR_Module/Output/artwork_masks_test'

        #To store final output,include fabric artworks(Reference/Test)
        artworksReferenceImages = 'Assets/BR_Module/Output/artworks_ref'
        artworksTestImages = 'Assets/BR_Module/Output/artworks_test'

        #creating output folders if not exists
        print("Creating directories for output images..")
        try:
            if not path.exists(outerRemReferenceImages):
                os.makedirs(outerRemReferenceImages)

            if not path.exists(outerRemTestImages):
                os.makedirs(outerRemTestImages)

            if not path.exists(registratedTestImages):
                os.makedirs(registratedTestImages)

            if not path.exists(edgeReferenceImages):
                os.makedirs(edgeReferenceImages)

            if not path.exists(edgeTestImages):
                os.makedirs(edgeTestImages)

            if not path.exists(artworksDraftsRef):
                os.makedirs(artworksDraftsRef)

            if not path.exists(artworksDraftsTest):
                os.makedirs(artworksDraftsTest)

            if not path.exists(artworkMasksReferenceImages):
                os.makedirs(artworkMasksReferenceImages)

            if not path.exists(artworkMasksTestImages):
                os.makedirs(artworkMasksTestImages)

            if not path.exists(artworksReferenceImages):
                os.makedirs(artworksReferenceImages)
                
            if not path.exists(artworksTestImages):
                os.makedirs(artworksTestImages)

        except:
            import traceback
            traceback.print_exc()

        #delete created files
        print("Deleting created output images...")
        self.deleteGeneratedFiles(outerRemReferenceImages)
        self.deleteGeneratedFiles(registratedTestImages)
        self.deleteGeneratedFiles(edgeReferenceImages)
        self.deleteGeneratedFiles(edgeTestImages)
        self.deleteGeneratedFiles(artworksDraftsRef)
        self.deleteGeneratedFiles(artworksDraftsTest)
        self.deleteGeneratedFiles(artworkMasksReferenceImages)
        self.deleteGeneratedFiles(artworkMasksTestImages)
        self.deleteGeneratedFiles(artworksReferenceImages)
        self.deleteGeneratedFiles(artworksTestImages)

        print("Removing outer backgrounds...")
        #Remove outer background of reference images
        self.removeOuterBackground(referenceImages,outerRemReferenceImages)
        #Remove outer background of test images
        self.removeOuterBackground(testImages,outerRemTestImages)

        #Registrated test images using reference point
        print("Image Registrating...")
        self.registratedMachingFiles(outerRemReferenceImages,outerRemTestImages,registratedTestImages)

        #Creating uniform fabric edges
        print("Creating uniform fabric edges...")
        self.generateUniformFabricEdge(outerRemReferenceImages,edgeReferenceImages)
        self.generateUniformFabricEdge(registratedTestImages,edgeTestImages)

        #Creating uniform artwork mask
        print("Creating uniform artwork masks...")
        self.generateUniformArtWorkMask(edgeReferenceImages,artworkMasksReferenceImages)
        self.generateUniformArtWorkMask(edgeTestImages,artworkMasksTestImages)

        #Creating textured artwork Darfts..
        print("Creating textured artwork Darfts...")
        self.generateteTexturedArtworkDarft(texSamples,outerRemReferenceImages,artworksDraftsRef)
        self.generateteTexturedArtworkDarft(texSamples,registratedTestImages,artworksDraftsTest)

        #Sharping textured artwork Darfts..
        print("Sharping textured artwork Darfts...")
        self.sharpTexturedArtworkDraft(artworksDraftsRef)
        self.sharpTexturedArtworkDraft(artworksDraftsTest)

        #Creating uniform artwork mask
        print("Creating textured artwork masks...")
        self.generateteTexturedArtworkMask(artworksDraftsRef,artworkMasksReferenceImages)
        self.generateteTexturedArtworkMask(artworksDraftsTest,artworkMasksTestImages)

        #Creating uniform artwork mask
        print("Isolating fabric artworks...")
        self.isolateUniformFabArtwork(outerRemReferenceImages,artworkMasksReferenceImages,artworksReferenceImages)
        self.isolateUniformFabArtwork(registratedTestImages,artworkMasksTestImages,artworksTestImages)

        print('End Background Removal Module Execution...')



BRModule().run()