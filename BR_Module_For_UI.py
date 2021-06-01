import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
from PIL import Image,ImageFilter
from os import path


class BRModule():

    rect = (0,0,1,1)
    resizeMark = 1800
    resizerVal = .3
    
    def removeOuterBackground(self,imgPath,saveFolder,type):

        editedFileName = imgPath
        filename = self.splitFileNames(imgPath)
        img = cv.imread(cv.samples.findFile(editedFileName))
            
        width = img.shape[1]

        if width > self.resizeMark :
            self.resizerVal = self.resizeMark/width

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
        self.generateFabricMask(self.img_copy,filename,type)

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

        return outputName


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

        return savePath

            

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

    

    def generateFabricMask(self,image,name,type):
        
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

        nameWithMask = ""

        if type == "ref":
            nameWithMask = "Assets/BR_Module/Output/fabric_masks_ref/"+name

        if type == "test":
            nameWithMask = "Assets/BR_Module/Output/fabric_masks_test/"+name
              
        cv.imwrite(nameWithMask, newmask)



    def generateUniformArtWorkMask(self,filename,type):

        editedFileName = ""
        if type == "ref":
            editedFileName = 'Assets/BR_Module/Output/edge_ref/'+ filename

        if type == "test":
            editedFileName = 'Assets/BR_Module/Output/edge_test/'+ filename


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


        self.mask = np.zeros(img.shape[:2],np.uint8)
        newmask = image_binary

        self.mask[newmask == 0] = 0
        self.mask[newmask == 255] = 1

        maskName = ""
        if type == "ref":
            maskName = "Assets/BR_Module/Output/artwork_masks_ref/"+filename

        if type == "test":
            maskName = "Assets/BR_Module/Output/artwork_masks_test/"+filename

        cv.imwrite(maskName, image_binary)



    def generateUniformFabricEdge(self,filePath,saveFolder):

        filename = self.splitFileNames(filePath)

        if 'uni_' in filename:

            img = cv.imread(cv.samples.findFile(filePath))
            self.getOptimalThresholdVal(img)

            gaussian = cv.GaussianBlur(img,(3,3),cv.BORDER_DEFAULT)
            edges = cv.Canny(gaussian,self.minThresholdVal,self.maxThresholdVal)

            nameWithEdge = saveFolder+"/"+filename        

            cv.imwrite(nameWithEdge, edges)


    
    def isolateFabArtwork(self,filePath,type,saveFolder):

        filename = self.splitFileNames(filePath)

        if 'uni_' in filename or '_tex_' in filename:

            editedFileName = filePath
            img = cv.imread(cv.samples.findFile(editedFileName))

            img = img.copy()
            self.img = img
            self.img_copy = self.img.copy()

            self.output = np.zeros(self.img.shape, np.uint8)

            self.getOptimalThresholdVal(self.img_copy)


            if 'uni_' in filename:
                self.generateUniformArtWorkMask(filename,type)

            if '_tex_' in filename:
                self.generateteTexturedArtworkMask(filename,type)

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

            return outputName
            


    def registratedMachingFiles(self,refOutputFilePath,testOutputFilePath,saveFolder):

        outputFileName = self.splitFileNames(testOutputFilePath)

        imgRef = cv.imread(cv.samples.findFile(refOutputFilePath))

        img = cv.imread(cv.samples.findFile(testOutputFilePath))

        return self.generateRegistratedImage(imgRef,img,outputFileName,saveFolder)

    
    
    def deleteGeneratedFiles(self,folder):

        for filename in os.listdir(folder):

            editedFileName = folder +'/'+ filename
            os.remove(editedFileName)

    def splitFileNames(self,fileName):
        fileName = fileName.split('/')[-1]
        return fileName


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
                    outputImage.save(saveFolder+"/"+filename)



    def sharpTexturedArtworkDraft(self,folder):

        for filename in os.listdir(folder):

            editedFileName = folder +'/'+ filename
            
            image = cv.imread(editedFileName)

            kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
            erode = cv.erode(image, kernel, iterations=1)


            kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
            opening = cv.morphologyEx(erode, cv.MORPH_OPEN, kernel)

            cv.imwrite(editedFileName, opening)

            image = Image.open(editedFileName)
            image = image.filter(ImageFilter.MedianFilter(size=13))
            image.save(editedFileName)



    def generateteTexturedArtworkMask(self,filename,type):

        editedFileName = ''

        if type == "ref":
            editedFileName = 'Assets/BR_Module/Output/artworks_drafts_ref/'+ filename

        if type == "test":
            editedFileName = 'Assets/BR_Module/Output/artworks_drafts_test/'+ filename

       
        img = np.array(Image.open(editedFileName).convert('L'))

        img = cv.normalize(img, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
        res, img = cv.threshold(img, 64, 255, cv.THRESH_BINARY)

        cv.floodFill(img, None, (0,0), 0)
   
        self.mask = np.zeros(img.shape[:2],np.uint8)
        newmask = img

        self.mask[newmask == 0] = 0
        self.mask[newmask == 255] = 1

        maskName = ""
        if type == "ref":
            maskName = "Assets/BR_Module/Output/artwork_masks_ref/"+filename

        if type == "test":
            maskName = "Assets/BR_Module/Output/artwork_masks_test/"+filename

        Image.fromarray(img).save(maskName)

    def generateOutputPath(self,outerRemovedOutPath,type):
        
        absPath = os.path.abspath(outerRemovedOutPath)

        outputPath = absPath.replace('\\', '/')

        return outputPath
        
        # fileName = outerRemovedOutPath.split('/')[-1]
        # outputPath = ""
        # if type == "ref":
        #     outputPath = os.getcwd()+"Assets/BR_Module/Output/artworks_ref"+fileName

        # if type == "test":
        #     outputPath = os.getcwd()+"Assets/BR_Module/Output/artworks_test"+fileName

        # return outputPath




    def run(self,imgRefPath,imgTestPath,texSamplesPath,type):

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

        #To store final output,include fabric artworks(Reference/Test)
        fabricMasksRef = 'Assets/BR_Module/Output/fabric_masks_ref'
        fabricMasksTest = 'Assets/BR_Module/Output/fabric_masks_test'

        #creating output folders if not exists
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

            if not path.exists(fabricMasksRef):
                os.makedirs(fabricMasksRef)

            if not path.exists(fabricMasksTest):
                os.makedirs(fabricMasksTest)

        except:
            import traceback
            traceback.print_exc()

        # self.deleteGeneratedFiles(outerRemReferenceImages)
        # self.deleteGeneratedFiles(outerRemTestImages)
        # self.deleteGeneratedFiles(registratedTestImages)
        # self.deleteGeneratedFiles(edgeReferenceImages)
        # self.deleteGeneratedFiles(edgeTestImages)
        # self.deleteGeneratedFiles(artworksDraftsRef)
        # self.deleteGeneratedFiles(artworksDraftsTest)
        # self.deleteGeneratedFiles(artworkMasksReferenceImages)
        # self.deleteGeneratedFiles(artworkMasksTestImages)
        # self.deleteGeneratedFiles(artworksReferenceImages)
        # self.deleteGeneratedFiles(artworksTestImages)
        # self.deleteGeneratedFiles(fabricMasksRef)
        # self.deleteGeneratedFiles(fabricMasksTest)

        refOuterRemovedFilePath =""
        testOuterRemovedFilePath =""

        if type == "ref":

            refOuterRemovedFilePath = self.removeOuterBackground(imgRefPath,outerRemReferenceImages,"ref")

            self.generateUniformFabricEdge(refOuterRemovedFilePath,edgeReferenceImages)

            # self.generateteTexturedArtworkDarft(texSamplesPath,refOutputFilePath,artworksDraftsRef)

            # self.sharpTexturedArtworkDraft(artworksDraftsRef)

            refOuterRemovedFilePath = self.isolateFabArtwork(refOuterRemovedFilePath,'ref',artworksReferenceImages)

            return self.generateOutputPath(refOuterRemovedFilePath,"ref")

        if type == "test":

            refOuterRemovedFilePath = self.removeOuterBackground(imgRefPath,outerRemReferenceImages,"ref")
            testOuterRemovedFilePath = self.removeOuterBackground(imgTestPath,outerRemTestImages,"test")
            
            testOuterRemovedFilePath = self.registratedMachingFiles(refOuterRemovedFilePath,testOuterRemovedFilePath,registratedTestImages)

            self.generateUniformFabricEdge(testOuterRemovedFilePath,edgeTestImages)

            # self.generateteTexturedArtworkDarft(texSamples,registratedTestImages,artworksDraftsTest)

            # self.sharpTexturedArtworkDraft(artworksDraftsTest)

            testOuterRemovedFilePath = self.isolateFabArtwork(testOuterRemovedFilePath,'test',artworksTestImages)

            return self.generateOutputPath(testOuterRemovedFilePath,"test")