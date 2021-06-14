import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
from PIL import Image,ImageFilter
from os import path


class BRModule():

    #Global variables

    rect = 0
    resizeMark = 2000
    resizeMarkMask = 500
    resizerVal = 0
    originalWidth = 0
    referenceImageType = 'ref'
    testImageType = 'test'
    uniformCode = 'uni_'
    texturedCode = '_tex_'


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

    
    def removeOuterBackground(self,imgPath,saveFolder,type):

        editedFileName = imgPath
        filename = self.splitFileNames(imgPath)
        img = cv.imread(cv.samples.findFile(editedFileName))
            
        width = img.shape[1]
        self.originalWidth = width

        if width > self.resizeMark :
            self.resizerVal = self.resizeMark/width

            img = cv.resize(img,None,fx=self.resizerVal,fy=self.resizerVal,interpolation=cv.INTER_AREA)

        img = img.copy()
        self.img = img
        self.img_copy = self.img.copy()

        self.output = np.zeros(self.img.shape, np.uint8)

        self.getOptimalThresholdVal(self.img_copy)

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

        if type == self.referenceImageType:
            nameWithMask = self.fabricMasksRef+"/"+name

        if type == self.testImageType:
            nameWithMask = self.fabricMasksTest+"/"+name
              
        cv.imwrite(nameWithMask, newmask)



    def generateUniformArtWorkMask(self,filename,type):

        editedFileName = ""
        maskName = ""

        if type == self.referenceImageType:
            editedFileName = self.edgeReferenceImages+'/'+ filename
            maskName = self.artworkMasksReferenceImages+"/"+filename

        if type == self.testImageType:
            editedFileName = self.edgeTestImages+'/'+ filename
            maskName = self.artworkMasksTestImages+"/"+filename


        img = cv.imread(cv.samples.findFile(editedFileName))

        self.getOptimalThresholdVal(img)
        
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        _,mask = cv.threshold(img,self.minThresholdVal,self.maxThresholdVal,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

        kernal = np.ones((5,5), np.uint8)
        mg = cv.morphologyEx(mask, cv.MORPH_GRADIENT, kernal)
        mg = cv.medianBlur(mg,5)

        gaussian = cv.GaussianBlur(mg,(3,3),cv.BORDER_DEFAULT)
        edges = cv.Canny(gaussian,self.minThresholdVal,self.maxThresholdVal)

        contours, hierarchy = cv.findContours(edges,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)
        
        sorteddata = sorted(contours, key = cv.contourArea, reverse=True)

        image_binary = np.zeros(img.shape[:2],np.uint8)

        for n in range(0,len(sorteddata)):

            if n > 4:
                
                cv.drawContours(image_binary, [sorteddata[n]],-1, (255, 255, 255), -1)


        self.mask = np.zeros(img.shape[:2],np.uint8)
        newmask = image_binary

        self.mask[newmask == 0] = 0
        self.mask[newmask == 255] = 1

        cv.imwrite(maskName, image_binary)



    def generateUniformFabricEdge(self,filePath,saveFolder):

        filename = self.splitFileNames(filePath)

        if self.uniformCode in filename:

            img = cv.imread(cv.samples.findFile(filePath))
            image_copy = img.copy()

            resizerVal = 1

            if img.shape[1] > self.resizeMarkMask :
                resizerVal = self.resizeMarkMask/img.shape[1]

                img = cv.resize(img,None,fx=resizerVal,fy=resizerVal,interpolation=cv.INTER_AREA)

            self.getOptimalThresholdVal(img)

            gaussian = cv.GaussianBlur(img,(3,3),cv.BORDER_DEFAULT)
            edges = cv.Canny(gaussian,self.minThresholdVal,self.maxThresholdVal)

            nameWithEdge = saveFolder+"/"+filename        

            cv.imwrite(nameWithEdge, edges)

            img = cv.imread(cv.samples.findFile(nameWithEdge))

            reserVal = 1/resizerVal

            if not reserVal == 1:

                image = cv.resize(img,(self.resizeMark,image_copy.shape[0]),interpolation=cv.INTER_CUBIC)

                cv.imwrite(nameWithEdge,image)


    
    def isolateFabArtwork(self,filePath,type,saveFolder):

        filename = self.splitFileNames(filePath)

        if self.uniformCode in filename or self.texturedCode in filename:

            editedFileName = filePath
            img = cv.imread(cv.samples.findFile(editedFileName))

            img = img.copy()
            self.img = img
            self.img_copy = self.img.copy()

            self.output = np.zeros(self.img.shape, np.uint8)

            self.getOptimalThresholdVal(self.img_copy)


            if self.uniformCode in filename:
                self.generateUniformArtWorkMask(filename,type)

            if self.texturedCode in filename:
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


    def generateteTexturedArtworkDarft(self,sampleFilePath,filePath,saveFolder):

        blockSizeRows = 5
        blockSizeColumns = 5

        filename = self.splitFileNames(filePath)

        if self.texturedCode in filename:
            
            sampleFilename = self.splitFileNames(sampleFilePath)

            editedSampleFileName = sampleFilePath

            sampleImage = cv.imread(editedSampleFileName)
            sampleImageHSV = cv.cvtColor(sampleImage, cv.COLOR_BGR2HSV)
            sampleImageHIST = cv.calcHist([sampleImageHSV], [0,1], None, [180,256], [0,180,0,256])
            cv.normalize(sampleImageHIST, sampleImageHIST, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)


            sampleFilenameExeptExt = os.path.splitext(sampleFilename)[0]

            if sampleFilenameExeptExt in filename:

                editedFileName = filePath

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

                return saveFolder+"/"+filename
        
        return ""



    def sharpTexturedArtworkDraft(self,filePath):

        filename = self.splitFileNames(filePath)

        editedFileName = filePath
            
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
        maskName = ''

        if type == self.referenceImageType:
            editedFileName = self.artworksDraftsRef+'/'+ filename
            maskName = self.artworkMasksReferenceImages+"/"+filename

        if type == self.testImageType:
            editedFileName = self.artworksDraftsTest+'/'+ filename
            maskName = self.artworkMasksTestImages+"/"+filename

       
        img = np.array(Image.open(editedFileName).convert('L'))

        img = cv.normalize(img, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
        res, img = cv.threshold(img, 64, 255, cv.THRESH_BINARY)

        cv.floodFill(img, None, (0,0), 0)

        # kernel = np.ones((5,5),np.uint8)
        # img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
   
        self.mask = np.zeros(img.shape[:2],np.uint8)
        newmask = img

        self.mask[newmask == 0] = 0
        self.mask[newmask == 255] = 1            

        Image.fromarray(img).save(maskName)

    def generateOutputPath(self,outerRemovedOutPath,type):
        
        absPath = os.path.abspath(outerRemovedOutPath)

        outputPath = absPath.replace('\\', '/')

        return outputPath

    def sharpUniformArtworkMask(self,artworkPath):

        filename = self.splitFileNames(artworkPath)

        if self.uniformCode in filename:
            editedFileName = artworkPath
            
            image = cv.imread(editedFileName)

            
            mg = cv.medianBlur(image,9)

            # kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
            # erode = cv.erode(image, kernel, iterations=1)


            # kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
            # opening = cv.morphologyEx(erode, cv.MORPH_OPEN, kernel)

            cv.imwrite(editedFileName, mg)

            # image = Image.open(editedFileName)
            # image = image.filter(ImageFilter.MedianFilter(size=3))
            # image.save(editedFileName)

    def checkHavingRef(self,path):

        filename = self.splitFileNames(path)
        outerRemovedImagepath = self.outerRemReferenceImages+'/'+filename

        isTrue = os.path.isfile(outerRemovedImagepath)

        if isTrue:
            return True
        else:
            return False


    def setDefaultResolutionToAll(self,path,type):

        filename = self.splitFileNames(path)

        outerRemReferenceImagePath = self.outerRemReferenceImages+"/"+filename
        outerRemTestImagePath = self.outerRemTestImages+"/"+filename

        registratedTestImagePath = self.registratedTestImages+"/"+filename

        edgeReferenceImagePath = self.edgeReferenceImages+"/"+filename
        edgeTestImagePath = self.edgeTestImages+"/"+filename

        artworksDraftsRefImagePath = self.artworksDraftsRef+"/"+filename
        artworksDraftsTestImagePath = self.artworksDraftsTest+"/"+filename

        artworkMasksReferenceImagePath = self.artworkMasksReferenceImages+"/"+filename
        artworkMasksTestImagePath =  self.artworkMasksTestImages+"/"+filename

        artworksReferenceImagePath = self.artworksReferenceImages+"/"+filename
        artworksTestImagePath = self.artworksTestImages+"/"+filename

        fabricMasksRefImagePath = self.fabricMasksRef+"/"+filename
        fabricMasksTestImagePath = self.fabricMasksTest+"/"+filename

        if type == self.referenceImageType:
            if self.uniformCode in filename:
                self.resetImageResolution(edgeReferenceImagePath)

            if self.texturedCode in filename:
                self.resetImageResolution(artworksDraftsRefImagePath)

            self.resetImageResolution(artworkMasksReferenceImagePath)
            self.resetImageResolution(artworksReferenceImagePath)
            self.resetImageResolution(fabricMasksRefImagePath)

        if type == self.testImageType:
            # self.resetImageResolution(outerRemReferenceImagePath)
            self.resetImageResolution(outerRemTestImagePath)
            self.resetImageResolution(registratedTestImagePath)

            if self.uniformCode in filename:
                self.resetImageResolution(edgeTestImagePath)

            if self.texturedCode in filename:
                self.resetImageResolution(artworksDraftsTestImagePath)

            self.resetImageResolution(artworkMasksTestImagePath)
            self.resetImageResolution(artworksTestImagePath)
            self.resetImageResolution(fabricMasksTestImagePath)


    def resetImageResolution(self,path):

        image = cv.imread(path)

        imageWidth =  image.shape[1]

        resizerVal = imageWidth/self.originalWidth

        reserVal = 1/resizerVal

        if not reserVal == 1:

            image = cv.resize(image,None,fx=reserVal,fy=reserVal,interpolation=cv.INTER_CUBIC)

            cv.imwrite(path,image)






    def run(self,imgRefPath,imgTestPath,texSamplesPath,type):

        print('Start Background Removal Module Execution...')

        
        #creating output folders if not exists
        try:
            if not path.exists(self.outerRemReferenceImages):
                os.makedirs(self.outerRemReferenceImages)

            if not path.exists(self.outerRemTestImages):
                os.makedirs(self.outerRemTestImages)

            if not path.exists(self.registratedTestImages):
                os.makedirs(self.registratedTestImages)

            if not path.exists(self.edgeReferenceImages):
                os.makedirs(self.edgeReferenceImages)

            if not path.exists(self.edgeTestImages):
                os.makedirs(self.edgeTestImages)

            if not path.exists(self.artworksDraftsRef):
                os.makedirs(self.artworksDraftsRef)

            if not path.exists(self.artworksDraftsTest):
                os.makedirs(self.artworksDraftsTest)

            if not path.exists(self.artworkMasksReferenceImages):
                os.makedirs(self.artworkMasksReferenceImages)

            if not path.exists(self.artworkMasksTestImages):
                os.makedirs(self.artworkMasksTestImages)

            if not path.exists(self.artworksReferenceImages):
                os.makedirs(self.artworksReferenceImages)
                
            if not path.exists(self.artworksTestImages):
                os.makedirs(self.artworksTestImages)

            if not path.exists(self.fabricMasksRef):
                os.makedirs(self.fabricMasksRef)

            if not path.exists(self.fabricMasksTest):
                os.makedirs(self.fabricMasksTest)

        except:
            import traceback
            traceback.print_exc()

        if type == self.referenceImageType:
            self.deleteGeneratedFiles(self.outerRemReferenceImages)
            self.deleteGeneratedFiles(self.edgeReferenceImages)
            self.deleteGeneratedFiles(self.artworksDraftsRef)
            self.deleteGeneratedFiles(self.artworkMasksReferenceImages)
            self.deleteGeneratedFiles(self.artworksReferenceImages)
            self.deleteGeneratedFiles(self.fabricMasksRef)


        if type == self.testImageType:
            self.deleteGeneratedFiles(self.outerRemTestImages)
            self.deleteGeneratedFiles(self.registratedTestImages)
            self.deleteGeneratedFiles(self.edgeTestImages)
            self.deleteGeneratedFiles(self.artworksDraftsTest)
            self.deleteGeneratedFiles(self.artworkMasksTestImages)
            self.deleteGeneratedFiles(self.artworksTestImages)
            self.deleteGeneratedFiles(self.fabricMasksTest)

        refOuterRemovedFilePath =""
        testOuterRemovedFilePath =""

        if type == self.referenceImageType:

            refOuterRemovedFilePath = self.removeOuterBackground(imgRefPath,self.outerRemReferenceImages,type)

            self.generateUniformFabricEdge(refOuterRemovedFilePath,self.edgeReferenceImages)

            artworkDraftPath = self.generateteTexturedArtworkDarft(texSamplesPath,refOuterRemovedFilePath,self.artworksDraftsRef)

            if artworkDraftPath != "":
                self.sharpTexturedArtworkDraft(artworkDraftPath)

            refOuterRemovedFilePath = self.isolateFabArtwork(refOuterRemovedFilePath,type,self.artworksReferenceImages)

            self.setDefaultResolutionToAll(refOuterRemovedFilePath,type)

            return self.generateOutputPath(refOuterRemovedFilePath,type)

        if type == self.testImageType:

            refOuterRemovedFilePath = self.outerRemReferenceImages+"/"+self.splitFileNames(imgRefPath)
            testOuterRemovedFilePath = self.removeOuterBackground(imgTestPath,self.outerRemTestImages,type)
            
            testOuterRemovedFilePath = self.registratedMachingFiles(refOuterRemovedFilePath,testOuterRemovedFilePath,self.registratedTestImages)

            self.generateUniformFabricEdge(testOuterRemovedFilePath,self.edgeTestImages)

            artworkDraftPath = self.generateteTexturedArtworkDarft(texSamplesPath,testOuterRemovedFilePath,self.artworksDraftsTest)

            if artworkDraftPath != "":
                self.sharpTexturedArtworkDraft(artworkDraftPath)

            testOuterRemovedFilePath = self.isolateFabArtwork(testOuterRemovedFilePath,type,self.artworksTestImages)
            
            self.setDefaultResolutionToAll(testOuterRemovedFilePath,type)

            return self.generateOutputPath(testOuterRemovedFilePath,type)