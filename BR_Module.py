import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
from fuzzywuzzy import fuzz
from PIL import Image


class App():

    rect = (0,0,1,1)
    resizeMark = 1500
    resizerVal = .3
    
    def removeOuterBackground(self,folder,saveFolder):

        for filename in os.listdir(folder):

            print("Reading file : ",filename)

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

        img1_color = imageRef  # Image to be aligned. 
        img2_color = imageSample # Reference image.

        # Convert to grayscale. 
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

        nameWithMask = "fabric_masks/Mask_"+name        

        cv.imwrite(nameWithMask, newmask)

    def generateUniformArtWorkMask(self,folder,saveFolder):

        for filename in os.listdir(folder):

            editedFileName = folder +'/'+ filename
            img = cv.imread(cv.samples.findFile(editedFileName))

            print("Reading file",filename)

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

            editedFileName = folder +'/'+ filename
            img = cv.imread(cv.samples.findFile(editedFileName),cv2.COLOR_RGB2HSV)

            print("Reading file",filename)

            gaussian = cv.GaussianBlur(img,(3,3),cv.BORDER_DEFAULT)
            edges = cv.Canny(gaussian,self.minThresholdVal,self.maxThresholdVal)

            nameWithEdge = saveFolder+"/"+filename        

            cv.imwrite(nameWithEdge, edges)

    
    def isolateUniformFabArtwork(self,folder,maskFolder,saveFolder):

        for filename in os.listdir(folder):

            print("Reading file : ",filename)

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

            img = cv.imread(outputName)
            # kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
            # opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
            mg = cv.medianBlur(img,5)
            cv.imwrite(outputName,mg)

    def getUniformArtworkMask(self,img,filename,folder):

        maskName = "Mask_"+filename

        editedMaskFileName = folder +'/'+ maskName
        maskImg = cv.imread(cv.samples.findFile(editedMaskFileName))
        maskImg = cv.cvtColor(maskImg, cv.COLOR_BGR2GRAY)

        self.mask = np.zeros(img.shape[:2],np.uint8)
        newmask = maskImg

        self.mask[newmask == 0] = 0
        self.mask[newmask == 255] = 1
            




    def registratedMachingFiles(self,folder,refFolder,saveFolder):

        for refFilename in os.listdir(refFolder):

            for filename in os.listdir(folder):

                rat = fuzz.token_sort_ratio(filename, refFilename)

                if rat > 90:

                    editedRefFileName = refFolder +'/'+ refFilename

                    editedFileName = folder +'/'+ filename

                    imgRef = cv.imread(cv.samples.findFile(editedRefFileName))

                    img = cv.imread(cv.samples.findFile(editedFileName))

                    self.generateRegistratedImage(imgRef,img,filename,saveFolder)

    
    
    
    def deleteGeneratedFiles(self,folder):

        for filename in os.listdir(folder):

            editedFileName = folder +'/'+ filename
            os.remove(editedFileName)

            print("Deleted file",filename)

    
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

                if sampleFilenameExeptExt in filename:

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

            # img = cv.imread(saveFileName)
            # blur = cv.GaussianBlur(img,(5,5),0)
            # median = cv.medianBlur(blur,5)
            # blur = cv.bilateralFilter(median,9,75,75)

            # kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
            # opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
            # mg = cv.medianBlur(opening,5)

            # cv.imwrite(saveFileName, blur)


    def generateteTexturedArtworkMask2(self,folder,saveFolder):

        for filename in os.listdir(folder):

            editedFileName = folder +'/'+ filename
            
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            _,mask = cv.threshold(img,minThresholdVal,maxThresholdVal,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

            kernal = np.ones((5,5), np.uint8)
            mg = cv.morphologyEx(mask, cv.MORPH_GRADIENT, kernal)
            mg = cv.medianBlur(mg,5)

            gaussian = cv.GaussianBlur(mg,(3,3),cv.BORDER_DEFAULT)
            edges = cv.Canny(gaussian,minThresholdVal,maxThresholdVal)

            contours, hierarchy = cv.findContours(edges,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)
        

            #first sort the array by area
            sorteddata = sorted(contours, key = cv.contourArea, reverse=True)
        

            image_binary = np.zeros(img.shape[:2],np.uint8)

            for n in range(0,len(contours)):

                cv.drawContours(image_binary, [sorteddata[n]],-1, (255, 255, 255), -1)       

            cv.imwrite('ummaPatiya.jpg', image_binary)
            
            

    def run(self):

        # include defects free referece images of uniform fabrics
        refUniformfolder = 'Assets/BR_Module/Input/temp/ref_uniform'

        # include defects free outer background removed referece images of uniform fabrics
        outerRemovedRefUniformfolder = 'Assets/BR_Module/Input/temp/outer_removed_ref_uniform'

        # include defects test images of uniform fabrics
        uniformfolder = 'Assets/BR_Module/Input/temp/samples/Uniform'

        # include defects, outer background removed test images of uniform fabrics
        outerRemovedUniformfolder = 'Assets/BR_Module/Input/temp/outer_removed_samples/Uniform'

        # include defects, outer background removed and registrated test images of uniform fabrics
        uniformRegistrated = "Assets/BR_Module/Input/temp/registrated_samples/Uniform"

        # include edge images of uniform fabrics including the artwork
        edge = 'Assets/BR_Module/Input/temp/edge'

        # include artwork masks of uniform fabrics
        artworksMasks = 'Assets/BR_Module/Input/temp/artworks_masks'

        # include artwork of uniform fabrics/final output
        artworks = 'Assets/BR_Module/Input/temp/artworks'

        texturedSamples = 'Assets/BR_Module/Input/temp/textured_samples'
        texturedfolder = 'Assets/BR_Module/Input/temp/samples/Textured'
        outerRemovedTexturedfolder = 'Assets/BR_Module/Input/temp/outer_removed_samples/Textured'
        texArtworksMasks = 'Assets/BR_Module/Input/temp/tex_artworks_masks'
        texArtworksDrafts = 'Assets/BR_Module/Input/temp/tex_artworks_drafts'
        texArtworks = 'Assets/BR_Module/Input/temp/tex_artworks'

        # self.deleteGeneratedFiles(outerRemovedRefUniformfolder)
        # self.deleteGeneratedFiles(outerRemovedUniformfolder)
        # self.deleteGeneratedFiles(uniformRegistrated)
        # self.deleteGeneratedFiles(edge)
        # self.deleteGeneratedFiles(artworksMasks)
        # self.deleteGeneratedFiles(artworks)

        # self.deleteGeneratedFiles(outerRemovedTexturedfolder)
        # self.deleteGeneratedFiles(texArtworksDrafts)
        # self.deleteGeneratedFiles(texArtworksMasks)
        # self.deleteGeneratedFiles(texArtworks)

        # param : input folder,output folder
        self.removeOuterBackground(refUniformfolder,outerRemovedRefUniformfolder)
        self.removeOuterBackground(uniformfolder,outerRemovedUniformfolder)

        # param : input two folders,output folder
        self.registratedMachingFiles(outerRemovedUniformfolder,outerRemovedRefUniformfolder,uniformRegistrated)

        # param : input folder,output folder
        self.generateUniformFabricEdge(outerRemovedRefUniformfolder,edge)
        self.generateUniformFabricEdge(uniformRegistrated,edge)

        # param : input folder,output folder
        self.generateUniformArtWorkMask(edge,artworksMasks)

        # param : input two folders,output folder
        self.isolateUniformFabArtwork(outerRemovedRefUniformfolder,artworksMasks,artworks)
        self.isolateUniformFabArtwork(uniformRegistrated,artworksMasks,artworks)

        # self.removeOuterBackground(texturedfolder,outerRemovedTexturedfolder)
        self.generateteTexturedArtworkDarft(texturedSamples,outerRemovedTexturedfolder,texArtworksDrafts)
        self.sharpTexturedArtworkDraft(texArtworksDrafts)
        self.generateteTexturedArtworkMask(texArtworksDrafts,texArtworksMasks)
        self.isolateUniformFabArtwork(outerRemovedTexturedfolder,texArtworksMasks,texArtworks)

if __name__ == '__main__':
    # print(__doc__)
    App().run()
    cv.destroyAllWindows()