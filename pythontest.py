import cv2 as cv
import os
import numpy as np


import cv2.ccm

#directory data is in
dataPath = "./images/"
checkers = []
images = []

for img in os.listdir(dataPath):   # access all images in folder
    name = img #save the name
    img = cv.imread(os.path.join(dataPath, img), cv.IMREAD_COLOR) #read color image

    #create a colorcheck detetor instance
    detector = cv.mcc.CCheckerDetector.create()
    #find the MacBeth Color Checker chart in image


    found = cv.mcc.CCheckerDetector.process(detector, img, cv.mcc.MCC24, nc = 1)
    print(f"{name} detected chart: {found}")
    if (found):
        checker=detector.getBestColorChecker()
        cdraw = cv.mcc.CCheckerDraw.create(checker)
        img_draw = img.copy()
        cdraw.draw(img_draw)

        chartsRGB= checker.getChartsRGB()
        width, height = chartsRGB.shape[:2]
        ROI = chartsRGB[0:width, 1]
        print(ROI)

        rows = int(ROI.shape[:1][0])
        src = chartsRGB[:,1].copy().reshape(int(rows/3), 1, 3)
        src/=255
        print(src.shape)

        model= cv.ccm.ColorCorrectionModel(src, cv.ccm.COLORCHECKER_Macbeth)
        model.setColorSpace(cv.ccm.COLOR_SPACE_sRGB)
        model.setCCM_TYPE(cv.ccm.CCM_3X3)
        model.setDistance(cv2.ccm.DISTANCE_CIE2000)
        model.setLinear(cv.ccm.LINEARIZATION_GAMMA)
        model.setLinearGamma(2.2)
        model.setLinearDegree(3)
        model.setSaturatedThreshold(0,0.98)
        model.run()

        ccm=model.getCCM()
        print('ccm:\n{}\n'.format(ccm))
        loss=model.getLoss()
        print('loss:\n{}\n'.format(loss))

        img_=cv.cvtColor(img,  cv.COLOR_BGR2RGB)
        img_=img_.astype(np.float64)
        img_=img_/255
        calibratedImage = model.infer(img_)
        out_=calibratedImage * 255
        out_[out_<0]=0
        out_[out_>255] = 255
        out_ = out_.astype(np.uint8)

        out_img = cv.cvtColor(out_, cv.COLOR_RGB2BGR)
        ccimg_filename = name[:-4] + '_corrected.jpg'

        cv.imwrite(ccimg_filename, out_img)

        width, height = img.shape[:2]
        image=cv.resize(img, (int(height/4), int(width/4)), interpolation = cv.INTER_CUBIC)
        img_draw = cv.resize(img_draw, (int(height/4), int(width/4)), interpolation= cv.INTER_CUBIC)
        out_img = cv.resize(out_img, (int(height/4), int(width/4)), interpolation= cv.INTER_CUBIC)

        cv.namedWindow('Image')
        cv.imshow("image", image)
        cv.imshow('img_draw', img_draw)
        cv.imshow('Out Image', out_img)
        cv.waitKey(0)










