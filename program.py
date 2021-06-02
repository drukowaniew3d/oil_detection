import cv2
import numpy as np
import sys

def getContours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 50:
            cv2.drawContours(newFrame, cnt, -1, (50, 0, 255), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            srodek_x = int(x+w/2)
            srodek_y = int(y+h/2)
            #print(srodek_x, srodek_y)
            #cv2.rectangle(newFrame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.rectangle(newFrame, (srodek_x-5, srodek_y-5), (srodek_x+5, srodek_y+5), (255, 0, 0), 2)
            color = newFrame[srodek_y, srodek_x]
            print(color)


            if (color[0] > 58-50) and (color[0]<150) and (color[1] >= 10) and (color[1]<150) and (color[2] > 20) and (color[2]<190):
                #print('oil')
                cv2.putText(newFrame, "OIL LEAKAGE", (srodek_x-58, srodek_y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 0, 255, 150), 2)

            elif (color[0] > 165-50) and (color[0]<165+50) and (color[1] >= 155-50) and (color[1]<155+50) and (color[2] > 75-50) and (color[2]<75+50):
                #print('glikol')
                cv2.putText(newFrame, "COOLING FLUID LEAKAGE", (srodek_x - 120, srodek_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255, 100, 10, 150), 2)
            #else:
               # print('niezidentyfikowana zmiana')
               # cv2.putText(newFrame, "OTHER CHANGE", (srodek_x - 90, srodek_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255, 100, 10, 150), 2)

            #print(" ")


### constants,gloabal variables, frame parameters and paths ###
frameWidth = 960
frameHeight = 590
source = sys.argv[2]
cap = cv2.VideoCapture(source)
mask = cv2.imread("mask.png")

FirstChannels = np.zeros(3)

Dots = [[105, 408] , [150, 422]]   #first value is y coordinate and number of row
DotsLen = len(Dots)
DotSize = 6                                              #Half of rectangle size
DotSens = 30
FirstMeans = np.zeros((DotsLen, 3))

DotStarted = 0

### Add mask to hide moving parts ###
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
(thresh1, mask) = cv2.threshold(mask, 3, 255, cv2.THRESH_BINARY)

### Remember frame from the begining of run ###
success0, img = cap.read()
first = cv2.resize(img, (frameWidth, frameHeight))


writer = None

### Main program loop ###
while True: 

    key = cv2.waitKey(1) & 0xFF

 ### Subtract of first and actual frame ###
    success2, img_2 = cap.read()
    newFrame = cv2.resize(img_2, (frameWidth, frameHeight))
    imgResult = cv2.subtract(first,newFrame)
    #cv2.imshow("imgResult",imgResult)

#########################
### LEAKAGE DETECTION ###
#########################

 
    if sys.argv[1] == "lake_detection":


        ### Remember frame to compare when input0 = 1 (overwrite initial frame) ###
        if key == ord("0"):
            print('input0 = 1')
            success1, img = cap.read()
            first = cv2.resize(img, (frameWidth, frameHeight))
            #cv2.imshow("First", first)

       
        ### Erosion ###
        kernel = np.ones((10, 10), np.uint8)
        img_erosion = cv2.erode(imgResult, kernel, iterations=1)
        kernel2 = np.ones((5, 5), np.uint8)
        img_dilatation = cv2.dilate(img_erosion, kernel2, iterations=1)
        #cv2.imshow("img_erosion ",img_erosion )

        ### Transforamtion to Grayscale and Black&White image with treshold ###
        grayImage = cv2.cvtColor(img_dilatation, cv2.COLOR_BGR2GRAY)
        (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 10, 255, cv2.THRESH_BINARY)
        #cv2.imshow("blackAndWhiteImage",blackAndWhiteImage)

        ### Use mask to hide moving parts ###
        masked = cv2.subtract(blackAndWhiteImage, mask)
        #cv2.imshow("masked", masked)

        ### Contour detection and area filtration (area, color) ###
        imgCanny = cv2.Canny(masked,50,50)
        imgContour = np.zeros_like(imgCanny)
        getContours(imgCanny)
        #cv2.imshow("imgCanny ",imgCanny)

###################
### SAFETY DOTS ###
###################

    elif sys.argv[1] == "safety_dots":

        ### show all safety areas ###
        for Iter in range(DotsLen):
                cv2.rectangle(newFrame, (Dots[Iter][1] - DotSize, Dots[Iter][0] - DotSize), (Dots[Iter][1] + DotSize, Dots[Iter][0] + DotSize), (255, 0, 0), 2)

        ### Save mean colour of selected areas when input1=1 ###
        if key == ord("1"):
            print('input1 = 1')
            DotStarted = 1

            for Iter in range(DotsLen):
                FirstDot = newFrame[Dots[Iter][0]-DotSize:Dots[Iter][0] + DotSize, Dots[Iter][1]-DotSize:Dots[Iter][1] + DotSize]
                mean = cv2.mean(FirstDot)
                FirstMeans[Iter, 0] = mean[0]
                FirstMeans[Iter, 1] = mean[1]
                FirstMeans[Iter, 2] = mean[2]

        ### Compare first and actual mean colour of selected areas ###
        if DotStarted == 1:
            for Iter2 in range(DotsLen):
                ActualDot = newFrame[Dots[Iter2][0] - DotSize:Dots[Iter2][0] + DotSize, Dots[Iter2][1]-DotSize:Dots[Iter2][1] + DotSize]
                ActualMean = cv2.mean(ActualDot)

                if Iter2 == 1:
                    print('Kolor w obszarze:', Iter2)
                    print(ActualMean)
                    print('')

                if ((abs(FirstMeans[Iter2, 0] - ActualMean[0]) > DotSens) or (abs(FirstMeans[Iter2, 1] - ActualMean[1]) > DotSens) or (
                    abs(FirstMeans[Iter2, 2] - ActualMean[2]) > DotSens)):
                   # cv2.imshow('ActualDot',ActualDot)
                    print('Wykryto zmianę w obszarze nr', Iter2, 'o współrzędnych środka x=', Dots[Iter2][1], ' y=', Dots[Iter2][0])
                    cv2.rectangle(newFrame, (Dots[Iter2][1] - DotSize, Dots[Iter2][0] - DotSize), (Dots[Iter2][1] + DotSize, Dots[Iter2][0] + DotSize), (0, 0, 255), 2)
                    #cv2.imshow('ActualDot',ActualDot)
    else: 
        print("Wrong method. Please choose from: lake_detection or safety_dots")
        exit()

    ### SHOW RESULTS ###
    cv2.imshow("newFrame", newFrame)

    
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break

    if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter('output.mp4', fourcc, 30, (newFrame.shape[1], newFrame.shape[0]))
    else:
        writer.write(newFrame)

    if key == ord("q"):
            print('done')
            writer.release()
            cap.release()
            cv2.destroyAllWindows()
            exit()



