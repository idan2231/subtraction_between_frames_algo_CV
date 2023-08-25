import cv2
import numpy as np

cap = cv2.VideoCapture("video.mp4")

algo = cv2.bgsegm.createBackgroundSubtractorMOG()

count_line_position = 550
min_width_react = 55
min_hieght_react = 55
detect = []
offset = 6
counterIN = 0
counterOUT = 0

def center_handle(x,y,w,h):
    x1=int(w/2)
    y1=int(h/2)
    cx=x+x1
    cy=y+y1
    return cx,cy

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver




while True:
    ret, frame1 = cap.read()
    #frame1 = cv2.resize(frame1, (0, 0), None, .4, .4) #out
    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    #grey_3_channel = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)

    blur = cv2.GaussianBlur(grey,(3, 3), 5)
    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub, np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
    contourShape,h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #numpy_vertical = np.concatenate((frame1, grey_3_channel),axis=0) #out
    #numpy_horizontal = np.concatenate((img_sub, dilatada),axis=0) #out

    cv2.line(frame1,(1,count_line_position),(1280,count_line_position), (255,127,0),3)


    for (i,c) in enumerate(contourShape):
        x,y,w,h = cv2.boundingRect(c)
        validate_counter = (w>=min_width_react) and (h>=min_hieght_react)
        if not validate_counter:
            continue
        cv2.rectangle(frame1, (x,y), (x+w,y+h),(0,0,255),2)
        if x<640: #Right side
            cv2.putText(frame1, "Vehicle" + str(counterIN), (x,y-20), cv2.FONT_HERSHEY_TRIPLEX,1,(255,244,0),2 )
        else: #Left side
            cv2.putText(frame1, "Vehicle" + str(counterOUT), (x,y-20), cv2.FONT_HERSHEY_TRIPLEX,1,(255,244,0),2 )


        center = center_handle(x,y,w,h)
        detect.append(center)
        cv2.circle(frame1, center,4, (0,0,255), -1)

        for (x,y) in detect:
            if y<count_line_position+offset and y>count_line_position-offset and x<640:
                counterIN+=1
                cv2.line(frame1,(0,count_line_position),(640,count_line_position), (0,127,255),3)
                detect.remove((x,y))
                print("Vehicle Counter Enter:" + str(counterIN))
            elif y < count_line_position + offset and y > count_line_position - offset and x > 640:
                counterOUT += 1
                cv2.line(frame1, (640, count_line_position), (1280, count_line_position), (255, 25, 255), 3)
                detect.remove((x, y))
                print("Vehicle Counter Out:" + str(counterOUT))

    cv2.putText(frame1, "Vehicle Counter Enter: " + str(counterIN), (25,70), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2 )
    cv2.putText(frame1, "Vehicle Counter Out: " + str(counterOUT), (650,70), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2 )
    cv2.putText(frame1, "Total in the city: " + str(counterIN-counterOUT), (460,140), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2 )


    #print(detect)
    #cv2.imshow('detecter', dilatada)
    #cv2.imshow('Video1', img_sub)
    cv2.imshow('Result',stackImages(0.7, ([img_sub,dilatada],[grey,frame1])))

    #cv2.imshow('Video', frame1) #in


    #cv2.imshow('Video2', grey)
    #cv2.imshow('detecter1', numpy_vertical) #out
    #cv2.imshow('detecter2', numpy_horizontal) #out




    if cv2.waitKey(1)== 13 : #enter
        break

cv2.destroyAllWindows()
cap.release()