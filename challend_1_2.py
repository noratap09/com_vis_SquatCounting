import numpy as np
import cv2
import glob

path = glob.glob("SquatCounting/*.avi")
cout=0
start = [360,310,390]
end = [800,910,1030]
for vedio in path:
    cap = cv2.VideoCapture(vedio)
    _,bg = cap.read()

    fram = 0
    while(cap.isOpened()):
        haveFrame,im = cap.read()
        if(fram>start[cout] and fram<end[cout]):
            cv2.putText(im,str(fram),(50,100),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0))
        else:
            cv2.putText(im,str(fram),(50,100),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255))
        fram=fram+1
        if (not haveFrame) or (cv2.waitKey(1) & 0xFF == ord('q')):
            break


        diffc = cv2.absdiff(im,bg)
        diffg = cv2.cvtColor(diffc,cv2.COLOR_BGR2GRAY)
        bwmask = cv2.inRange(diffg,50,255)

        bwmask = cv2.medianBlur(bwmask,35)

        kernel = np.ones((90,1), np.uint8)
        kernel_o = np.ones((1,30), np.uint8)
        kernel_d = np.ones((5,1), np.uint8)

        bwmask = cv2.dilate(bwmask,kernel)
        bwmask = cv2.morphologyEx(bwmask, cv2.MORPH_CLOSE, kernel)
        bwmask = cv2.morphologyEx(bwmask, cv2.MORPH_OPEN, kernel_o)



        temp = bwmask.copy()
        contours,hierarchy = cv2.findContours(temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


        if(fram==start[cout]):
            line = np.zeros((len(contours),3))
            score = np.zeros(len(contours))
            ck = np.zeros(len(contours))
            _,bg = cap.read()



        for i in range(0,len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i]) #หากรอบสี่เหลียม
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(im,str(x)+"/"+str(y),(x,y),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255))
            if(fram==start[cout]):
                line[i][0] = x+w/2
                line[i][1] = y+h*0.1
                line[i][2] = y+h*0.2

        if(fram>start[cout] and fram<end[cout]):
            for i in range(0,len(line)):
                cv2.circle(im, (int(line[i][0]), int(line[i][1])), 10, (0, 0, 255), -1)
                cv2.circle(im, (int(line[i][0]), int(line[i][2])), 10, (0, 255, 255), -1)
                cv2.putText(im,str(score[i]),(int(line[i][0])+10,int(line[i][1])),cv2.FONT_HERSHEY_PLAIN,2,(0,255,255))
                #cv2.putText(im,str(bwmask[int(line[i][1])][int(line[i][0])]),(int(line[i][0]),int(line[i][1])),cv2.FONT_HERSHEY_PLAIN,2,(0,255,255))
                if(bwmask[int(line[i][2])][int(line[i][0])]==255):
                    if(ck[i]==0):
                        score[i] = score[i]+1
                        ck[i]=1
                elif(bwmask[int(line[i][1])][int(line[i][0])]==0):
                    ck[i]=0




        cv2.drawContours(im, contours, -1, (0, 255, 0), 2)

        cv2.imshow('bwmask', bwmask)
        #cv2.moveWindow('bwmask',10,10)
        cv2.imshow('gray', diffg)
        #cv2.imshow('temp', temp)
        cv2.imshow('im', im)
        #cv2.moveWindow('im', 800, 10)

    cap.release()
    cv2.destroyAllWindows()
    cout=cout+1
