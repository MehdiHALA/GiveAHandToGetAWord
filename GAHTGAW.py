import cv2
import numpy as np
import math
import time
import sqlite3
import string as st

sqlite_file = './DDB.sq3'
conn = sqlite3.connect(sqlite_file)
cur = conn.cursor()

A =st.ascii_uppercase #Alphabet


LR=[]   #List of rectangle areas
LC=[]   #List of circle areas
LF=[]
t=time.time()
cap = cv2.VideoCapture(0)
while(cap.isOpened()):
    # Opens the image
    ret, img = cap.read()

    # Shows the square
    cv2.rectangle(img, (300,300), (100,100), (0,255,0)) # img pt1 pt2 color 
    coupe_img = img[100:300, 100:300]

    # BGR to Grey
    grey = cv2.cvtColor(coupe_img, cv2.COLOR_BGR2GRAY) #Source , BGR to Grey

    # Guassian Blur
    value = (35, 35)
    blurred = cv2.GaussianBlur(grey, value,0)   #Flou
    cv2.imshow('gris', grey) # Nom fenêtre / Source 


    # Otsu's thresholding
    _, thresh1 = cv2.threshold(blurred, 127, 255,
                               cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Thresholding window
    cv2.imshow('Binarisation', thresh1) # Nom fenêtre / Source 

    # Contours with maximum area
    image, contours, hierarchy = cv2.findContours(thresh1.copy(), \
        cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) #source, finding method, approximation method  

    
    cnt = max(contours, key = lambda x: cv2.contourArea(x))
    # Shows bounding rectangle 
    
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(coupe_img, (x, y), (x+w, y+h), (0, 0, 255), 0)  # x,y coord du pt haut-gauche(0,0)/ w:largeur / h:hauteur
    
    # Shows enclosing circle
    
    centre ,rayon = cv2.minEnclosingCircle(cnt)
    c1,c2=centre
    c1,c2=int(c1),int(c2)
    rayon=int(rayon)
    cv2.circle(coupe_img, (c1,c2), rayon, (255,0,0), 1)

    t1=time.time()
    
    if t1-t<=5:
        LR.append(w*h)
        LC.append(np.pi*rayon**2)
        LF.append(cv2.contourArea(cnt))
        text=5+int(t-t1)
        text=str(text)
        cv2.putText(img,text,(600, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,10,5)   # Compte à rebours 
    else : 
        t=t1
        sur_rec=np.mean(LR)   #average of areas
        sur_cer=np.mean(LC)
        sur_fr=np.mean(LF)
        LR,LC=[],[]
        print(sur_rec,sur_cer,sur_fr) 
        cur.execute("SELECT id FROM Gestes WHERE Rec_mi<{} and Rec_ma>{} and Cer_mi<{} and Cer_ma>{} and Cnt_mi<{} and Cnt_ma>{}".format(sur_rec,sur_rec,sur_cer,sur_cer,sur_fr,sur_fr))
        for e in cur:
            cv2.putText(img,A[e[0]],(50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,10,5) 
            print(A[e[0]])
        
        
    # Convex Hull
    hull = cv2.convexHull(cnt)
    

    # Drawing the contours
    drawing = np.zeros(coupe_img.shape,np.uint8)
    cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
    cv2.drawContours(drawing, [hull], 0,(0, 0, 255), 0) #source contour .... couleur epaisseur 

  
    hull = cv2.convexHull(cnt, returnPoints=False)

    # If there's a convexity defects 
    defects = cv2.convexityDefects(cnt, hull)
    count_defects = 0
    cv2.drawContours(thresh1, contours, -1, (0, 255, 0), 3)

    # Angle (Alkashi's method)
    # if angle > 90, ignore it
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]

        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])

        # lenght of each edge of the triangle
        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

        #Alkashi
        angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
        # ignorer les angles>90 et mettre un point rouge sur les autres
        if angle <= 90:
            count_defects += 1
            cv2.circle(coupe_img, far, 1, [0,0,255], -1)
        #dist = cv2.pointPolygonTest(cnt,far,True)

        # trace the outline (fingertips)
        cv2.line(coupe_img,start, end, [0,255,0], 2)



    # show
    cv2.imshow('Gesture', img)
    all_img = np.hstack((drawing, coupe_img))
    cv2.imshow('Contours', all_img)

    k = cv2.waitKey(10)
    if k == 27:
        break
