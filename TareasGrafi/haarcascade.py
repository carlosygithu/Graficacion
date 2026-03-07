import cv2 as cv 

rostro = cv.CascadeClassifier('C:\\Users\\Asus\\OneDrive\\Documentos\\GRAFICACION\\TareasGrafi\\haarcascade_frontalface_alt2.xml')
cap = cv.VideoCapture(0)

while True:
    ret, img = cap.read()
    gris = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    rostros = rostro.detectMultiScale(gris, 1.3, 5)
    for(x,y,w,h) in rostros:
        res = int((w+h)/8)
        img = cv.rectangle(img, (x,y), (x+w, y+h), (234, 23,23), 5)
        #img = cv.rectangle(img, (x,int(y+h/2)), (x+w, y+h), (0,255,0),5 )
        img = cv.circle(img, (x + int(w*0.3), y + int(h*0.4)) , 21, (0, 0, 0), 2 )
        img = cv.circle(img, (x + int(w*0.7), y + int(h*0.4)) , 21, (0, 0, 0), 2 )
        img = cv.circle(img, (x + int(w*0.3), y + int(h*0.4)) , 20, (255, 255, 255), -1 )
        img = cv.circle(img, (x + int(w*0.7), y + int(h*0.4)) , 20, (255, 255, 255), -1 )
        img = cv.circle(img, (x + int(w*0.3), y + int(h*0.4)) , 5, (0, 0, 255), -1 )
        img = cv.circle(img, (x + int(w*0.7), y + int(h*0.4)) , 5, (0, 0, 255), -1 )
        img = cv.rectangle(img, (x+10,y+10), (x+w, y+h), (234,0 ,234), 5)
         # NARIZ
        img = cv.circle(img, (x + int(w*0.5), y + int(h*0.6)), 10, (0,0,255), -1)
        # BIGOTE IZQUIERDO
        img = cv.ellipse(img, (x + int(w*0.4), y + int(h*0.7)), (20,10), 0, 0, 180, (0,0,0), 3)
        # BIGOTE DERECHO
        img = cv.ellipse(img, (x + int(w*0.6), y + int(h*0.7)), (20,10), 0, 0, 180, (0,0,0), 3)
        img2=  img[y:y+h,x:x+w]
        cv.imshow('img2', img2)
    cv.imshow('img', img)
    if cv.waitKey(1)== ord('q'):
        break
    
cap.release
cv.destroyAllWindows()