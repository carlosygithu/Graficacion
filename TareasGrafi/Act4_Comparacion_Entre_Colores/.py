colores = {

"rojo": ([0,120,70],[10,255,255]),
"verde": ([35,80,80],[85,255,255]),
"amarillo": ([20,100,100],[30,255,255])

}

for nombre,(lower,upper) in colores.items():

    lower = np.array(lower)
    upper = np.array(upper)

    mask = cv2.inRange(hsv,lower,upper)

    kernel = np.ones((5,5),np.uint8)
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)

    num_labels,labels,stats,centroids = cv2.connectedComponentsWithStats(mask)

    contador = 0

    for i in range(1,num_labels):

        area = stats[i,cv2.CC_STAT_AREA]

        if area > 500:
            contador += 1

    print("Color:",nombre,"Frutas detectadas:",contador)
