num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clean_mask)

contador = 0

for i in range(1, num_labels):

    area = stats[i, cv2.CC_STAT_AREA]

    if area > 500:
        contador += 1
        print("Fruta detectada:", contador)
        print("Area aproximada:", area)

print("Total de frutas detectadas:", contador)
