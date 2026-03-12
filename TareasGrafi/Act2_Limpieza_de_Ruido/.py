kernel = np.ones((5,5), np.uint8)

# apertura morfológica
clean_mask = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)

cv2.imshow("Mascara original", mask_red)
cv2.imshow("Mascara limpia", clean_mask)

cv2.waitKey(0)
cv2.destroyAllWindows()
