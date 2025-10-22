import cv2
from ultralytics import YOLO

# import model
model = YOLO(r"\THERMAL_GUN_DETECT\runs\detect\handgun_train3\weights\best.pt")  # path of the best.pt

# image path
image_path = r"\THERMAL_GUN_DETECT\test_images\test1.jpg"  # your own image way

# read the image
img = cv2.imread(image_path)
if img is None:
    print("Görsel yüklenemedi, yolu kontrol et!")
    exit()

# predict with model
results = model.predict(source=img, conf=0.5, imgsz=640, device=0, verbose=False)

annotated_img = results[0].plot()

# show
cv2.imshow("Thermal Gun Detection", annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# (optional) save the result
cv2.imwrite(r"C:\TEMP\THERMAL_GUN\predictions\result.jpg", annotated_img)
print("Sonuç kaydedildi: C:\\TEMP\\THERMAL_GUN\\predictions\\result.jpg")
