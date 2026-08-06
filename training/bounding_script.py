import cv2

image_path = "/home/hitesh/dev/rescue-swarm/datasets/usable/VisDrone2019-DET-train/images/9999998_00385_d_0000337.jpg"
annotation_path= "/home/hitesh/dev/rescue-swarm/datasets/usable/VisDrone2019-DET-train/annotations/9999998_00385_d_0000337.txt"
image = cv2.imread(image_path)

with open(annotation_path,"r") as file:
    for line in file:
        values= [int(value) for value in line.strip().split(",")]
        if values[5]==1:
            x, y, width, height = values[:4]
            x2=x+width
            y2=y+height
            cv2.rectangle(
                image,
                (x,y),
                (x2,y2),
                color=(0,255,0),
                thickness=2
            )
small = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
cv2.imshow("VisDrone", small)
cv2.waitKey(0)
cv2.destroyAllWindows()