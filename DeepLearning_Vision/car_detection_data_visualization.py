import cv2
import matplotlib.pyplot as plt

def draw_boxes_on_img(img_file, anno_file):
    img = cv2.imread(img_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with open(anno_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        values = list(map(float, line.strip().split(' ')))
        class_id = int(values[0])
        x_min, y_min = int(round(values[1])), int(round(values[2]))
        x_max= int(round(max(values[3], values[5], values[7])))
        y_max = int(round(max(values[4], values[6], values[8])))

        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0,255,0), 2)
        cv2.putText(img, str(class_id), (x_min, y_min - 5), cv2.FONT_HERSHEY_PLAIN, 5, (0,255,0))

    cv2.imshow('test', img)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        exit()


if __name__ == '__main__':
    img_file = './data/car_detection_dataset/train/syn_00000.png'
    anno_file = './data/car_detection_dataset/train/syn_00000.txt'

    draw_boxes_on_img(img_file, anno_file)