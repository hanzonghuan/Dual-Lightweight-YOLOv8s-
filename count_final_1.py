import cv2
from ultralytics import YOLO
import os

# model = YOLO('yolov8s.pt')  # 模型文件路径，准确度n<s<m<l
model = YOLO('C:/Users/CAU/Desktop/ultralytics-20240803/ultralytics-main/runs_train/exp16/weights/best.pt')  # 模型文件路径，准确度n<s<m<l


# 图像检测
def process_image(image_path):
    image = cv2.imread(image_path)

    # 检查图片是否成功加载
    if image is not None:
        # Run YOLOv8 inference on the image
        results = model(image, conf=0.5,classes=0)

        # Visualize the results on the image
        annotated_image = results[0].plot()

        # Calculate the number of objects in each class
        num_0 = sum(1 for box in results[0].boxes if box.cls == 0)

        # 在图像上添加文本信息
        cv2.putText(annotated_image, f"person: {num_0}", (50, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 保存带注释的图片
        output_path = '3.jpg'  # 设置输出图片的文件名
        cv2.imwrite(output_path, annotated_image)
        print('图片保存为：', output_path)

        # 显示带注释的图片
        cv2.imshow("YOLOv8 Inference", annotated_image)
        cv2.waitKey(0)  # 等待任意键继续
        cv2.destroyAllWindows()
    else:
        print("Error: Image could not be loaded.")


# 视频检测
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    # Get the video frame width and height
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('9.avi', fourcc, 20.0, (frame_width, frame_height))  # 设置输出的文件名
    # print('视频保存为：', out)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            results = model(frame, conf=0.5)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()
            cv2.putText
            # Calculate the number of objects in each class
            num_0 = sum(1 for box in results[0].boxes if box.cls == 0)
            cv2.putText(annotated_frame, f"person: {num_0}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Write the annotated frame to the output video
            out.write(annotated_frame)

            # Display the annotated frame
            cv2.imshow("检测计数", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) == 27:
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object, release the video writer, and close the display window
    cap.release()
    out.release()
    cv2.destroyAllWindows()


# 摄像头检测
def process_camera(camera_index_or_url):
    cap = cv2.VideoCapture(camera_index_or_url)

    # Get the video frame width and height
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('9.avi', fourcc, 20.0, (frame_width, frame_height))  # 设置输出的文件名

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            results = model(frame, conf=0.5)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()
            cv2.putText
            # Calculate the number of objects in each class
            num_0 = sum(1 for box in results[0].boxes if box.cls == 0)

            cv2.putText(annotated_frame, f"person: {num_0}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Write the annotated frame to the output video
            out.write(annotated_frame)

            # Display the annotated frame
            cv2.imshow("检测计数", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) == 27:
                break
        else:
            # Break the loop if the end of the video is reached
            break
    # Release the video capture object, release the video writer, and close the display window
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def process_images_in_folder(folder_path):
    output_folder = 'C:/Users/CAU/Desktop/ultralytics-20240803/ultralytics-main/VOCdevkit/images/test/test/'      # 输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)

            results = model(image, conf=0.5)
            annotated_image = results[0].plot()

            num_0 = sum(1 for box in results[0].boxes if box.cls == 0)
            num_1 = sum(1 for box in results[0].boxes if box.cls == 1)
            cv2.putText(annotated_image, f"Germinated: {num_0}", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.putText(annotated_image, f"Ungerminated: {num_1}", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, annotated_image)

    print(f"Detection results saved in '{output_folder}' folder.")


def main(input_source):



    if input_source == 0 or input_source.startswith(('http://', 'https://', 'rtsp://')):
        process_camera(input_source)
    elif input_source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        process_video(input_source)
    elif input_source.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
        process_image(input_source)
    elif os.path.isdir(input_source):
        process_images_in_folder(input_source)
    else:
        print("Unsupported input format.")


if __name__ == "__main__":

    input_source = 'C:/Users/CAU/Desktop/ultralytics-20240803/ultralytics-main/VOCdevkit/images/test/'     #设置输入文件路径
    main(input_source)
