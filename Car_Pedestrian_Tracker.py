import cv2

video=cv2.VideoCapture('videocp.mp4')

#Pre-trained car classifier
classifier_file='car_detector.xml'

#Pre-trained pedestrian clasifier
classifier_file_ped='haarcascade_fullbody.xml'

#run classifier
car_tracker=cv2.CascadeClassifier(classifier_file)
pedestrian_tracker=cv2.CascadeClassifier(classifier_file_ped)

while True:

    #Read current frame
    (read_successful,frame)=video.read()

    #Safe coding
    if read_successful:
        #Must be in grayscale
        grayscaled_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    else:
        break
    
    #detect cars
    cars=car_tracker.detectMultiScale(grayscaled_frame)

    #detect pedestrians
    pedestrians=pedestrian_tracker.detectMultiScale(grayscaled_frame)

    #Draw rectangles around cars
    for (x,y,w,h) in cars:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

    #Draw rectangles around pedestrians
    for (x,y,w,h) in pedestrians:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow('Car and Pedestrian detector',frame)
    key=cv2.waitKey(1)

    if key==81 or key==113:
        break

#Release video capture object
video.release()

print('Code Completed')