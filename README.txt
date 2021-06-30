The file "back_change.py" contains code, which takes two inputs:
1) path to the input video(0 can be used to take input from webcam)
2) path to the background image.
While the code is running it will also write the output in the "output.avi" file.

Working of the code:

The input video and the image should be at 720x1280p.
The code utilizes the pretrained TFLite model in the file "deeplabv3_1_default_1.tflite" which returns the output in the form of the label corresponding to the pixel. 
Then these labels( label # 15 corresponds to person class) is used to create mask of a person.
With the help of the mask and inverse matrix of the mask the background image is merged with the image of the person through numpy matrix operations.
