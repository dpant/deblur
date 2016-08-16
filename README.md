You need to have python opencv2 and matlab installed in your system to run the software.

Sample usage

detect blur
detect_and_recover.py img_file.png  --detect_face_blur 1

deblur out of focus

detect_and_recover.py img_file.png  --defocus_deblur 1

deblur linear motion deblur

detect_and_recover.py img_file.png  --motion_deblur 1
