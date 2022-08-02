working on this one...

security camera base on HOG+SVM.
can improve results if using YOLO, but YOLO is heavier for real time on old pc.

install with 
```bash
pip install human_detection
```
run from terminal:
```bash
human_motion_detection -c "C:\Users\my_name\Downloads\configurations.yml"
```
configurations.yml for example:
```yml
mail:
  mail_user: my_google_user
  mail_pw: asdfwasdfwefsvdv
  send_to: my_friend@gmail.com
detection:
  check_human_every_x_images: 2
output:
  output_folder: "@format {env[HOME]}/Downloads/captures"
  # output_folder: C:\Users\my_name\Downloads\captures
  save_all_images: False
  save_images_with_detections: True
  add_detection_box: True
  duplications_of_each_image_at_video: 5
debug:
  print_debug_log: True

```

```
TODO
    add demo video
    add image with detection box
```