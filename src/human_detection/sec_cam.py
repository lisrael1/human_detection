"""
short design doc to understand the tool.
what is the target of this package?
    outside security camera but with no web interface
    detect human motion and ignore bushes and shadows
    on detection, send mail with mp4 video of the detection.
    human detection based on real time AI (HOG+SVM at cv2).
what it does?
    get images from camera in a loop.
    every x images, check for human detection.
    on detection, taking all detected images and dumping them to mp4, then sending it over mail.
alternative:
    linux motion
        pros:
            already implemented and is stable
        cons:
            but it has a lot of false positive as it's just checking pixels change and outside you have
            a lot of pixel change due to clouds, trees, trees shadows and more.
            Nowadays, you don't buy security camera without AI detection
how to run this script:
    example for configuration file (it's a yaml format):
        mail:
          mail_user: my_gmail_user
          mail_pw: my_pw
          send_to: my_friend@gmail.com
        detection:
          check_human_every_x_images: 2
        output:
          output_folder: "@format {env[HOME]}/Downloads/captures"
          # output_folder: C:\\Users\\israelil\\Downloads\\captures
          save_all_images: False
          save_images_with_detections: True
          add_detection_box: True
          duplications_of_each_image_at_video: 5
        debug:
          print_debug_log: True


objects:
    HumanDetection
    ImageLogger
        gets the images and saves them.
        can export to jpg or mp4
    EmailSender
        get the mail user, pw, send to and the path to the mp4 file and send them
        using yagmail package.
        for gmail users:
            need to set 2 steps verification, set app pw (it's 16 letters)
            note that this user should not be under parental control otherwise it will be blocked.
            what is app pw?
                you have the user pw that can do a lot of modification, and app pw that can only
                send mails and that's it.
                why it's like this? because we don't want to save pw with weak permissions, and at the
                case of a breach, you can still change the pw easily without being blocked
    Camera
        read images from the camera
    DetectionStatus
        holds the detection status with history and its meaning - stop saving
        images and dump to mp4, or start saving images until detection ends.
    Flow
        control the whole flow.
        calling all other objects.
TODO
    check that yaml configuration file exist and it's correct
    test if it fails to detect or gives false alarm
    test at different place
    test when pointing to a tree
    delete old captures to avoid filling the disk
    at tests, put detection speed test, and also camera speed test
"""
import datetime
import os
import logging
import collections
from dataclasses import dataclass

import cv2
import pylab as plt
import yagmail
from dynaconf import Dynaconf


class HumanDetection:
    def __init__(self):
        self.image_path = None

        self.image_cv2_rgb = None
        self.image_gray = None
        self.image_with_boxes = None
        self.detection_boxes = None
        self.detection_weights = None
        self.detection_avg = None
        self.human_detected = None

        self.detection_brightness_level = 180

        # initializing the detector
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def set_image(self, image_path=None, image_content=None):
        """
            set an image, then the detector will check this image for human.
            can have image_path/image_content, but not both.
        :param image_path:
        :param image_content:
        :return:
        """
        self._reset_last_results()
        if image_path is None and image_content is None:
            raise ValueError('both image_path and image_content are none')
        if image_path is not None:
            self.image_path = image_path
            self.image_cv2_rgb = cv2.imread(self.image_path)
        else:
            self.image_cv2_rgb = image_content
        # detection is on colored image, and reading it as gray gives the exact same results
        # and overall time is the same
        self.detection_boxes, self.detection_weights = self.hog.detectMultiScale(self.image_cv2_rgb,
                                                                                 winStride=(8, 8),
                                                                                 # finalThreshold=-10,
                                                                                 )

        self._calc_average_per_detection_box()
        self._is_image_contains_human()

    def _reset_last_results(self):
        self.image_path = None

        self.image_cv2_rgb = None
        self.image_gray = None
        self.image_with_boxes = None
        self.detection_boxes = None
        self.detection_weights = None
        self.detection_avg = None
        self.human_detected = None

    def _calc_average_per_detection_box(self):
        self.detection_avg = []
        for (x, y, w, h) in self.detection_boxes:
            self.detection_avg.append(self.image_cv2_rgb[y:y + h, x:x + w].mean())

    def _is_image_contains_human(self):
        # detected bright area which probably false positive and also ignore low detection threshold
        if not len(self.detection_avg) or min(self.detection_avg) >= self.detection_brightness_level or max(
                self.detection_weights) < 0.5:
            self.human_detected = False
        else:
            self.human_detected = True

    def _draw_detection_boxes(self):
        if self.image_with_boxes is not None or self.human_detected is False:
            return False
        self.image_with_boxes = self.image_cv2_rgb.copy()
        for i, (x, y, w, h) in enumerate(self.detection_boxes):
            if self.detection_avg[i] > self.detection_brightness_level:
                continue
            cv2.rectangle(self.image_with_boxes, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(self.image_with_boxes,
                        text=f'W{self.detection_weights[i]:.2f}m{self.detection_avg[i]:.0f}',
                        org=(x + w // 10, y + h),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.7,
                        color=(0, 10, 255),
                        thickness=1,
                        lineType=cv2.LINE_AA,
                        )

    def get_image_with_boxes(self):
        """
            add detection box to last image, if having any detection.
            get image as 3D array
        :return:
        """
        if self.image_cv2_rgb is None:
            raise ReferenceError('no given image. need to run set_image first')
        self._draw_detection_boxes()
        return self.image_with_boxes

    def plot_image_with_boxes(self):
        """
            add detection box to last image, if having any detection, and plot it.
        :return:
        """
        self.get_image_with_boxes()
        rgb_img = cv2.cvtColor(self.image_with_boxes, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb_img)
        plt.show()

    def save_image_with_boxes(self, output_image_path):
        """
            add detection box to last image, if having any detection, and save it to disk.
        :param output_image_path:
        :return:
        """
        self.get_image_with_boxes()
        cv2.imwrite(output_image_path, self.image_with_boxes)


class ImageLogger:
    """
        get images and save them to history log.
        can export to mp4 or jpg
    """

    def __init__(self):
        self.images_history = None
        self.height = None
        self.width = None
        self.layers = None

        self.reset_history()

    def add_image(self, image):
        self.images_history.append(image)

    def reset_history(self):
        self.images_history = []
        self.height = None
        self.width = None
        self.layers = None

    def _image_properties(self):
        """
            all images in the history should be at the same shape.
            taking first image and extracting image shape (height and width) from it
        :return:
        """
        self.height, self.width, self.layers = self.images_history[0].shape

    def export_to_movie(self, video_full_path, duplications):
        """
            create a mp4 file from the history images and reset the history logger.
        :param video_full_path:
        :return:
        """
        if not len(self.images_history):
            raise 'no images to convert to a movie'
        self._image_properties()
        self._history_to_movie(video_full_path, duplications)
        self.reset_history()

    def _history_to_movie(self, video_full_path, duplications=1):
        """
            dump all images from history into mp4 file
        :param video_full_path: the mp4 file path to create
        :return:
        """
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        video = cv2.VideoWriter(filename=video_full_path, fourcc=fourcc, fps=24, frameSize=(self.width, self.height))
        # BTW, if you put fps=8, you will get green screen at iphone, but at android and windows it will be ok.

        for image in self.images_history:
            for _ in range(duplications):
                video.write(image)
        video.release()
        cv2.destroyAllWindows()

    def save_last_frame(self, file_path):
        cv2.imwrite(file_path, self.images_history[-1])


class EmailSender:
    def __init__(self, user=None, pw=None, send_to=None):
        self.disable = False
        if None in [user, pw, send_to]:
            self.disable = True
            return
        self.user = user
        self.pw = pw
        self.send_to = send_to

        self.connection = yagmail.SMTP(self.user, self.pw)

    def send_mail(self, video_full_path, subject):
        if self.disable:
            return
        contents = ["new caption", video_full_path]
        self.connection.send(self.send_to, subject, contents)


class Camera:
    def __init__(self, camera_number=0):
        self.caption = cv2.VideoCapture(camera_number)

    def take_picture(self):
        ret, frame = self.caption.read()
        return frame


@dataclass
class DetectionStatus:
    dump_video: bool = False
    save_images: bool = False
    how_many_detections: int = 3
    # FIFO:
    last_detections: list = collections.deque([False] * how_many_detections, maxlen=how_many_detections)

    @property
    def detection(self):
        return sum(self.last_detections) >= self.how_many_detections / 2

    def update_detection_status(self, last_detection_result):
        self.last_detections.append(last_detection_result)

        if not self.save_images and self.detection:
            # new recording
            self.save_images = True
        if self.save_images and not self.detection:
            # end of recording
            self.dump_video = True  # the dumper should return this to false
            self.save_images = False


class Flow:
    def __init__(self, path_to_conf_yml_file):
        self.path_to_conf_yml_file = path_to_conf_yml_file

        self.output_folder = None
        self.conf = None
        self.image_date = None

        self.update_configurations()
        self.detection_status = DetectionStatus()
        self.detector = HumanDetection()
        self.logger = ImageLogger()
        self.sender = EmailSender(user=self.conf.mail.mail_user,
                                  pw=self.conf.mail.mail_pw,
                                  send_to=self.conf.mail.send_to)
        self.camera = Camera()

        while True:
            self.looping()

    def update_configurations(self):
        os.environ['HOME'] = os.path.expanduser("~")
        self.conf = Dynaconf(settings_files=[self.path_to_conf_yml_file], environments=False)
        if self.conf.debug.print_debug_log:
            logging.basicConfig(level=logging.DEBUG, format='%(asctime)s: %(message)s', datefmt='%Y.%m.%d - %H:%M:%S')
        else:
            logging.basicConfig()
        self.output_folder = self.conf.output.output_folder

    def prepare_mail(self):
        frames = len(self.logger.images_history)
        mp4_full_path = f'{self.output_folder}/caption_{self.image_date}_{frames:>04}.mp4'
        self.logger.export_to_movie(mp4_full_path, duplications=self.conf.output.duplications_of_each_image_at_video)
        subject = f'new motion capture at {self.image_date} with {frames} frames'
        return subject, mp4_full_path

    def take_images(self):
        for _ in range(self.conf.detection.check_human_every_x_images):
            self.logger.add_image(self.camera.take_picture())
        self.image_date = datetime.datetime.now().strftime('%Y.%m.%d_%H.%M.%S')
        self.detector.set_image(image_content=self.logger.images_history[-1])
        self.detection_status.update_detection_status(self.detector.human_detected)
        logging.debug(f'last detections {sum(self.detection_status.last_detections)}/'
                      f'{self.detection_status.how_many_detections}')

    def take_actions(self):
        if self.conf.output.save_all_images:
            self.logger.save_last_frame(f'{self.output_folder}/debug_all_images_{self.image_date}.jpg')
        if self.conf.output.save_images_with_detections and self.detection_status.save_images:
            self.logger.save_last_frame(f'{self.output_folder}/debug_detected_images_{self.image_date}.jpg')
        if self.detection_status.save_images:
            if self.conf.output.add_detection_box:
                self.logger.images_history[-1] = self.detector.get_image_with_boxes()
            logging.debug('detected')
        if self.detection_status.dump_video:  # end of a caption
            subject, mp4_full_path = self.prepare_mail()
            self.sender.send_mail(mp4_full_path, subject)
            logging.debug(f'done capturing. sending mail with subject {subject}')
            self.detection_status.dump_video = False
        if not self.detection_status.save_images:
            self.logger.reset_history()
            logging.debug('not detected')

    def looping(self):
        logging.debug('*** starting new loop ***')
        self.take_images()
        self.take_actions()


