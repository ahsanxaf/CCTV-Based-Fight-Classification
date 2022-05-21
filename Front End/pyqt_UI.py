from PyQt5 import QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QTimer,QDateTime
import sys
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import os
import numpy as np
from tensorflow.python.util import compat

import tensorflow as tf
import keras


another_strategy = tf.distribute.MirroredStrategy()
filename=os.path.join(os.getcwd(), 'fight_det_model.h5')
print(compat.as_bytes(filename))
print(os.getcwd())

dataset_dir = r'dataset_/'
Categories = ['nofight', 'fight']

data = []
labels = []
# im = cv2.imread("sample_2.jpeg")

loaded=keras.models.load_model(filename)

global predict_sig
global predict_text
global predict_text_pre

predict_text_pre="";
predict_sig=False
predict_text="......"
class VideoThread(QThread):
    
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        global predict_sig
        global predict_text
        # capture from web cam
        self._run_flag = True
        self.cap = cv2.VideoCapture(0)
        while self._run_flag:
            try:
                ret, cv_img = self.cap.read()
                if ret:
                    if(predict_sig==True):
                        
                        data=[]
                        #cv_img = cv2.resize(cv_img, (224,224))
                        im = cv2.resize(cv_img, (224, 224)) 
                        data.append(im)
                        data = np.array(data)
                        
                        l1=loaded.predict(data)
                        y_classes = l1.argmax(axis=-1)
                        print(y_classes)
                        print(Categories[int(y_classes)])
                        predict_text=str(Categories[int(y_classes)])
                    self.change_pixmap_signal.emit(cv_img)
            except:
                pass
                

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.cap.release()
        cv2.destroyAllWindows()


class MyWindow(QMainWindow):
    
    def __init__(self):
        
        super(MyWindow, self).__init__()
        #self.available_cameras = QCameraInfo.availableCameras()  # Getting available cameras

        cent = QDesktopWidget().availableGeometry().center()  # Finds the center of the screen
        self.setStyleSheet("background-color: white;")
        self.resize(1400, 800)
        self.frameGeometry().moveCenter(cent)
        self.setWindowTitle('CCTV Fight Classification')
        self.initWindow()
        self.timer = QTimer()
        self.timer.setInterval(500)
        self.timer.timeout.connect(self.recurring_timer_300)
        self.timer.start()

########################################################################################################################
#                                                   Windows                                                            #
########################################################################################################################
    def initWindow(self):
        
        # create the video capture thread
        self.thread = VideoThread()
        
        self.label_1 = QtWidgets.QLabel(self)  # Create label
        self.label_1.setStyleSheet("font: bold 30pt Times New Roman;""color:black;border:1px solid  #b0c4de")
        self.label_1.setFixedSize(700,70)
        self.label_1.setText("CCTV Fight Classification System")
        self.label_1.move(300, 20)  # Allocate label in window
##        self.label.resize(300, 20)  # Set size for the label
        self.label_1.setAlignment(Qt.AlignCenter)  # Align text in the label
        
        self.label = QtWidgets.QLabel(self)  # Create label
        self.label.setStyleSheet("font: bold 25pt Times New Roman;""color:black;border:1px solid  #b0c4de")
        self.label.setFixedSize(300,100)
        self.label.move(850, 520)  # Allocate label in window
        self.label.setAlignment(Qt.AlignCenter)  # Align text in the label

        # Button to start video
        self.ss_video = QtWidgets.QPushButton(self)
        self.ss_video.setStyleSheet("font: bold 25pt Times New Roman;""color:black;background-color:#b0c4de")
        self.ss_video.setText('Start video')
        self.ss_video.move(850, 170)
        self.ss_video.resize(300, 100)
        self.ss_video.clicked.connect(self.ClickStartVideo)

        # Button to predict video
        self.p_video = QtWidgets.QPushButton(self)
        self.p_video.setStyleSheet("font: bold 25pt Times New Roman;""color:black;background-color:#b0c4de")
        self.p_video.setText('PREDICT')
        self.p_video.move(850, 370)
        self.p_video.resize(300, 100)
        self.p_video.clicked.connect(self.ClickPredictVideo)

        # Status bar
        self.status = QStatusBar()
        self.status.setStyleSheet("background : white;")  # Setting style sheet to the status bar
        self.setStatusBar(self.status)  # Adding status bar to the main window
        self.status.showMessage('Ready to start')

        self.image_label = QLabel(self)
        self.disply_width = 669
        self.display_height = 501
        self.image_label.resize(self.disply_width, self.display_height)
        self.image_label.setStyleSheet("background : gray;")
        self.image_label.move(80, 140)
        self.setStyleSheet('background-color:#696969')

########################################################################################################################
#                                                   Buttons                                                            #
########################################################################################################################
    # Activates when Start/Stop video button is clicked to Start (ss_video
    def ClickStartVideo(self):
        # Change label color to light blue
        self.ss_video.clicked.disconnect(self.ClickStartVideo)
        self.status.showMessage('Video Running...')
        # Change button to stop
        self.ss_video.setText('Stop video')
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)

        # start the thread
        self.thread.start()
        self.ss_video.clicked.connect(self.thread.stop)  # Stop the video if button clicked
        self.ss_video.clicked.connect(self.ClickStopVideo)

    # Activates when Start/Stop video button is clicked to Stop (ss_video)
    def ClickStopVideo(self):
        self.thread.change_pixmap_signal.disconnect()
        self.ss_video.setText('Start video')
        self.status.showMessage('Ready to start')
        self.ss_video.clicked.disconnect(self.ClickStopVideo)
        self.ss_video.clicked.disconnect(self.thread.stop)
        self.ss_video.clicked.connect(self.ClickStartVideo)

     # Activates when Start/Stop video button is clicked to Stop (ss_video)
    def ClickPredictVideo(self):
        global predict_sig
        
        if(predict_sig==True):
            
            predict_sig=False
        elif(predict_sig==False):
            predict_sig=True
        
    def recurring_timer_300(self):
        global predict_text
        global predict_text_pre
        if(predict_text!=predict_text_pre):
            
##            self.label.setText(str(predict_text))
            print(str(predict_text))
            print(predict_text)
            self.label.setText(predict_text)
##            self.label.adjustSize()
            predict_text_pre=predict_text
            print(predict_text)

########################################################################################################################
#                                                   Actions                                                             #
########################################################################################################################

    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        #p = convert_to_Qt_format.scaled(801, 801, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


if __name__ == '__main__':
    
    app = QApplication(sys.argv)
    win = MyWindow()
    win.show()
    sys.exit(app.exec())
   

# Clean up
cv2.destroyAllWindows()

