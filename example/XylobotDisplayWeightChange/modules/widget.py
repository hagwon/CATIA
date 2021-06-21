import math


from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QDockWidget
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QSplitter
from PyQt5.QtWidgets import QFrame, QLineEdit, QPushButton, QLabel, QProgressBar, QTextEdit
from PyQt5.QtWidgets import QGroupBox, QComboBox, QMessageBox, QInputDialog
from PyQt5.QtGui import QIcon, QPixmap, QColor, QPalette, QFont, QPainter, QPen, QImage
from PyQt5.QtCore import Qt, QTimer, QSize

if __name__ == "__main__" or __name__ == "widget":
    import communication
else :
    from . import communication

class TitleLayout(QFrame):

    def __init__(self, title_text='', main_text='', sub_text=''):
        super().__init__()

        self. initUI(title_text, main_text, sub_text)
    
    def initUI(self, title_text, main_text, sub_text):

        # title label
        title_label = QLabel(title_text, self)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet(
            'color: white;'
            'font: bold 50px Tahoma;'
            )

        # change window button
        self.prev_btn = QPushButton('', self)
        self.prev_btn.setStyleSheet(
            'color: white;'
            'background-color: rgb(23, 85, 218);'
            'image:url(./image/common_left_arrow.png);'
            'border-radius: 5px;'
            )
        self.prev_btn.setFixedSize(50, 50)
        self.next_btn = QPushButton('', self)
        self.next_btn.setStyleSheet(
            'color: white;'
            'background-color: rgb(23, 85, 218);'
            'image:url(./image/common_right_arrow.png);'
            'border-radius: 5px;'
            )
        self.next_btn.setFixedSize(50, 50)
        btn_hbox = QHBoxLayout()
        btn_hbox.addStretch(3)
        btn_hbox.addWidget(self.prev_btn)
        btn_hbox.addWidget(self.next_btn)

        # title document label
        main_label = QLabel(main_text, self)
        main_label.setAlignment(Qt.AlignCenter)
        main_label.setStyleSheet(
            'color: white;'
            'font: 25px Tahoma;'
            )
        sub_label = QLabel(sub_text, self)
        sub_label.setAlignment(Qt.AlignCenter)
        sub_label.setStyleSheet(
            'color: white;'
            'font: 25px Tahoma;'
            )
        
        # apply layout
        self.vbox = QVBoxLayout()
        self.vbox.addWidget(title_label)
        self.vbox.addLayout(btn_hbox)
        self.vbox.addWidget(main_label)
        self.vbox.addWidget(sub_label)

class HLineLayout(QFrame):
    def __init__(self, color=QColor(58, 134, 255)):
        super().__init__()

        self. initUI(color)
    
    def initUI(self, color):
        # dividing line
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Plain)
        self.setLineWidth(0)
        self.setMidLineWidth(3)
        self.setContentsMargins(0, 0, 0, 0)

        pal = self.palette()
        pal.setColor(QPalette.WindowText, color)
        self.setPalette(pal)

