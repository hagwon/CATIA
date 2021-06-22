import os
import sys
import math

from PyQt5 import QtWidgets, QtGui, QtCore  # py -m pip install PyQt5
import widgets
import robotics
import comport
import xylobot

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, title_name, icon_path, size):
        super().__init__()
        self.title = title_name
        self.icon = QtGui.QIcon(icon_path)
        self.size = size

        self.link = [64, 44, 210]
        self.kinematics = robotics.inversKinematics(self.link[0], self.link[1], self.link[2])
        self.comport = comport.ComPort()
        self.xyPara = xylobot.Parameters()
        
        self.initUI()
        self.openXylobot()
    
    def initUI(self):
        # UI 화면을 생성한다.
        self.stack_widget = QtWidgets.QStackedWidget(self)
        self.stack_widget.addWidget(widgets.InputDimensionWidget(self))
        self.stack_widget.addWidget(widgets.MoveRobotWidget(self))
        self.setCentralWidget(self.stack_widget)

        # 배경화면을 생성한다.
        back_img = QtGui.QImage("./images/background_0.png")
        background = QtGui.QPalette()
        background.setBrush(10, QtGui.QBrush(back_img))
        self.setPalette(background)
        
        # 상태창을 생성한다.
        self.statusBar().showMessage('Ready')
        self.statusBar().setStyleSheet(
            'color: rgb(54, 134, 255);'
            'font: bold 20px Tahoma;'
            )
        
        self.setWindowTitle(self.title)
        self.setWindowIcon(self.icon)

        # 화면 중앙에서 실행한다.
        self.setFixedSize(self.size[0], self.size[1])
        #self.resize(self.size[0], self.size[1])
        qr = self.frameGeometry()
        cp = QtWidgets.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def openXylobot(self):
        # 실로봇에게 할당된 컴포트를 연결한다.
        self.comport.open()
        if self.comport.is_connected == True:
            # 실로봇의 토크를 켠다.
            ON = xylobot.Protocol.TORQUE_ON
            self.comport.writeTorqueStatus(ON, ON, ON)
            # 실로봇의 회전속도를 200으로 설정한다.
            self.comport.writeSpeed(200, 200, 200)
            # 상태창을 '연결' 상태로 변경한다.
            self.statusBar().setStyleSheet(
                'color: rgb(54, 134, 255);'
                'font: bold 20px Tahoma;'
                )
            self.statusBar().showMessage('Com port : Connected')
        else:
            # 상태창을 '연결해제' 상태로 변경한다.
            self.statusBar().setStyleSheet(
                'color: red;'
                'font: bold 20px Tahoma;'
                )
            self.statusBar().showMessage('Com port : Connection failure')
    
    def closeEvent(self, event):
        self.comport.close()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    form = MainWindow("Invers Kinematics Control v1.0", "./images/logo.png", (1600, 900))
    form.show()
    sys.exit(app.exec_())