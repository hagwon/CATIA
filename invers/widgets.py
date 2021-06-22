import math
from PyQt5 import QtWidgets, QtGui, QtCore

class Title(QtWidgets.QFrame):

    def __init__(self, title='Untitled'):

        super().__init__()
        
        #Top 이미지르르 생성한다.
        main_title = Image('./images/main_title.png')
        
        # 타이틀 글자를 생성한다.
        title_label = QtWidgets.QLabel(title, self)
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        title_label.setStyleSheet(
            'color: #000000;'
            'font: bold 50px Tahoma;'
            )

        # '이전' 버튼을 생성한다.
        self.prev_btn = QtWidgets.QPushButton('', self)
        self.prev_btn.setFixedSize(50, 50)
        self.prev_btn.setStyleSheet(
            'color: white;'
            'background-color: rgb(54, 134, 255);'
            'image:url(./images/enable_left_arrow.png);'
            'border-radius: 5px;'
            )

        # '다음' 버튼을 생성한다.
        self.next_btn = QtWidgets.QPushButton('', self)
        self.next_btn.setFixedSize(50, 50)
        self.next_btn.setStyleSheet(
            'color: white;'
            'background-color: rgb(54, 134, 255);'
            'image:url(./images/enable_right_arrow.png);'
            'border-radius: 5px;'
            )

        btn_hbox = QtWidgets.QHBoxLayout()
        btn_hbox.addStretch(3)
        btn_hbox.addWidget(self.prev_btn)
        btn_hbox.addWidget(self.next_btn)
        
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(main_title.image)
        self.layout.addLayout(btn_hbox)
        self.layout.addWidget(title_label)
    
    def setButtonState(self, index, enabled):
        if index == 0:
            if enabled == False:
                # '이전' 버튼을 비활성화 시킨다.
                self.prev_btn.setStyleSheet(            
                    'color: rgb(200, 200, 200);'
                    'background-color: rgb(25, 110, 210);'
                    'image:url(./images/disable_left_arrow.png);'
                    'border-radius: 5px;'
                    )
            else:
                # '이전' 버튼을 비활성화 시킨다.
                self.prev_btn.setStyleSheet(
                'color: white;'
                'background-color: rgb(54, 134, 255);'
                'image:url(./images/enable_left_arrow.png);'
                'border-radius: 5px;'
                )
        else:
            if enabled == False:
                # '다음' 버튼을 비활성화 시킨다.
                self.next_btn.setStyleSheet(
                    'color: rgb(200, 200, 200);'
                    'background-color: rgb(25, 110, 210);'
                    'image:url(./images/disable_right_arrow.png);'
                    'border-radius: 5px;'
                    )
            else:
                # '다음' 버튼을 활성화 시킨다.
                self.next_btn.setStyleSheet(
                    'color: white;'
                    'background-color: rgb(54, 134, 255);'
                    'image:url(./images/inable_right_arrow.png);'
                    'border-radius: 5px;'
                    )

class Image(QtWidgets.QFrame):

    def __init__(self, path=''):
        super().__init__()
        # 이미지를 생성한다.
        pixmap = QtGui.QPixmap(path)
        self.image = QtWidgets.QLabel()
        self.image.setPixmap(pixmap)

class InputLabel(QtWidgets.QFrame):

    def __init__(self, label_text, textbox_text):
        super().__init__()
        # 버튼을 생성한다.
        label = QtWidgets.QPushButton(label_text, self)
        label.setStyleSheet(
            'color: rgb(54, 134, 255);'
            'background-color: transparent;'
            'border-radius: 5px;'
            'font: bold 30px Tahoma;'
            )
        label.setFixedWidth(100)
        self.textbox = QtWidgets.QPushButton(textbox_text, self)
        self.textbox.setStyleSheet(
            'color: rgb(54, 134, 255);'
            'background-color: transparent;'
            'border-radius: 5px;'
            'font: bold 30px Tahoma;'
            'text-align: right;'
            )
        self.textbox.setFixedWidth(100)
                
        self.layout = QtWidgets.QHBoxLayout()
        self.layout.addWidget(label)    
        self.layout.addWidget(self.textbox)

# 수치 입력 화면 클레스
class InputDimensionWidget(QtWidgets.QWidget):

    def __init__(self, parent):
        super().__init__()
        self.main = parent

        self.initUI()
    
    def initUI(self):
        # 타이틀을 생성한다.
        title = Title(title='Input Dimensions')
        title.setButtonState(0, False)
        title.next_btn.clicked.connect(self.clickNextButton)

        # 치수 이미지를 생성한다.
        d_img = Image('./images/background_1.png')

        # 치수 입력 글자를 InputLabel Widget을 이용하여 생성한다.
        unit = InputLabel('Unit', 'mm')
        self.length = []
        self.length.append(InputLabel('L1 : ', '{0}'.format(self.main.link[0])))
        self.length.append(InputLabel('L2 : ', '{0}'.format(self.main.link[1])))    
        self.length.append(InputLabel('L3 : ', '{0}'.format(self.main.link[2])))
        self.length[0].textbox.clicked.connect(self.clickButton_L1)
        self.length[1].textbox.clicked.connect(self.clickButton_L2)
        self.length[2].textbox.clicked.connect(self.clickButton_L3)

        dimension_vbox = QtWidgets.QVBoxLayout()
        dimension_vbox.addLayout(unit.layout)
        dimension_vbox.addLayout(self.length[0].layout)
        dimension_vbox.addLayout(self.length[1].layout)
        dimension_vbox.addLayout(self.length[2].layout)

        main_hbox = QtWidgets.QHBoxLayout()
        main_hbox.addStretch(1)
        main_hbox.addWidget(d_img.image)
        main_hbox.addLayout(dimension_vbox)
        main_hbox.addStretch(1)

        main_vbox = QtWidgets.QVBoxLayout()
        main_vbox.addLayout(title.layout)
        main_vbox.addStretch(1)
        main_vbox.addLayout(main_hbox)
        main_vbox.addStretch(1)

        self.setLayout(main_vbox)
        self.setWindowTitle("Input Dimensions")

    def clickNextButton(self):
        self.main.stack_widget.setCurrentIndex(1)
        
    def clickButton_L1(self):
        self.showInputDialog(0)

    def clickButton_L2(self):
        self.showInputDialog(1)
        
    def clickButton_L3(self):
        self.showInputDialog(2)
        
    def showInputDialog(self, index):
        error = True
        text, ok = QtWidgets.QInputDialog.getText(self, 'L{0} Dimension'.format(index + 1), 'mm:')

        if ok == True:
            # 'OK' 버튼을 클릭 시 숫자 입력인지 확인한다.
            for i in range(len(text)):
                check = ord(text[i])
                if (48 <= check and check <= 57):
                    error = False

        if error == False:
            # 숫자 입력이 확인되면 입력값을 적용한다.
            self.length[index].textbox.setText(text)
            self.main.link[index] = int(text)
            self.main.kinematics.inputDimension(index, self.main.link[index])
        else:
            # 숫자가 아니면 실패 메시지를 생성한다.
            QtWidgets.QMessageBox.warning(self, 'Failure', 'This text is not number.')

class MoveRobotWidget(QtWidgets.QWidget):

    def __init__(self, parent):
        super().__init__()
        self.main = parent
        self.draw_point = []

        self.x_rad = math.radians(0)
        self.y_rad = math.radians(90)
        #self.z_rad = math.radians(90)
        self.o_axis = [477, 770]
        #self.x1_axis = [self.o_axis[0] + 300 * math.cos(self.x_rad), self.o_axis[1] + 300 * math.sin(self.x_rad)]
        #self.x2_axis = [self.o_axis[0] - 300 * math.cos(self.x_rad), self.o_axis[1] - 300 * math.sin(self.x_rad)]
        #self.y_axis = [self.o_axis[0] + 320 * math.cos(self.y_rad), self.o_axis[1] - 320 * math.sin(self.y_rad)]
        #self.z_axis = [self.o_axis[0] + 300 * math.cos(self.z_rad), self.o_axis[1] - 300 * math.sin(self.z_rad)]      
        
        self.j0_pos = [self.o_axis[0], self.o_axis[1]]
        self.j1_pos = [self.o_axis[0], self.o_axis[1] - (self.main.link[1] * 2)]
        self.j2_pos = [self.o_axis[0] - 300, self.o_axis[1] - 400]
        self.j3_pos = [self.o_axis[0] - 400, self.o_axis[1] - 400]

        self.coordinates = [0, 230, 80]

        self.initUI()
    
    def initUI(self):
        # 타이틀을 생성한다.
        title = Title(title='Control Robot')
        title.setButtonState(1, False)
        title.prev_btn.clicked.connect(self.clickPrevButton)

        # 좌표 이미지를 생성한다.
        c_img = Image('./images/background_2-1.png')

        # 마우스 입력 틀을 생성한다.
        i_img =  Image('./images/background_2-2.png')
        # 마우스 좌표를 3축 좌표로 변환한다.
        self.setMouseTracking(False)

        # 좌표 라벨을 생성한다
        self.coord_x =  QtWidgets.QLabel('x : 230 mm', self)
        self.coord_y =  QtWidgets.QLabel('y : 000 mm', self)
        self.coord_z =  QtWidgets.QLabel('z : 080 mm', self)
        self.coord_x.setStyleSheet(
            'color: rgb(54, 134, 255);'
            'background-color: transparent;'
            'border-radius: 2px;'
            'font: bold 15px Tahoma;'
            )
        self.coord_y.setStyleSheet(
            'color: rgb(54, 134, 255);'
            'background-color: transparent;'
            'border-radius: 2px;'
            'font: bold 15px Tahoma;'
            )
        self.coord_z.setStyleSheet(
            'color: rgb(54, 134, 255);'
            'background-color: transparent;'
            'border-radius: 2px;'
            'font: bold 15px Tahoma;'
            )

        label_vbox = QtWidgets.QVBoxLayout()
        label_vbox.addWidget(self.coord_x)
        label_vbox.addWidget(self.coord_y)
        label_vbox.addWidget(self.coord_z)
        label_vbox.addStretch(1)

        main_hbox = QtWidgets.QHBoxLayout()
        main_hbox.addStretch(1)
        main_hbox.addWidget(c_img.image)
        main_hbox.addWidget(i_img.image)
        main_hbox.addLayout(label_vbox)
        main_hbox.addStretch(1)

        main_vbox = QtWidgets.QVBoxLayout()
        main_vbox.addLayout(title.layout)
        main_vbox.addStretch(1)
        main_vbox.addLayout(main_hbox)
        main_vbox.addStretch(1)

        self.setLayout(main_vbox)
        self.setWindowTitle("Control Robot")

    def makeJoint(self):
        joint = QtWidgets.QLabel(self)
        joint.setStyleSheet(
            'image:url(./images/joint.png);'
            'background-color: transparent;'
            )
        joint.setFixedSize(11, 11)

        return joint

    def clickPrevButton(self):
        self.draw_point.clear()
        self.update()  
        self.main.stack_widget.setCurrentIndex(0)

    def paintEvent(self, e):
        qp = QtGui.QPainter()
        qp.begin(self)
        # 마우스 경로를 표시한다.
        self.updateMousePath(qp)
        # 실로봇의 움직임을 표시한다.
        self.drawAxis(qp)
        qp.end()

    def mousePressEvent(self, e):
        if e.button() == QtCore.Qt.LeftButton:
            self.drawing = True
            self.last_point_x = e.x()
            self.last_point_y = e.y()
        
    def mouseMoveEvent(self, e):
        if self.drawing == True:
            point_x = e.x()
            point_y = e.y()

            if (112 <= point_x) and (point_x < 758):
                self.draw_point.append([self.last_point_x, self.last_point_y, point_x, point_y])
                self.coordinates[0] = int(round((point_x - 435) / 3))
                self.coordinates[1] = 230 + int(round((560 - point_y) / 10))
            elif (758 <= point_x) and (point_x < 1404):
                self.draw_point.append([self.last_point_x, self.last_point_y, point_x, point_y])
                self.coordinates[0] = int(round((point_x - 1081) / 3))
                self.coordinates[2] = 60 + int(round((560 - point_y) / 3))

            self.last_point_x = point_x
            self.last_point_y = point_y

            self.update()
    
    def mouseReleaseEvent(self, e):
        if e.button() == QtCore.Qt.LeftButton:
            self.drawing = False

    def updateMousePath(self, qp):
        for i in range(0, len(self.draw_point)):
            pixel = self.draw_point[i]
            qp.setPen(QtGui.QPen(QtGui.QColor(75, 217, 100), 2))
            qp.drawLine(pixel[0], pixel[1], pixel[2], pixel[3])

    def drawAxis(self, qp):
        scale_1 = 1.5
        scale_2 = 1.2

        # 역기구학을 계산한다.
        self.coord_x.setText('x : ' + format(self.coordinates[0], '03').format('03') + 'mm')
        self.coord_y.setText('y : ' + format(self.coordinates[1], '03') + 'mm')
        self.coord_z.setText('z : ' + format(self.coordinates[2], '03') + 'mm')
        self.main.kinematics.invers(self.coordinates[0], self.coordinates[1], self.coordinates[2])

        # 각도값을 포지션값으로 변환 후 로봇을 구동한다.
        pos = [0, 0, 0]
        pos[0] = round(self.main.xyPara.angleToPosition(0, self.main.kinematics.angle[0]))
        pos[1] = round(self.main.xyPara.angleToPosition(1, self.main.kinematics.angle[1]))
        pos[2] = round(self.main.xyPara.angleToPosition(2, self.main.kinematics.angle[2]))
        self.main.comport.writePosition(pos[0], pos[1], pos[2])
