import cv2
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *


class TableWidget(QTableWidget):
    def __init__(self, parent=None):
        super(TableWidget, self).__init__(parent=parent)
        self.mainwindow = parent
        self.setShowGrid(True)  # 显示网格
        self.setAlternatingRowColors(True)  # 隔行显示颜色
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.horizontalHeader().setVisible(False)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().sectionResizeMode(QHeaderView.Stretch)
        self.verticalHeader().sectionResizeMode(QHeaderView.Stretch)
        self.horizontalHeader().setStretchLastSection(True)
        self.setFocusPolicy(Qt.NoFocus)

    def signal_connect(self):
        for spinbox in self.findChildren(QSpinBox):
            spinbox.valueChanged.connect(self.update_item)
        for doublespinbox in self.findChildren(QDoubleSpinBox):
            doublespinbox.valueChanged.connect(self.update_item)
        for combox in self.findChildren(QComboBox):
            combox.currentIndexChanged.connect(self.update_item)
        for checkbox in self.findChildren(QCheckBox):
            checkbox.stateChanged.connect(self.update_item)


    def update_item(self):
        param = self.get_params()
        self.mainwindow.useListWidget.currentItem().update_params(param)
        self.mainwindow.update_image()

    def update_params(self, param=None):
        for key in param.keys():
            box = self.findChild(QWidget, name=key)
            if isinstance(box, QSpinBox) or isinstance(box, QDoubleSpinBox):
                box.setValue(param[key])
            elif isinstance(box, QComboBox):
                box.setCurrentIndex(param[key])
            elif isinstance(box, QCheckBox):
                box.setChecked(param[key])

    def get_params(self):
        param = {}
        for spinbox in self.findChildren(QSpinBox):
            param[spinbox.objectName()] = spinbox.value()
        for doublespinbox in self.findChildren(QDoubleSpinBox):
            param[doublespinbox.objectName()] = doublespinbox.value()
        for combox in self.findChildren(QComboBox):
            param[combox.objectName()] = combox.currentIndex()
        for combox in self.findChildren(QCheckBox):
            param[combox.objectName()] = combox.isChecked()
        return param


class GrayingTableWidget(TableWidget):
    def __init__(self, parent=None):
        super(GrayingTableWidget, self).__init__(parent=parent)


class FilterTabledWidget(TableWidget):
    def __init__(self, parent=None):
        super(FilterTabledWidget, self).__init__(parent=parent)

        self.kind_comBox = QComboBox()
        self.kind_comBox.addItems(['均值滤波', '高斯滤波', '中值滤波','Box滤波',
                                   '低通滤波','高通滤波','增强','同态滤波','陷波滤波'])
        self.kind_comBox.setObjectName('kind')

        self.ksize_spinBox = QSpinBox()
        self.ksize_spinBox.setObjectName('ksize')
        self.ksize_spinBox.setMinimum(1)
        self.ksize_spinBox.setSingleStep(2)

        self.cutoff_spinbox = QDoubleSpinBox()
        self.cutoff_spinbox.setObjectName('cutoff')
        self.cutoff_spinbox.setMinimum(0.01)
        self.cutoff_spinbox.setSingleStep(0.01)

        self.cutoff_box = QSpinBox()
        self.cutoff_box.setObjectName('cutoffSize')
        self.cutoff_box.setMinimum(10)
        self.cutoff_box.setSingleStep(10)

        self.notch_x_box = QSpinBox()
        self.notch_x_box.setObjectName('notch_x')
        self.notch_x_box.setMinimum(-60)
        self.notch_x_box.setMaximum(60)
        self.notch_x_box.setSingleStep(10)

        self.notch_y_box = QSpinBox()
        self.notch_y_box.setObjectName('notch_y')
        self.notch_y_box.setMinimum(-60)
        self.notch_y_box.setMaximum(60)
        self.notch_y_box.setSingleStep(10)

        self.setColumnCount(2)
        self.setRowCount(6)
        self.setItem(0, 0, QTableWidgetItem('类型'))
        self.setCellWidget(0, 1, self.kind_comBox)
        self.setItem(1, 0, QTableWidgetItem('核大小'))
        self.setCellWidget(1, 1, self.ksize_spinBox)
        self.setItem(2, 0, QTableWidgetItem('截止频率(0,0.5]'))
        self.setCellWidget(2,1,self.cutoff_spinbox)
        self.setItem(3,0,QTableWidgetItem('Cutoff尺寸(或陷波直径)'))
        self.setCellWidget(3,1, self.cutoff_box)
        self.setItem(4, 0, QTableWidgetItem('陷波x坐标'))
        self.setCellWidget(4, 1, self.notch_x_box)
        self.setItem(5, 0, QTableWidgetItem('陷波x坐标'))
        self.setCellWidget(5, 1, self.notch_y_box)


        self.signal_connect()


class MorphTabledWidget(TableWidget):
    def __init__(self, parent=None):
        super(MorphTabledWidget, self).__init__(parent=parent)

        self.op_comBox = QComboBox()
        self.op_comBox.addItems(['腐蚀操作', '膨胀操作', '开操作', '闭操作', '梯度操作', '顶帽操作', '黑帽操作'])
        self.op_comBox.setObjectName('op')

        self.ksize_spinBox = QSpinBox()
        self.ksize_spinBox.setMinimum(1)
        self.ksize_spinBox.setSingleStep(2)
        self.ksize_spinBox.setObjectName('ksize')

        self.kshape_comBox = QComboBox()
        self.kshape_comBox.addItems(['方形', '十字形', '椭圆形'])
        self.kshape_comBox.setObjectName('kshape')

        self.setColumnCount(2)
        self.setRowCount(3)
        self.setItem(0, 0, QTableWidgetItem('类型'))
        self.setCellWidget(0, 1, self.op_comBox)
        self.setItem(1, 0, QTableWidgetItem('核大小'))
        self.setCellWidget(1, 1, self.ksize_spinBox)
        self.setItem(2, 0, QTableWidgetItem('核形状'))
        self.setCellWidget(2, 1, self.kshape_comBox)
        self.signal_connect()


class GradTabledWidget(TableWidget):
    def __init__(self, parent=None):
        super(GradTabledWidget, self).__init__(parent=parent)

        self.kind_comBox = QComboBox()
        self.kind_comBox.addItems(['Sobel算子', 'Scharr算子', 'Laplacian算子'])
        self.kind_comBox.setObjectName('kind')

        self.ksize_spinBox = QSpinBox()
        self.ksize_spinBox.setMinimum(1)
        self.ksize_spinBox.setSingleStep(2)
        self.ksize_spinBox.setObjectName('ksize')

        self.dx_spinBox = QSpinBox()
        self.dx_spinBox.setMaximum(1)
        self.dx_spinBox.setMinimum(0)
        self.dx_spinBox.setSingleStep(1)
        self.dx_spinBox.setObjectName('dx')

        self.dy_spinBox = QSpinBox()
        self.dy_spinBox.setMaximum(1)
        self.dy_spinBox.setMinimum(0)
        self.dy_spinBox.setSingleStep(1)
        self.dy_spinBox.setObjectName('dy')

        self.setColumnCount(2)
        self.setRowCount(4)

        self.setItem(0, 0, QTableWidgetItem('类型'))
        self.setCellWidget(0, 1, self.kind_comBox)
        self.setItem(1, 0, QTableWidgetItem('核大小'))
        self.setCellWidget(1, 1, self.ksize_spinBox)
        self.setItem(2, 0, QTableWidgetItem('x方向'))
        self.setCellWidget(2, 1, self.dx_spinBox)
        self.setItem(3, 0, QTableWidgetItem('y方向'))
        self.setCellWidget(3, 1, self.dy_spinBox)

        self.signal_connect()


class ThresholdTableWidget(TableWidget):
    def __init__(self, parent=None):
        super(ThresholdTableWidget, self).__init__(parent=parent)

        self.thresh_spinBox = QSpinBox()
        self.thresh_spinBox.setObjectName('thresh')
        self.thresh_spinBox.setMaximum(255)
        self.thresh_spinBox.setMinimum(0)
        self.thresh_spinBox.setSingleStep(1)

        self.maxval_spinBox = QSpinBox()
        self.maxval_spinBox.setObjectName('maxval')
        self.maxval_spinBox.setMaximum(255)
        self.maxval_spinBox.setMinimum(0)
        self.maxval_spinBox.setSingleStep(1)

        self.method_comBox = QComboBox()
        self.method_comBox.addItems(['二进制阈值化', '反二进制阈值化', '截断阈值化', '阈值化为0', '反阈值化为0', '大津算法'])
        self.method_comBox.setObjectName('method')

        self.setColumnCount(2)
        self.setRowCount(3)

        self.setItem(0, 0, QTableWidgetItem('类型'))
        self.setCellWidget(0, 1, self.method_comBox)
        self.setItem(1, 0, QTableWidgetItem('阈值'))
        self.setCellWidget(1, 1, self.thresh_spinBox)
        self.setItem(2, 0, QTableWidgetItem('最大值'))
        self.setCellWidget(2, 1, self.maxval_spinBox)

        self.signal_connect()


class EdgeTableWidget(TableWidget):
    def __init__(self, parent=None):
        super(EdgeTableWidget, self).__init__(parent=parent)

        self.thresh1_spinBox = QSpinBox()
        self.thresh1_spinBox.setMinimum(0)
        self.thresh1_spinBox.setMaximum(255)
        self.thresh1_spinBox.setSingleStep(1)
        self.thresh1_spinBox.setObjectName('thresh1')

        self.thresh2_spinBox = QSpinBox()
        self.thresh2_spinBox.setMinimum(0)
        self.thresh2_spinBox.setMaximum(255)
        self.thresh2_spinBox.setSingleStep(1)
        self.thresh2_spinBox.setObjectName('thresh2')

        self.setColumnCount(2)
        self.setRowCount(2)

        self.setItem(0, 0, QTableWidgetItem('阈值1'))
        self.setCellWidget(0, 1, self.thresh1_spinBox)
        self.setItem(1, 0, QTableWidgetItem('阈值2'))
        self.setCellWidget(1, 1, self.thresh2_spinBox)
        self.signal_connect()


class ContourTableWidget(TableWidget):
    def __init__(self, parent=None):
        super(ContourTableWidget, self).__init__(parent=parent)

        self.bbox_comBox = QComboBox()
        self.bbox_comBox.addItems(['正常轮廓', '外接矩形', '最小外接矩形', '最小外接圆'])
        self.bbox_comBox.setObjectName('bbox')

        self.mode_comBox = QComboBox()
        self.mode_comBox.addItems(['外轮廓', '轮廓列表', '外轮廓与内孔', '轮廓等级树'])
        self.mode_comBox.setObjectName('mode')

        self.method_comBox = QComboBox()
        self.method_comBox.addItems(['无近似', '简易近似'])
        self.method_comBox.setObjectName('method')

        self.setColumnCount(2)
        self.setRowCount(3)

        self.setItem(0, 0, QTableWidgetItem('轮廓模式'))
        self.setCellWidget(0, 1, self.mode_comBox)
        self.setItem(1, 0, QTableWidgetItem('轮廓近似'))
        self.setCellWidget(1, 1, self.method_comBox)
        self.setItem(2, 0, QTableWidgetItem('边界模式'))
        self.setCellWidget(2, 1, self.bbox_comBox)
        self.signal_connect()


class EqualizeTableWidget(TableWidget):
    def __init__(self, parent=None):
        super(EqualizeTableWidget, self).__init__(parent=parent)
        self.red_checkBox = QCheckBox()
        self.red_checkBox.setObjectName('red')
        self.red_checkBox.setTristate(False)
        self.blue_checkBox = QCheckBox()
        self.blue_checkBox.setObjectName('blue')
        self.blue_checkBox.setTristate(False)
        self.green_checkBox = QCheckBox()
        self.green_checkBox.setObjectName('green')
        self.green_checkBox.setTristate(False)

        self.setColumnCount(2)
        self.setRowCount(3)

        self.setItem(0, 0, QTableWidgetItem('R通道'))
        self.setCellWidget(0, 1, self.red_checkBox)
        self.setItem(1, 0, QTableWidgetItem('G通道'))
        self.setCellWidget(1, 1, self.green_checkBox)
        self.setItem(2, 0, QTableWidgetItem('B通道'))
        self.setCellWidget(2, 1, self.blue_checkBox)
        self.signal_connect()


class HoughLineTableWidget(TableWidget):
    def __init__(self, parent=None):
        super(HoughLineTableWidget, self).__init__(parent=parent)

        self.thresh_spinBox = QSpinBox()
        self.thresh_spinBox.setMinimum(0)
        self.thresh_spinBox.setSingleStep(1)
        self.thresh_spinBox.setObjectName('thresh')

        self.min_length_spinBox = QSpinBox()
        self.min_length_spinBox.setMinimum(0)
        self.min_length_spinBox.setSingleStep(1)
        self.min_length_spinBox.setObjectName('min_length')

        self.max_gap_spinbox = QSpinBox()
        self.max_gap_spinbox.setMinimum(0)
        self.max_gap_spinbox.setSingleStep(1)
        self.max_gap_spinbox.setObjectName('max_gap')

        self.setColumnCount(2)
        self.setRowCount(3)

        self.setItem(0, 0, QTableWidgetItem('交点阈值'))
        self.setCellWidget(0, 1, self.thresh_spinBox)
        self.setItem(1, 0, QTableWidgetItem('最小长度'))
        self.setCellWidget(1, 1, self.min_length_spinBox)
        self.setItem(2, 0, QTableWidgetItem('最大间距'))
        self.setCellWidget(2, 1, self.max_gap_spinbox)
        self.signal_connect()

class LightTableWidget(TableWidget):
    def __init__(self, parent=None):
        super(LightTableWidget, self).__init__(parent=parent)

        self.alpha_spinBox = QDoubleSpinBox()
        self.alpha_spinBox.setMinimum(0)
        self.alpha_spinBox.setMaximum(3)
        self.alpha_spinBox.setSingleStep(0.1)
        self.alpha_spinBox.setObjectName('alpha')

        self.beta_spinbox = QSpinBox()
        self.beta_spinbox.setMinimum(0)
        self.beta_spinbox.setSingleStep(1)
        self.beta_spinbox.setObjectName('beta')

        self.setColumnCount(2)
        self.setRowCount(2)

        self.setItem(0, 0, QTableWidgetItem('alpha'))
        self.setCellWidget(0, 1, self.alpha_spinBox)
        self.setItem(1, 0, QTableWidgetItem('beta'))
        self.setCellWidget(1, 1, self.beta_spinbox)
        self.signal_connect()


class GammaITabelWidget(TableWidget):
    def __init__(self, parent=None):
        super(GammaITabelWidget, self).__init__(parent=parent)
        self.gamma_spinbox = QDoubleSpinBox()
        self.gamma_spinbox.setMinimum(0)
        self.gamma_spinbox.setSingleStep(0.1)
        self.gamma_spinbox.setObjectName('gamma')

        self.setColumnCount(2)
        self.setRowCount(1)

        self.setItem(0, 0, QTableWidgetItem('gamma'))
        self.setCellWidget(0, 1, self.gamma_spinbox)
        self.signal_connect()


class MatchingWidget(TableWidget):
    def __init__(self, parent=None):
        super(MatchingWidget, self).__init__(parent=parent)
        self.selection_box = QDialogButtonBox()
        self.selection_box.setObjectName('pushed1')
        self.selected_img = None

        self.select_button = QPushButton('选择图片')
        self.select_button.clicked.connect(self.refImg)
        self.select_button.setObjectName('pushed2')

        self.setColumnCount(2)
        self.setRowCount(1)
        self.setItem(0, 0, QTableWidgetItem('参考图'))
        self.setCellWidget(0, 1, self.selection_box)
        self.selection_box.addButton(self.select_button,QDialogButtonBox.ActionRole)

        self.signal_connect()

    def signal_connect(self):
        for spinbox in self.findChildren(QSpinBox):
            spinbox.valueChanged.connect(self.update_item)
        for doublespinbox in self.findChildren(QDoubleSpinBox):
            doublespinbox.valueChanged.connect(self.update_item)
        for combox in self.findChildren(QComboBox):
            combox.currentIndexChanged.connect(self.update_item)
        for checkbox in self.findChildren(QCheckBox):
            checkbox.stateChanged.connect(self.update_item)
        for button in self.findChildren(QPushButton):
            button.clicked.connect(self.update_item)
        current_item = self.mainwindow.useListWidget.currentItem()
        if current_item is not None:
            current_item.update_params()
        else:
            print("None")
    def update_params(self, param=None):
        for key in param.keys():
            box = self.findChild(QWidget, name=key)
            if isinstance(box, QSpinBox) or isinstance(box, QDoubleSpinBox):
                box.setValue(param[key])
            elif isinstance(box, QComboBox):
                box.setCurrentIndex(param[key])
            elif isinstance(box, QCheckBox):
                box.setChecked(param[key])
            elif isinstance(box, QPushButton):
                button_state = param[key]
                if isinstance(button_state, bool):
                    box.setChecked(button_state)

    def get_params(self):
        param = {}
        for spinbox in self.findChildren(QSpinBox):
            param[spinbox.objectName()] = spinbox.value()
        for doublespinbox in self.findChildren(QDoubleSpinBox):
            param[doublespinbox.objectName()] = doublespinbox.value()
        for combox in self.findChildren(QComboBox):
            param[combox.objectName()] = combox.currentIndex()
        for combox in self.findChildren(QCheckBox):
            param[combox.objectName()] = combox.isChecked()
        for button in self.findChildren(QPushButton):
            param[button.objectName()] = button.isChecked()
            if self.selected_img is not None:
                param['refImg'] = self.selected_img
        return param

    def refImg(self):
        # 打开文件对话框让用户选择图片
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "选择图片", "",
                                                   "Images (*.png *.xpm *.jpg *.jpeg);;All Files (*)", options=options)
        if file_name:
            self.selected_img = file_name


class NoiseWidget(TableWidget):
    def __init__(self, parent=None):
        super(NoiseWidget, self).__init__(parent=parent)
        self.noise_comBox = QComboBox()
        self.noise_comBox.addItems(['高斯噪声', '椒盐噪声', '动态模糊'])
        self.noise_comBox.setObjectName('noise')

        self.mean_spinbox = QSpinBox()
        self.mean_spinbox.setObjectName('mean')
        self.mean_spinbox.setMinimum(0)
        self.mean_spinbox.setSingleStep(5)

        self.variance_spinbox = QSpinBox()
        self.variance_spinbox.setObjectName('variance')
        self.variance_spinbox.setMinimum(0)
        self.variance_spinbox.setSingleStep(100)

        self.salt_spinbox = QDoubleSpinBox()
        self.salt_spinbox.setObjectName('saltProb')
        self.salt_spinbox.setMinimum(0.0)
        self.salt_spinbox.setMaximum(1.0)
        self.salt_spinbox.setSingleStep(0.1)

        self.pepper_spinbox = QDoubleSpinBox()
        self.pepper_spinbox.setObjectName('pepperProb')
        self.pepper_spinbox.setMinimum(0.0)
        self.pepper_spinbox.setMaximum(1.0)
        self.pepper_spinbox.setSingleStep(0.1)

        self.ksize_spinBox = QSpinBox()
        self.ksize_spinBox.setObjectName('ksize')
        self.ksize_spinBox.setMinimum(1)
        self.ksize_spinBox.setSingleStep(2)

        self.setColumnCount(2)
        self.setRowCount(6)
        self.setItem(0, 0, QTableWidgetItem('类型'))
        self.setCellWidget(0, 1, self.noise_comBox)
        self.setItem(1, 0, QTableWidgetItem('高斯均值'))
        self.setCellWidget(1, 1, self.mean_spinbox)
        self.setItem(2, 0, QTableWidgetItem('高斯方差'))
        self.setCellWidget(2, 1, self.variance_spinbox)
        self.setItem(3, 0, QTableWidgetItem('盐率'))
        self.setCellWidget(3, 1, self.salt_spinbox)
        self.setItem(4, 0, QTableWidgetItem('椒率'))
        self.setCellWidget(4, 1, self.pepper_spinbox)
        self.setItem(5, 0, QTableWidgetItem('核大小'))
        self.setCellWidget(5, 1, self.ksize_spinBox)
