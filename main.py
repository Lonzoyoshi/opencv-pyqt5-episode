import sys
import cv2
import numpy as np
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtCore import QThread, pyqtSignal
import matplotlib.pyplot as plt

from custom.stackedWidget import StackedWidget
from custom.treeView import FileSystemTreeView
from custom.listWidgets import FuncListWidget, UsedListWidget
from custom.graphicsView import GraphicsView


class MyApp(QMainWindow):
    def __init__(self):
        super(MyApp, self).__init__()
        self.tool_bar = self.addToolBar('工具栏')
        self.action_right_rotate = QAction(QIcon("icons/右旋转.png"), "向右旋转90", self)
        self.action_left_rotate = QAction(QIcon("icons/左旋转.png"), "向左旋转90°", self)
        self.action_histogram = QAction(QIcon("icons/直方图.png"), "直方图", self)
        self.action_spectrum = QAction(QIcon("icons/spectrum.png"),"频谱图",self)
        self.action_phase = QAction(QIcon("icons/phase.png"),"相位谱图",self)
        self.action_right_rotate.triggered.connect(self.right_rotate)
        self.action_left_rotate.triggered.connect(self.left_rotate)
        self.action_histogram.triggered.connect(self.histogram)
        self.action_spectrum.triggered.connect(self.spectrum)
        self.action_phase.triggered.connect(self.phase)
        self.tool_bar.addActions((self.action_left_rotate, self.action_right_rotate, self.action_histogram,
                                  self.action_spectrum, self.action_phase))     #  新增按钮频谱 相位
        self.useListWidget = UsedListWidget(self)
        self.funcListWidget = FuncListWidget(self)
        self.stackedWidget = StackedWidget(self)
        self.fileSystemTreeView = FileSystemTreeView(self)
        self.graphicsView = GraphicsView(self)

        self.dock_file = QDockWidget(self)
        self.dock_file.setWidget(self.fileSystemTreeView)
        self.dock_file.setTitleBarWidget(QLabel('目录'))
        self.dock_file.setFeatures(QDockWidget.NoDockWidgetFeatures)

        self.dock_func = QDockWidget(self)
        self.dock_func.setWidget(self.funcListWidget)
        self.dock_func.setTitleBarWidget(QLabel('图像操作'))
        self.dock_func.setFeatures(QDockWidget.NoDockWidgetFeatures)

        self.dock_used = QDockWidget(self)
        self.dock_used.setWidget(self.useListWidget)
        self.dock_used.setTitleBarWidget(QLabel('已选操作'))
        self.dock_used.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.dock_used.setFeatures(QDockWidget.NoDockWidgetFeatures)

        self.dock_attr = QDockWidget(self)
        self.dock_attr.setWidget(self.stackedWidget)
        self.dock_attr.setTitleBarWidget(QLabel('属性'))
        self.dock_attr.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.dock_attr.close()

        self.setCentralWidget(self.graphicsView)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_file)
        self.addDockWidget(Qt.TopDockWidgetArea, self.dock_func)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_used)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_attr)

        self.setWindowTitle('Opencv图像处理(LonzoYoshi_Beta_1103)')
        self.setWindowIcon(QIcon('icons/main.png'))
        self.src_img = None
        self.cur_img = None

    def update_image(self):
        if self.src_img is None:
            return
        img = self.process_image()
        self.cur_img = img
        self.graphicsView.update_image(img)

    def change_image(self, img):
        self.src_img = img
        img = self.process_image()
        self.cur_img = img
        self.graphicsView.change_image(img)

    def process_image(self):
        img = self.src_img.copy()
        for i in range(self.useListWidget.count()):
            img = self.useListWidget.item(i)(img)
        return img

    def right_rotate(self):
        self.graphicsView.rotate(90)

    def left_rotate(self):
        self.graphicsView.rotate(-90)

    def histogram(self):
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            histr = cv2.calcHist([self.cur_img], [i], None, [256], [0, 256])
            histr = histr.flatten()
            plt.plot(range(256), histr, color=col)
            plt.xlim([0, 256])
        plt.show()

    def spectrum(self):
        if self.cur_img is not None:
            cur_img = cv2.cvtColor(self.cur_img, cv2.COLOR_BGR2RGB)
            b_channel, g_channel, r_channel = cv2.split(cur_img)
            mag_spectra = []
            for channel in [b_channel, g_channel, r_channel]:
                img_float = np.float32(channel)
                f = cv2.dft(img_float, flags=cv2.DFT_COMPLEX_OUTPUT)
                fshift = np.fft.fftshift(f)
                mag, _ = cv2.cartToPolar(fshift[:, :, 0], fshift[:, :, 1])
                mag += 1e-10  # 避免对数为零
                mag = 20 * np.log(mag)
                mag_normalized = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                mag_spectra.append(mag_normalized)
            magnitude_image = cv2.merge(mag_spectra)
            plt.imshow(magnitude_image)
            plt.xticks([])  # 去掉坐标轴刻度
            plt.yticks([])
            plt.title('Magnitude Spectrum of Color Image')
            plt.show()
        else:
            print("图像加载失败，请检查路径和格式。")

    def phase(self):
        if self.cur_img is not None:
            cur_img = cv2.cvtColor(self.cur_img, cv2.COLOR_BGR2RGB)
            b_channel, g_channel, r_channel = cv2.split(cur_img)
            phase_spectra = []
            for channel in [b_channel, g_channel, r_channel]:
                img_float = np.float32(channel)
                f = cv2.dft(img_float, flags=cv2.DFT_COMPLEX_OUTPUT)
                fshift = np.fft.fftshift(f)
                _, phase = cv2.cartToPolar(fshift[:, :, 0], fshift[:, :, 1])
                # 将相位映射到0-255之间
                phase_normalized = cv2.normalize(phase, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                phase_spectra.append(phase_normalized)
            phase_image = cv2.merge(phase_spectra)
            plt.imshow(phase_image)
            plt.xticks([])  # 去掉坐标轴刻度
            plt.yticks([])
            plt.title('Phase Spectrum of Color Image')
            plt.show()
        else:
            print("图像加载失败，请检查路径和格式。")




if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(open('./custom/styleSheet.qss', encoding='utf-8').read())
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
