import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QWidget, QDesktopWidget
from PyQt5.QtGui import * 
from PyQt5.QtCore import QTimer,QDateTime
import datetime
import time


# we catch the global time from an NTP server 
# and calculate the offset to our local system time
import ntplib
ntp_client = ntplib.NTPClient()
sys_time = time.time()
response = ntp_client.request('europe.pool.ntp.org', version=3)
system_time_datetime = datetime.datetime.fromtimestamp(response.tx_time)
timestamp_offset = response.orig_time - response.tx_time

def get_system_time():
	#response = ntp_client.request('europe.pool.ntp.org', version=3)
	#system_time_datetime = datetime.datetime.fromtimestamp(response.tx_time)
	return datetime.datetime.fromtimestamp(time.time() - timestamp_offset)
	#return system_time_datetime

class MainWindow(QWidget):

    def __init__(self, system_func):
        super(MainWindow, self).__init__()
        self.system_func = system_func

        self.layout = QVBoxLayout()
        self.label = QLabel(self.system_func().strftime("%H:%M:%S:%f"))
        self.label.setFont(QFont('Arial', 100))

        self.timer=QTimer()
        self.timer.timeout.connect(self.showTime)
        self.timer.start(1)

        self.layout.addWidget(self.label)
        self.setWindowTitle("My Own Title")
        self.setLayout(self.layout)

        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        # qr.moveCenter(cp)
        self.move(qr.bottomLeft())

    def showTime(self):
        self.label.setText(self.system_func().strftime("%H:%M:%S:%f"))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mw = MainWindow(get_system_time)
    mw.show()
    sys.exit(app.exec_())