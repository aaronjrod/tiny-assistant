import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLineEdit, QTextEdit, QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import subprocess
import os
import ollama
import pytesseract
import cv2
import re
import time
import threading

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tiny Assistant")

        # Create widgets
        #self.capture_button = QPushButton("Start Capture", self)
        self.pause_button = QPushButton("Resume", self)
        self.exit_button = QPushButton("Exit", self)
        self.output_textedit = QTextEdit(self)
        self.output_textedit.setReadOnly(True)
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)

        image_path = "clippy.png"
        self.image_label.setPixmap(QPixmap(image_path))

        # Layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.image_label)
        button_layout.addWidget(self.pause_button)
        button_layout.addWidget(self.exit_button)
        button_layout.setContentsMargins(0, 0, 0, 0)  # No margins between widgets

        layout = QVBoxLayout()
        #layout.addWidget(self.capture_button)
        layout.addLayout(button_layout)
        layout.addWidget(self.output_textedit)
        self.setLayout(layout)

        # Connect button clicks to functions
        #self.capture_button.clicked.connect(self.start_capture)
        self.pause_button.clicked.connect(self.pause_capture)
        self.exit_button.clicked.connect(self.exit_app)
    
        # Initialize capture loop
        self.capturing = False
        self.last_capture_time = time.time()
        #self.capture_loop()

    def start_capture(self):
        self.capturing = True
        self.capture_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.exit_button.setEnabled(True)
        self.capture_loop()

    def pause_capture(self):
        if self.capturing:
            self.capturing = False
            self.pause_button.setText("Resume")
        else:
            self.capturing = True
            self.pause_button.setText("Pause")
            self.capture_loop()

    def exit_app(self):
        self.capturing = False
        self.close()

    def capture_loop(self):
        while self.capturing:
            self.capture_screenshot()
            QApplication.processEvents()
            if self.capturing:
                time.sleep(2)
                # 

    def capture_screenshot(self):
        outfile = os.path.expanduser('~/Desktop/captured.png')
        outcmd = "{} {} {}".format('screencapture', '-x', outfile)

        # Take screenshot of desktop background
        try:
            subprocess.check_output(outcmd, shell=True)
            print(f'Screenshot saved to: {outfile}')
        except subprocess.CalledProcessError as e:
            print(f'Error: {e}')

        img = cv2.imread(outfile)

        # Pre-process the image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        noise_removal = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)

        # Run Tesseract OCR
        text = pytesseract.image_to_string(noise_removal, lang='eng', config='--psm 11')

        def clean_text(text):
            # Remove excess newlines and whitespace
            cleaned_text = re.sub(r'\n+', '\n', text)  # Replace multiple newlines with a single newline
            cleaned_text = cleaned_text.strip()  # Remove leading and trailing whitespace
            return cleaned_text

        # Print the recognized text
        print('Image read!')

        res = ollama.chat(
            model="llava",
            messages=[
                {
                    'role': 'user',
                    'content': f'''
                    You are my extremely creative and insightful service that analyzes screenshots and provides helpful insights or suggestions, similar to Clippy. 
                    Analyze the screenshot image to identify any noteworthy elements, such as software open, text content on the screen, and user activities. 
                    
                    Do not give suggestions about multitasking, tabs, organization, or time management.
                    Do not give suggestions that are obvious to someone used to the internet. Provide insights that are profound and extremely creative based on what I'm doing.

                    If it helps you infer what I'm doing or working on, here are the current words captured on the screen: {clean_text(text)}
                    
                    Limit yourself to two suggestions, keep each short and to the point (2 sentences max). Tell me the action or suggestion only. Begin your response starting with Consider
                    ''',
                    #'images': [outfile] # Look at the image itself
                }
            ],
            stream=True
        )

        self.output_textedit.clear()
        for chunk in res:
            self.update_textbox(chunk['message']['content'])
            QApplication.processEvents()

    def update_textbox(self, text):
        if text:
            current_text = self.output_textedit.toPlainText()
            updated_text = current_text + text if current_text else text
            self.output_textedit.setPlainText(updated_text)  # Set updated text to 

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())