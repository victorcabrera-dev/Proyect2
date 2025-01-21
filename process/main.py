import os
from tkinter import *
import tkinter as tk
import inutils
import cv2
from PIL import Image, ImageTk
import cv2
from PIL.ImageOps import expand
from jax.experimental.export import export

from process.gui.image_paths import ImagePaths
from process.database.config import DataBasePaths
#from process.face_processing.face_signup import FacialSignUp
#from process.face_processing.face_login import FacialLogIn
#from process.com_interface.serial_com import SerialCommunication

#Empaquetar la ventana principal
class CustomFrame(tk.Frame):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.pack(fill=tk.BOTH, expand=True)


class GraphicalUserInterface:
    def __init__(self, root):
        self.main_window = root
        self.main_window.title('control de acceso facial')
        self.main_window.geometry('1280x720')
        self.frame = CustomFrame(self.main_window)
        pass
    def main(self):
        print('algo')


