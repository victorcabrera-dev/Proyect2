import os
from cProfile import label
from tkinter import *
import tkinter as tk
import inutils
import imutils
import cv2
from PIL import Image, ImageTk
import cv2
from datetime import datetime
import pyodbc
from tkinter import messagebox
from PIL.ImageOps import expand
from fontTools.afmLib import writelines
from jax.experimental.export import export
from tensorflow.python.ops.signal.shape_ops import frame

from process.gui.image_paths import ImagePaths
from process.database.config import DataBasePaths
from process.face_processing.face_signup import FaceSignUp
from process.face_processing.face_login import FaceLogIn
#from process.com_interface.serial_com import SerialCommunication

# Empaquetar la ventana principal
class CustomFrame(tk.Frame):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.pack(fill=tk.BOTH, expand=True)


class GraphicalUserInterface:
    def __init__(self, root):
        self.main_window = root
        self.main_window.title('Control de Acceso Facial')
        # Obtener las dimensiones de la pantalla
        screen_width = self.main_window.winfo_screenwidth()
        screen_height = self.main_window.winfo_screenheight()
        # Dimensiones de la ventana
        window_width = 1280
        window_height = 720
        # Calcular las coordenadas para centrar la ventana
        x_position = (screen_width // 2) - (window_width // 2)
        y_position = (screen_height // 2) - (window_height // 2)
        # Establecer la geometría de la ventana centrada
        self.main_window.geometry(f'{window_width}x{window_height}+{x_position}+{y_position}')
        #self.main_window.resizable(False, False) #Desavilitar el boton de maximizar
        self.frame = CustomFrame(self.main_window)

        # Variable de control para evitar múltiples ventanas de mensaje--------------------BORRAR CUANDO HAYA PUERTA
        self.message_shown = False

        #DB_SQL CONEC
        try:
            # Conectar a la base de datos SQL Server
            self.conn = pyodbc.connect(
                'DRIVER={ODBC Driver 17 for SQL Server};'
                'SERVER=DESKTOP-Q1T957H;'
                'DATABASE=ControlAcceso;'
                'UID=sa;'
                'PWD=123'
            )
            self.cursor = self.conn.cursor()
        except Exception as e:
            print(f"Error al conectar a la base de datos: {e}")
            self.conn = None  # Aseguramos que self.conn sea None si ocurre un error

        #VIDEO CAPTURA , CONFIG STREAM
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3 , 1280)
        self.cap.set(4 , 720)


        #Signup Window = registro
        self.signup_window= None
        self.input_name = None
        self.input_user_code = None
        self.name = None
        self.user_code = None
        self.user_list = None
        # Face capture,  captura facial
        self.face_signup_window = None
        self.signup_video = None
        self.user_codes = []
        self.data = []

        #LOGIN WINDOW - VENTANA DE LOGIN secion 4
        self.face_login_window = None
        self.login_video = None


        # Modulos
        self.images = ImagePaths()
        self.database = DataBasePaths()
        self.face_Sign_Up = FaceSignUp()
        self.face_login = FaceLogIn()

        # Process
        self.main()

    def close_login(self):
        self.face_login.__init__()
        self.face_login_window.destroy()
        self.login_video.destroy()
        self.message_shown = False  # Reinicia la variable de control cuando se cierra el login------BORRAR CUANDO HAYA PUERTA

    # SESION 04

    def facial_login(self):
            if self.cap:
                ret, frame_bgr = self.cap.read()

                if ret:
                    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    # process
                    frame, user_access, info = self.face_login.process(frame)
                    # config video
                    frame = imutils.resize(frame, width=1280)
                    im = Image.fromarray(frame)
                    img = ImageTk.PhotoImage(image=im)

                    # show video
                    self.login_video.configure(image=img)
                    self.login_video.image = img
                    self.login_video.after(10, self.facial_login)

                    #AGREGADO POR VICTOR PARA MOSTRAR VENTANA EMERGENTE SI-----BORRAR CUANDO HAYA PUERTA
                    if user_access and not self.message_shown:  # Verifica si el mensaje ya fue mostrado
                        self.message_shown = True  # Marca como mostrado
                        messagebox.showinfo("Control de Acceso", "¡Acceso permitido, Abriendo puerta!")
                        self.login_video.after(2000, self.close_login)
                    elif user_access is False and not self.message_shown:  # Verifica si el mensaje ya fue mostrado
                        self.message_shown = True  # Marca como mostrado
                        messagebox.showinfo("Control de Acceso", "¡Acceso denegado, rostro no registrado!")
                        self.login_video.after(2000, self.close_login)
                    # AGREGADO POR VICTOR PARA MOSTRAR VENTANA EMERGENTE SI-----BORRAR CUANDO HAYA PUERTA

                    '''
                    if user_access:
                        #Serial communication - comunicacion serial
                        print('Acceso permitido, Abriendo puerta')
                        self.login_video.after(2000, self.close_login())
                    elif user_access is False:
                        print('Acceso denegado, rostro no registrado')
                        self.login_video.after(2000, self.close_login())
                    '''
            else:
                self.cap.release()

    #SECION 04...........................................................
    def gui_login(self):
        #SE CREA UNA NUEVA VENTANA
        self.face_login_window = Toplevel()
        self.face_login_window.title('Inicio de Sesión')
        self.face_login_window.geometry('1280x720')

        self.login_video = Label(self.face_login_window)
        self.login_video.place(x=0, y=0)
        self.facial_login()


    #cerrando la ventana de captura facial
    def close_signup(self, ):
        self.face_Sign_Up.__init__()
        self.face_signup_window.destroy()
        self.signup_video.destroy()



    def facial_sign_up(self):
        if self.cap:
            ret, frame_bgr = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                # process
                frame, save_image, info = self.face_Sign_Up.process(frame, self.user_code)

                # config video
                frame = imutils.resize(frame, width=1280)
                im = Image.fromarray(frame)
                img = ImageTk.PhotoImage(image=im)

                # show frames
                self.signup_video.configure(image=img)
                self.signup_video.image = img
                self.signup_video.after(10, self.facial_sign_up)

                if save_image:
                    self.signup_video.after(3000, self.close_signup)
                else:
                    self.signup_video.after(3000, self.close_signup)


        else:
            self.cap.release()


    def data_sign_up(self):
        #Leer informacion =  extract data
        self.name, self.user_code = self.input_name.get(), self.input_user_code.get()
        #checkeo de datos , no debe aver datos vacios
        if len(self.name) ==0 or len(self.user_code) == 0:
            print('Formulario incompleto')
        else:

            # VERIFICAR SI EL USUARIO YA ESTA REGISTRADO
            self.cursor.execute("SELECT codigo_usuario FROM Usuario WHERE codigo_usuario = ?", self.user_code)
            result = self.cursor.fetchone()

            if result:
                #print('¡Usuario registrado previamente!')------SE AGREGO MESSAGEBOX
                messagebox.showinfo("Registro de Usuario", "¡Usuario registrado previamente!")
                # Cerrar la ventana de registro cuando exisite un usuario existente
                self.signup_window.destroy()
            #PARA GUARDAR EN SQL SERVER
            else:
                # Obtener la fecha de registro
                registration_date = datetime.now()

                # Insertar los datos en la base de datos
                self.cursor.execute(
                    "INSERT INTO Usuario (usuario, codigo_usuario, Fecha_registro) VALUES (?, ?, ?)",
                    self.name, self.user_code, registration_date
                )
                self.conn.commit()

                # Guardar la información en un archivo de texto (como en tu código original)
                file = open(f"{self.database.users}/{self.user_code}.txt", 'w')
                file.writelines(self.name + ', ')
                file.writelines(self.user_code + ', ')
                file.writelines('Fecha: ' + registration_date.strftime("%Y-%m-%d %H:%M:%S") + '\n')
                file.close()

                #clear = limpiar los inputs casillas de registro
                self.input_name.delete(0, END)
                self.input_user_code.delete(0, END)

                #face register = registro facial
                self.face_signup_window = Toplevel()
                self.face_signup_window.title('Captura Facial')
                self.face_signup_window.geometry('1280x720')

                self.signup_video = Label(self.face_signup_window)
                self.signup_video.place(x=0, y=0)
                self.signup_window.destroy()
                self.facial_sign_up()

    #CERRANDO LA CONEC A SQL
    def __del__(self):
        try:
            if hasattr(self, 'conn') and self.conn:
                self.conn.close()  # Cerrar la conexión a SQL Server
        except Exception as e:
            print(f"Error al cerrar la conexión: {e}")


    def gui_signup(self):
        #Ventana de registro
        self.signup_window = Toplevel(self.frame)
        self.signup_window.title("Registro Facial")
        self.signup_window.geometry("1280x720")
        # background
        background_signup_img = PhotoImage(file=self.images.gui_signup_img)
        background_signup =Label(self.signup_window, image=background_signup_img)
        background_signup.image = background_signup_img
        background_signup.place(x=0, y=0)

        #imput data = ingreso de informacion
        self.input_name = Entry(self.signup_window)
        self.input_name.place(x=600, y=320)
        self.input_user_code = Entry(self.signup_window, show='*') #Show ='*' para no ver la contrasena..........
        self.input_user_code.place(x=600, y=475)

        #input button = boton para insertar datos

        register_button_img = PhotoImage(file=self.images.register_img)
        register_button = Button(self.signup_window, image=register_button_img, height="40", width="200", command=self.data_sign_up)
        register_button.image = register_button_img
        register_button.place(x=1005, y=565)


    def main(self):
        # Estilos a la ventana (forma)
        # background
        background_img = PhotoImage(file=self.images.init_img)
        background = Label(self.frame, image=background_img, text='back')
        background.image = background_img
        background.place(x=0, y=0, relwidth=1, relheight=1)

        #Botones de entrar
        login_button_img = PhotoImage(file=self.images.login_img)
        login_button = Button(self.frame, image=login_button_img, height="40", width="200", command=self.gui_login)
        login_button.image = login_button_img
        login_button.place(x=655 , y=285)

        #boton de registro

        signup_button_img = PhotoImage(file=self.images.signup_img)
        signup_button = Button(self.frame, image=signup_button_img, height="40", width="200", command=self.gui_signup)
        signup_button.image = signup_button_img
        signup_button.place(x=655, y=510)









