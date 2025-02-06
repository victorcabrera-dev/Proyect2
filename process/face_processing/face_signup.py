import numpy as np
from typing import Tuple

from process.face_processing.face_utils import FaceUtils
from process.database.config import DataBasePaths


class FaceSignUp:
    def __init__(self):
        self.database = DataBasePaths()
        self.face_utilities = FaceUtils()

    def process(self, face_image: np.ndarray, user_code: str)-> Tuple[np.ndarray, bool, str]:
        # stap 1: check face detection = DETECTAR SI HAY DETECCION DE ROSTRO
        check_face_detect, face_info, face_save = self.face_utilities.check_face(face_image)
        if check_face_detect is False:
            return face_image, False, '!No se detecto ningun rostro!'

        # stap 2. MALLA FACIAL
        check_face_mesh, face_mesh_info = self.face_utilities.face_mesh(face_image)
        if check_face_mesh is False:
            return face_image, False, '!No se detecto la malla facial!'

        # stap 3. EXTRACCION DE PUNTOS FACIALES DE LA MALLA FACIAL
        face_mesh_points_list = self.face_utilities.extract_face_mesh(face_image, face_mesh_info)

        # Stap 4. Verificr si el rosotro esta centrado . check face center
        check_face_center = self.face_utilities.check_face_center(face_mesh_points_list)

        #Stap 5. Mostar estado de caputa (rojo no capturo la foto y verde ya capturo ) show state
        self.face_utilities.show_state_signup(face_image, state= check_face_center)
        if check_face_center:
            #stap 6. extract face info - estraccion de informacion
            face_bbox = self.face_utilities.extract_face_bbox(face_image, face_info)

            # stap 6. extract face points
            face_points = self.face_utilities.extract_face_points(face_image, face_info)

            # stap 7. alinear los rostros - face aligned
            face_aligned = self.face_utilities.face_alignment(face_save, face_points)

            # stap 8. face crop - recorte de rostro
            face_crop = self.face_utilities.face_crop(face_aligned, face_bbox)

            # stap 9. face save - guardar la foto
            check_save_image = self.face_utilities.save_face(face_crop, user_code, self.database.faces)
            return face_image, check_save_image, 'Imagen guardada'

        else:
            return  face_image, False, 'No se guardo la imagen'






