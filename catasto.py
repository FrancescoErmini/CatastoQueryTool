#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Francesco Ermini'



import logging

from wms import WMSTool, parse_geom_response, parse_html_response, ITALIA_WMS_URL, \
    CATASTO_ITALIA_LAYER_PARTICELLE, CATASTO_ITALIA_SRS
from utils.utils import *

from dao.dao import DAOParticella, CATASTO_DAO

LOG_LEVEL = logging.INFO
#logging.basicConfig(filename='catasto_errors.log', filemode='w+', format='%(message)s', level=logging.DEBUG)
logging.basicConfig(level=LOG_LEVEL)



from config import  DISTANCE_SAMPLING, MAX_POINTS, IMG_PIXEL_WIDTH, PRINT_UPDATES_EVERY_N_QUERY

"""
CatastoQueryTool e' la classe dove e' implementata la logica di ricostruzione
del dato catastale (comune, foglio, particella) a partire da un punto geografio.


CatastoQueryTool esponse un unico metodo, query_point().
Vedi documentazione diagramam flusso.

"""


class CatastoQueryTool:

    def __init__(self):
        self.srs = 4258
        self.wms = WMSTool(base_url=ITALIA_WMS_URL, srs=CATASTO_ITALIA_SRS, layer=CATASTO_ITALIA_LAYER_PARTICELLE)
        self._dao = CATASTO_DAO

    def query_point(self, lat, lon):
        if 39.0 < lat < 46.0 and 8.0 < lon < 14.0: 
            pass
        else:
            raise ValueError("FUCK BULSHIT I M VERY STUPID")

        """
        0. Crea bbox fittizzio intorno al punto
        """
        _bbox_hint = create_bbox(lat, lon, meters=2)

        """
        1. ottieni comune, foglio e particella in cui cade un punto.
        """
        data_info = self.wms.get_info(_bbox_hint)

        if data_info is None:
            logging.error("Error: No data received for point: LAT %s - LON: %s" % (str(lat), str(lon)))
            return False

        comune, foglio, particella = parse_html_response(data_info)

        if comune is None or foglio is None or particella is None:
            logging.error("Error: Issue parsing comune, foglio, particella for point: LAT %s - LON: %s" % (str(lat), str(lon)))
            return False
        """
        1.1 Salva, aggiorna o skippa. 
        Se la particella non esiste, si inserisce, altrimenti:
        
        Se la particella esiste ma la data di inserimento è assai recente, 
        si considera duplicata e si interrompe esecuzione della query.
        
        Se la particella esiste ma la data di inserimento non e' recente,
        si esegue la query su bbox e se uguale a quella nel db 
        si considera duplicata e si interrompe esecuzione.
         
        altrimenti (se non si interrompe esecuzione) si 
        prosegue algoritomo aggiornando bbox e geom.
        
        """
        if self._dao.exists(comune, foglio, particella):
            # TODO: logica di aggiornamento
            #
            logging.debug("particella duplicata: %s, %s, %s " % (comune, foglio, particella))
            return True
        
        logging.debug("NUOVA PARTICELLA: %s, %s, %s " % (comune, foglio, particella))
        self._dao.insert(comune, foglio, particella)
        """
        2. Ottieni bbox per quella particella. 
           Calcola la size in pixel usando le proporzioni ottenute da bbox.
           Memorizza il bbox per quella particella.
        """
        data_bbox = self.wms.get_geom(_bbox_hint)
        
        if data_bbox is None:
            logging.error("Error: No bbox data received for comune %s, foglio %s, particella %s" % (comune, foglio, particella))
            return False

        _bbox_rcv = parse_geom_response(data_bbox)

        if _bbox_rcv is None:
            logging.error("Error: Issue parsing bbox data for comune %s, foglio %s, particella %s" % (comune, foglio, particella))
            return False
        
        #todo aggiungi check validity coordinate e poligoni.
        bbox_poly = geometry.Polygon([[_bbox_rcv.lat1, _bbox_rcv.lon1], [_bbox_rcv.lat1, _bbox_rcv.lon2], [_bbox_rcv.lat2, _bbox_rcv.lon2], [_bbox_rcv.lat2, _bbox_rcv.lon1], [_bbox_rcv.lat1, _bbox_rcv.lon1]])
        self._dao.update_geom(comune, foglio, particella, key='bbox', poly=bbox_poly, srs=self.srs)

        #print("saved bbox for: %s, %s, %s ", (comune, foglio, particella))

        """
        3. Ottieni immagine della particella di interesse.
        
        Nota: Se l'immagine richiesta sfora i limiti di scala
        imposti dal castasto, l'immagine che ritorna e' bianca.
        Per aggirare questo problema, nel caso in cui l'immagine 
        richiesta abbia una scala non supportata, il bbox viene 
        spezzettato in tanti bboxes piu piccoli e poi le immagini
        vengono ricomposte per ottenere l'immagine con tutta la 
        particella al suo interno.
        """

        image = None
        _new_size = get_size_from_bbox(_bbox=_bbox_rcv, pixel_width=IMG_PIXEL_WIDTH)

        if not is_out_of_scale(_bbox_rcv):
            try:
                img_bytes = self.wms.get_map(_bbox_rcv, _size=_new_size)
                if img_bytes is None:
                    logging.error("WMS Error: Impossible get image. image is null.")
                    return False
                img_np = np.frombuffer(img_bytes, dtype=np.uint8)
                image = cv2.imdecode(img_np, flags=1)
            except Exception:
                logging.error("Error: Impossible to decode image.")
                return False

        else:
            final_image = None
            current_image = None
            bboxes_obj = create_bboxes(_bbox_rcv)
            bboxes = bboxes_obj["bboxes"]
            x_len = bboxes_obj["size"]["x"]
            y_len = bboxes_obj["size"]["y"]

            bboxes_matrix = [[None for j in range(y_len)] for i in range(x_len)]

            """
            Attenzione: bboxes hanno indici con riferimento dal basso. 
            Ovverro 0,0 è in basso a sinistra. 1,0 basso a destra. e così via.
            """

            for _bbox in bboxes:
                try:
                    _img_src = self.wms.get_map(_bbox, _size=get_size_from_bbox(_bbox=_bbox, pixel_width=IMG_PIXEL_WIDTH))
                    if _img_src is None:
                        logging.error("WMS Error: Impossible get image. image is null.")
                        return False
                    _img = cv2.imdecode(np.frombuffer( _img_src, dtype=np.uint8), flags=1)
                    bboxes_matrix[_bbox.x][_bbox.y] = _img
                except Exception:
                    logging.error("WMS Error: Impossible get image. image is null.")
                    return False

            for y in range(y_len):
                for x in range(x_len):
                    if x == 0:
                        current_image = bboxes_matrix[x][y]
                    else:
                        my_img = bboxes_matrix[x][y]
                        # concatena orizzonate aggiungendo nuvoa img a destra.
                        current_image = np.concatenate((current_image, my_img), axis=1)
                if y == 0:
                    final_image = current_image
                else:
                    # concatena verticale aggiungendo nuvoa img sopra.
                    final_image = np.concatenate((current_image, final_image), axis=0)
            image = final_image

        if image is None:
            logging.error("error image not received")
            return False

        """
        4. ottieni le coordinate dei punti della particella 
           processado l'immagine ottenuta da wms.
        """

        xy = compute_shape_from_map_image(image)

        if xy is None:
            logging.error("Error: unable to get geometry for: %s, %s, %s with bbox: %s, with distance %s" % (comune, foglio, particella, str(_bbox_rcv), str(distance(_bbox_rcv.lat1, _bbox_rcv.lon1, _bbox_rcv.lat2, _bbox_rcv.lon2))) )
            return False
        coords = reproj(_xy=xy, _size=_new_size, _bbox=_bbox_rcv)

        if coords is None:
            logging.error("Error: unable to reproject coordiantes in ESRI system")
            return False

        poly = geometry.Polygon([[p.lat, p.lon] for p in coords])
        print(poly.wkt)
        self._dao.update_geom(comune, foglio, particella, key='geom', poly=poly, srs=self.srs)

        print("saved geometry for: %s, %s, %s " % (comune, foglio, particella))

        return True


