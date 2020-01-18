#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Francesco Ermini'

import time
import datetime

import logging

from shapely import geometry, wkt, wkb
from shapely.geometry import shape, Point, LineString, Polygon

from shapely.geometry import Point

import psycopg2

from math import cos, asin, sqrt, pi
import numpy as np

from wms import WMSTool, parse_geom_response, parse_html_response, ITALIA_WMS_URL, CATASTO_ITALIA_LAYER_PARTICELLE, CATASTO_ITALIA_SRS
from utils import bbox, size, point
from utils import *


LOG_LEVEL = logging.INFO
#logging.basicConfig(filename='catasto_errors.log', filemode='w+', format='%(message)s', level=logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)



DISTANCE_SAMPLING = 100 #meters between points
MAX_POINTS = 10
IMG_PIXEL_WIDTH = 200
PRINT_UPDATES_EVERY_N_QUERY = 50


def calc_process_time(starttime, cur_iter, max_iter):
    telapsed = time.time() - starttime
    testimated = (telapsed/cur_iter)*(max_iter)

    finishtime = starttime + testimated
    finishtime = datetime.datetime.fromtimestamp(finishtime).strftime("%d/%m/%Y, %H:%M:%S")  # in time

    lefttime = testimated-telapsed  # in seconds
    print("time elapsed: %s(s), time left: %s(s), estimated finish time: %s" % (int(telapsed), int(lefttime), finishtime))



"""
09;248;048; 001 ;048001;Bagno a Ripoli;Bagno a Ripoli;;3;Centro;Toscana;Firenze;0;FI;48001;48001;48001;48001;A564;25.403;ITI;ITI1;ITI14
"""


class CatastoQueryTool:

    def __init__(self, id_comune):
        self.srs = 4258
        self.wms = WMSTool(base_url=ITALIA_WMS_URL, srs=CATASTO_ITALIA_SRS, layer=CATASTO_ITALIA_LAYER_PARTICELLE)
        self.points = []
        self.id_comune = id_comune
        self.connection = psycopg2.connect(dbname='cadastredb', user='biloba', host='127.0.0.1', password='biloba')
        self.cursor = self.connection.cursor()
        self.generate_points()

    def run(self):
        if self.points is None:
            self.logging.critical("Non ci sono punti da interrogare, bona ci si!")
            return False

        if self.cursor is None:
            self.cursor = self.connection.cursor()

        start_time = time.time()
        queries_succeeded = 0
        queries_index = 0
        print("START TIME: "+str(datetime.datetime.fromtimestamp(start_time).strftime("%d/%m/%Y, %H:%M:%S")))
        for p in self.points:
            queries_index += 1
            if self.query_point(p[1], p[0]):
                queries_succeeded += 1
            if queries_index % PRINT_UPDATES_EVERY_N_QUERY:
                calc_process_time(starttime=start_time, cur_iter=queries_succeeded, max_iter=len(self.points))
        end_time = time.time()
        duration = end_time - start_time
        print("%s QUERES LAST FOR  %s  seconds" % (len(self.points), str(duration)))

        self.cursor.execute("SELECT pg_size_pretty( pg_database_size('cadastredb'));")
        print("%s QUERIES SET DATABASE MEMORY TO %s" % (len(self.points), self.cursor.fetchall()))
        self.connection.commit()
        self.cursor.close()
        self.connection.close()
        return True

    def generate_points(self, _n_max=MAX_POINTS, _distance=DISTANCE_SAMPLING):
        logging.info("MAX NUMBER OF POINT BY CONFIG: "+str(_n_max))
        logging.info("PRECISION OF SAMPLING DISTANCE in meter: "+str(_distance))
        self.cursor.execute("SELECT  geom FROM comuni WHERE id='{0}';".format(str(self.id_comune)))
        _res = self.cursor.fetchall()
        if _res is None:
            raise ValueError("comune id non esiste nel db, riprova scemo!")
        elif len(_res) > 1:
            raise ValueError("Codice comune risulta dublicato, wtf?")
        
        poly = wkb.loads(_res[0][0], hex=True)
        minx, miny, maxx, maxy = poly.bounds
        _x = np.arange(minx, maxx, meters2degrees(_distance))
        _y = np.arange(miny, maxy, meters2degrees(_distance))

        _xy = np.meshgrid(_x, _y)
        mat = np.array(_xy).transpose()
        p_array = np.reshape(mat, (1,-1,2))
        logging.info("PUNTI TOTALI NEL COMUNE: " + str(len(p_array[0])))

        count = 0
        while len(self.points) < _n_max and count < len(p_array[0]):
            _point = Point(p_array[0][count][0], p_array[0][count][1])
            if poly.contains(_point):
                self.points.append([p_array[0][count][0],p_array[0][count][1]])
            count += 1
        logging.info("PUNTI SCELTI: " + str(len(self.points)))

    def reset(self):
        drop_particelle = "DROP TABLE IF EXISTS particelle;"

        create_particelle = "CREATE TABLE particelle (id SERIAL PRIMARY KEY, \
                                                    comune VARCHAR(64), foglio VARCHAR(64), particella VARCHAR(64),\
                                                    bbox GEOMETRY,\
                                                    geom GEOMETRY,\
                                                    updated TIMESTAMP DEFAULT NOW()\
                                                    );"
        self.cursor.execute(drop_particelle)
        self.cursor.execute(create_particelle)
        self.connection.commit()

    def query_point(self, lat, lon, store=True):
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

        self.cursor.execute('SELECT 1 from particelle WHERE comune=%s AND foglio=%s AND particella=%s', (comune, foglio, particella))
        if self.cursor.fetchone() is not None:
            logging.debug("particella duplicata: %s, %s, %s " % (comune, foglio, particella))
            return True
        
        logging.debug("NUOVA PARTICELLA: %s, %s, %s " % (comune, foglio, particella))
        self.cursor.execute('INSERT INTO particelle (comune,foglio,particella) VALUES (%s, %s, %s)', (comune, foglio, particella))

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
        self.cursor.execute('UPDATE particelle SET bbox=ST_GeomFromText(ST_AsText(%s),%s) WHERE comune=%s AND foglio=%s AND particella=%s;', (bbox_poly.wkt, str(self.srs), comune, foglio, particella))
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
        self.cursor.execute('UPDATE particelle SET geom=ST_GeomFromText(ST_AsText(%s),%s) WHERE  comune=%s AND foglio=%s AND particella=%s;', (poly.wkt, str(self.srs), comune, foglio, particella))
        #self.cursor.execute("UPDATE particelle SET geom VALUES (ST_AsEWKT(%s));", (poly.wkb,))
        #self.cursor.execute('INSERT INTO particelle (geom) VALUES (ST_GeomFromText(ST_AsText(%s),%s))', (poly.wkt, str(self.srs)))

        print("saved geometry for: %s, %s, %s " % (comune, foglio, particella))


        return True


if __name__ == '__main__':
    c = CatastoQueryTool(id_comune='048001')
    c.reset()
    c.run()
