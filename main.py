#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Francesco Ermini'
import os
import sys
import re
import time
import datetime
import random
import logging
from dataclasses import dataclass

from shapely import geometry, wkt, wkb
from shapely.geometry import shape, Point, LineString, Polygon

from shapely.geometry import Point



import psycopg2
from owslib.wms import WebMapService
import cv2


from math import cos, asin, sqrt, pi
import numpy as np

LOG_LEVEL = logging.INFO
logging.basicConfig(filename='logs/catasto_errors.log', filemode='w+', format='%(message)s', level=logging.INFO)
# logging.basicConfig(level=logging.ERROR)

DEBUG_IMAGE = False
DEBUG_IMAGE_SAVE = False
DEBUG_IMAGE_LIVE = False

ITALIA_WMS_URL = 'https://wms.cartografia.agenziaentrate.gov.it/inspire/wms/ows01.php'
CATASTO_ITALIA_SRS = 'EPSG:4258'
CATASTO_ITALIA_LAYER_PARTICELLE = 'CP.CadastralParcel'
MAX_CADASTRE_SCALE_THRESHOLD = 200.0 # metri oltre i quali il catasto mostra una immagine bianca (troppo zoom out)

DISTANCE_SAMPLING = 100 #meters between points
MAX_POINTS = 100000
IMG_PIXEL_WIDTH = 200
PRINT_UPDATES_EVERY_N_QUERY = 10
QUERY_CONNECTION_TIMEOUT = 10
#### MODEL ######


@dataclass
class bbox:
    lat1: float
    lon1: float
    lat2: float
    lon2: float
    x: int = -1
    y: int = -1

@dataclass
class size:
    w: int
    h: int


@dataclass
class point:
    lat: float
    lon: float


def parse_html_response(html_string):
    if html_string is None:
        return None
    #if html_string == 'b\'Content-Type: text/html\\r\\n\\r\\n\''
    if "<td>" not in html_string:
        return "na", "na", "na"
    try:
        res = re.findall(r'NationalCadastralReference</th><td>(.*?)</td>', html_string, re.M | re.I | re.S)
        codice = res[0]
        comune = codice[:4]
        foglio = codice[5:9]
        particella = codice.split(".")[1]#codice[-3:]
        return comune, foglio, particella
    except Exception:
        return None, None, None


def parse_geom_response(gml_string):
    if gml_string is None:
        return None
    try:
        res = re.findall(r'<gml:coordinates>(.*?)</gml:coordinates>', gml_string, re.M | re.I | re.S)
        points = res[0].split()
        p1 = points[0].split(",")
        p2 = points[1].split(",")
        _bbox = bbox(lon1=float(p1[0]), lat1=float(p1[1]), lon2=float(p2[0]), lat2=float(p2[1]))
        return _bbox
    except Exception:
        return None


def distance(lat1, lon1, lat2, lon2, unit='mt'):
    p = 0.017453292519943295  # Pi/180
    a = 0.5 - cos((lat2 - lat1) * p) / 2 + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
    res = 12742 * asin(sqrt(a)) # km
    if unit == 'mt':
        res = res * 1000
    return res


def get_size_from_bbox(_bbox, pixel_width):
    # get base x in meter
    real_width = distance(_bbox.lat1, _bbox.lon1, _bbox.lat1, _bbox.lon2)
    # get high y in meter
    real_height = distance(_bbox.lat1, _bbox.lon1, _bbox.lat2, _bbox.lon1)
    # proportion:
    # pixel_width : pixel_height = real_width : real_height
    pixel_height = int((real_height * pixel_width) / real_width)
    _size = size(w=pixel_width, h=pixel_height)
    return _size


def create_bbox(lat1, lon1, meters=1):

    #
    # 0.00001 in coordinaes is 1.37 meters
    #
    lat1 = round(lat1, 5)
    lon1 = round(lon1, 5)
    delta = round(0.00001 * meters, 5)
    lat2 = round(lat1 + delta, 5)
    lon2 = round(lon1 + delta, 5)
    _bbox = bbox(lat1=lat1, lon1=lon1, lat2=lat2, lon2=lon2)
    return _bbox


def debug_img(image, title="approx", contourn=None):
    try:
        import datetime
        current_time = datetime.datetime.now()
        suffix = "_{}.png".format(current_time)
        name = title + suffix

        if contourn is not None:
            approx = cv2.approxPolyDP(contourn, 0.001 * cv2.arcLength(contourn, True), True)
            cv2.drawContours(image, [approx], 0,
                             (random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255)), 2)
            # marca il punti 0,0  per riferimento.
            cv2.circle(image, (0, 0), 3, (155, 243, 198), -1)
            for i in approx:
                x, y = i.ravel()
                cv2.circle(image, (x, y), 3, (255, 0, 0), -1)

        if DEBUG_IMAGE_SAVE:
            img_path = os.path.join("imgs", name)
            cv2.imwrite(img_path, image)

        if DEBUG_IMAGE_LIVE:
            cv2.imshow(name, image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    except Exception:
        pass

def compute_shape_from_map_image(image, title="approx poly"):
    """

    Args:
        image: the image to analyse

    Returns:

        List of tuples if no errors occurs. None if any error occurs.
        x and y  are coordinates of points of polygon founded in the image (e.g [(120,56),(125,53)..])

    Notes:
        Ensure that input arg image has been converted using:
        img_np = np.frombuffer(image, dtype=np.uint8)
        image = cv2.imdecode(img_np, flags=1)

    """

    try:

        """
        1. trova tutti i contorni (poligoni) nell'immagine.
        """

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        """
        2. elimina tutti i poligoni il cui "rettangolo" che li contiene
           non combacia con la dimensione dell'immagine.
           l'immagine viene presa rispetto alle coordinate del bbox della particella.
           Questo permette di eliminare le particelle che sono poligoni chiususi adiacenti
           alla particella di ineteresse.
        """

        size_y, size_x, _ = image.shape
        new_contours = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            #cv2.rectangle(orig_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            #cv2.imshow('Bounding rect', orig_image)
            y_tollerance = 100.0 - (100.0 * h / size_y)
            x_tollerance = 100.0 - (100.0 * w / size_x)
            #if abs(w - size_x) < 10.0 and abs(h - size_y) < 10.0:
            if x_tollerance < 10.0 and y_tollerance < 10.0:
                new_contours.append(c)

        if DEBUG_IMAGE:
            for c in contours:
                debug_img(image, title="debug_"+title, contourn=c)


        """
        3. Scegli tra i poligoni rimaneneti quello con area minore.
           Se c'e' un solo poligono seleziona quello.
           
           Questo passaggio serve per eliminare i casi in cui i punti del
           poligono sono corretti ma l'ordine con cui sono uniti i punti
           e' sbagliato, quindi si generano delle distorsioni rispetto
           alla sagoma corretta del poligono da ricorstruire.
           
           Queste distorsioni risultano in un area maggiore del poligono errato.
           Per questo userò il criterio dell'area minore per selezionare il 
           poligono corretto.
           
        """

        try:
            c_opt = new_contours[0]
        except IndexError:
            # debugga il motivo per cui non esistono poligoni nell'immagine.
            if DEBUG_IMAGE:
                prefix = "failed_"
                name = prefix + title

                debug_img(image, title=name)
                #cv2.imshow(name, image)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
            return None
        # se esistono più poligoni, seleziona quello con area minore.
        if len(new_contours) > 1:
            area = cv2.contourArea(c_opt)
            for c in new_contours[1:]:
                new_area = cv2.contourArea(c)
                if new_area < area:
                    #print (" nuovo: " + str(new_area))
                    #print (" vecchio: " + str(area))
                    c_opt = c

        """
        4. Calcola i punti del poligono.
           Salva i punti come lista di tuple.
           
           L'algoritmo approxPolyDP estrapola dal poligono i punti rilevanti,
           quelli utili per la ricostruzione del poligono.
           
        """
        approx = cv2.approxPolyDP(c_opt, 0.001 * cv2.arcLength(c_opt, True), True)
        xy = []
        for i in approx:
            x, y = i.ravel()
            xy.append((x, y))


        """
        6. Opzionale: Disegna poligono e punti.
        """

        if DEBUG_IMAGE:
            prefix = "success"
            name = prefix + title
            debug_img(image, title=name, contours=c_opt)
            # cv2.drawContours(image, [approx], 0, (random.randrange(0, 255),random.randrange(0, 255),random.randrange(0, 255)), 2)
            # # marca il punti 0,0  per riferimento.
            # cv2.circle(image, (0, 0), 3, (155, 243, 198), -1)
            # for i in approx:
            #     x, y = i.ravel()
            #     cv2.circle(image, (x, y), 3, (255, 0, 0), -1)
            # import datetime
            # current_time = datetime.datetime.now()
            # prefix = "success_"
            # suffix = "_{}.png".format(current_time)
            # name = prefix + title + suffix
            # img_path = os.path.join("imgs", name)
            # cv2.imwrite(img_path, image)
            # #cv2.imshow('Approx polyDP', image)
            # #cv2.waitKey(0)
            # #cv2.destroyAllWindows()

        """
        7. Ritorna una lista di tuple contenete le coordinate x y dei punti trovati.
        """

        return xy

    except Exception:
        return None


def shift_point(_point, dx, dy):
    latitude = _point.lat
    longitude = _point.lon
    r_earth = 6372.797
    m = (1.0 / ((2.0 * pi / 360.0) * r_earth)) / 1000.0
    new_longitude = longitude + (dx * m) / cos(latitude * (pi / 180.0))
    new_latitude = latitude + (dy * m)
    _new_point = point(lat=new_latitude, lon=new_longitude)
    return _new_point


def shift_lat(latitude, delta):
    dy = delta
    r_earth = 6372.797
    m = (1.0 / ((2.0 * pi / 360.0) * r_earth)) / 1000.0
    new_latitude = latitude + (dy * m)
    return new_latitude


def shift_lon(longitude, delta, latitude):
    dx = delta
    r_earth = 6372.797
    m = (1.0 / ((2.0 * pi / 360.0) * r_earth)) / 1000.0
    new_longitude = longitude + (dx * m) / cos(latitude * (pi / 180.0))
    return new_longitude


def reproj(_xy, _size, _bbox):
    """
    5. Riproietta le coordiante su ESPG
    """
    _coords = []
    # define origin point: the up left corner of bbox.
    origin_point = point(lat=_bbox.lat2, lon=_bbox.lon1)

    real_width =  distance(_bbox.lat1, _bbox.lon1, _bbox.lat1, _bbox.lon2)
    real_height = distance(_bbox.lat1, _bbox.lon1, _bbox.lat2, _bbox.lon1)

    # define pixel width and height
    ox = float(real_width / _size.w)
    oy = float(real_height / _size.h)

    for x,y in _xy:
        _coords.append(shift_point(origin_point, dx=x * ox, dy=-(y * oy)))
        #print(p_proj.lon,",",p_proj.lat)

    #print("zero")
    #print (origin_point.lon,",",origin_point.lat)
    return _coords


def is_out_of_scale(_bbox):
    _d = distance(lat1=_bbox.lat1, lon1=_bbox.lon1, lat2=_bbox.lat2, lon2=_bbox.lon2)
    _d_h = distance(lat1=_bbox.lat1, lon1=_bbox.lon1, lat2=_bbox.lat1, lon2=_bbox.lon2)
    _d_v = distance(lat1=_bbox.lat1, lon1=_bbox.lon1, lat2=_bbox.lat2, lon2=_bbox.lon1)
    if _d_v > 200 or _d_h > 200:
        return True
    return False


def meters2degrees(_meters):
    return round(_meters * 0.00001, 5)


def degrees2meters(_degrees):
    return _degrees * 100000


def bbox2polygon(_bbox):
    _poly = geometry.Polygon([[_bbox.lat1, _bbox.lon1], [_bbox.lat1, _bbox.lon2], [_bbox.lat2, _bbox.lon2], [_bbox.lat2, _bbox.lon1], [_bbox.lat1, _bbox.lon1]])
    return _poly.wkt


def create_bboxes(_bbox, delta_x_max=MAX_CADASTRE_SCALE_THRESHOLD, delta_y_max=MAX_CADASTRE_SCALE_THRESHOLD):
    """
    Given an input bbox data, generate smaller bboxes that have delta_x and delta_y under the specified values (in meters).
    Args:
        _bbox(bbox): input Bbox
        delta_x_max(int): max x in meters for smaller bboxes
        delta_y_max(int): max y in meters for smaller bboxes

    Returns:
        Dictionary with bboxes, size.x and size.y.
        bboxes: list of bbox objects. Each bbox has the position i,j embedded in the object itself.
        size.x, size.y the dimension of the resulting  matrix of bboxes.
    """

    # TODO: effettuare controll su dimensione bbox in input.
    x_origin = min(_bbox.lon1, _bbox.lon2)
    y_origin = min(_bbox.lat1, _bbox.lat2)

    x_distance = distance(lat1=_bbox.lat1, lon1=_bbox.lon1, lat2=_bbox.lat1, lon2=_bbox.lon2)
    y_distance = distance(lat1=_bbox.lat1, lon1=_bbox.lon1, lat2=_bbox.lat2, lon2=_bbox.lon1)

    delta_x = x_distance
    while delta_x >= delta_x_max:
        delta_x = delta_x / 2.0

    delta_y = y_distance
    while delta_y >= delta_y_max:
        delta_y = delta_y / 2.0

    delta_x_array = np.cumsum(np.array([0.0] + [delta_x for _xx in range(int(x_distance / delta_x))]))
    delta_y_array = np.cumsum(np.array([0.0] + [delta_y for _yy in range(int(y_distance / delta_y))]))

    _longitudes = [shift_lon(x_origin, latitude=y_origin, delta=ii) for ii in delta_x_array]
    _latitudes = [shift_lat(y_origin, delta=jj) for jj in delta_y_array]

    _bboxes = []

    for _y, (_lat_adj_0, _lat_adj_1) in enumerate(zip(_latitudes, _latitudes[1:])):
        for _x, (_lon_adj_0, _lon_adj_1) in enumerate(zip(_longitudes, _longitudes[1:])):
            _bboxes.append(bbox(lat1=_lat_adj_0, lat2=_lat_adj_1, lon1=_lon_adj_0, lon2=_lon_adj_1, x=_x, y=_y))

    return {"bboxes": _bboxes, "size": {"x": len(_longitudes)-1, "y": len(_latitudes)-1}}


def calc_process_time(starttime, cur_iter, max_iter):
    try:
        telapsed = time.time() - starttime
        testimated = (telapsed/cur_iter)*(max_iter)

        finishtime = starttime + testimated
        finishtime = datetime.datetime.fromtimestamp(finishtime).strftime("%d/%m/%Y, %H:%M:%S")  # in time

        lefttime = testimated-telapsed  # in seconds
        # print("time elapsed: %s(s), time left: %s(s), estimated finish time: %s" % (int(telapsed), int(lefttime), finishtime))
    except Exception:
        pass


class WMSTool:

    def __init__(self, base_url, srs, layer, version='1.3.0'):
        self.wms = WebMapService(base_url, version=version)
        self.srs = srs
        self.layer = layer

    # def _sanity_check(self, layer):
    #     if layer not in self.wms.contents.keys():
    #         error("Layer name not valid: choose between: "+str(self.wms.contents.keys()))
    #         return False
    #     return True

    def get_info(self, _bbox, size=(2, 2), info_format="text/html"):

        # ATTENZIONE: bbox: lon, lat, lon, lat!
        #
        try:
            data_html = self.wms.getfeatureinfo(
                layers=[self.layer],
                srs=self.srs,
                bbox=(_bbox.lon1, _bbox.lat1, _bbox.lon2, _bbox.lat2),
                size=(size[0], size[1]),
                format='image/png',
                query_layers=[self.layer],
                info_format=info_format,
                xy=(int(size[0]/2), int(size[1]/2)),
                feature_count=10
                )
            res = str(data_html._response.content)
            return res
        except Exception:
            return None

    def get_geom(self, _bbox, size=(2, 2), info_format='application/vnd.ogc.gml'):
        try:
            geom_info = self.wms.getfeatureinfo(
                layers=[self.layer],
                srs=self.srs,
                bbox=(_bbox.lon1, _bbox.lat1, _bbox.lon2, _bbox.lat2),
                size=(size[0], size[1]),
                format='image/png',
                query_layers=[self.layer],
                info_format=info_format,
                xy=(int(size[0]/2), int(size[1]/2))
            )
            coords = str(geom_info._response.content)
            return coords
        except Exception:
            return None

    def get_map(self, _bbox, _size, resolution=72):
        try:
            map = self.wms.getmap(
                layers=[self.layer],
                srs=self.srs,
                bbox=(_bbox.lon1, _bbox.lat1, _bbox.lon2, _bbox.lat2),
                size=(_size.w, _size.h),
                format='image/png',
                query_layers=[self.layer],
                DPI=72,
                MAP_RESOLUTION=72,
                FORMAT_OPTIONS='dpi:72'
            )
            return bytes(map._response.content)
        except Exception:
            return None

"""
09;248;048; 001 ;048001;Bagno a Ripoli;Bagno a Ripoli;;3;Centro;Toscana;Firenze;0;FI;48001;48001;48001;48001;A564;25.403;ITI;ITI1;ITI14
"""


class CatastoQueryTool:

    def __init__(self):
        self.srs = 4258
        self.wms = WMSTool(base_url=ITALIA_WMS_URL, srs=CATASTO_ITALIA_SRS, layer=CATASTO_ITALIA_LAYER_PARTICELLE)
        # scan_points = []
        # id_comune = None
        self.connection = psycopg2.connect(dbname='cadastredb', user='biloba', host='127.0.0.1', password='biloba')
        self.cursor = self.connection.cursor()

    def run(self):
        self.cursor.execute("SELECT id FROM comuni WHERE regione='Toscana' AND geom is not NULL;")
        _comuni = self.cursor.fetchall()
        if _comuni is None:
            import sys
            logging.critical("Unable to read the db of comuni, annammo bene..")
            sys.exit(1)
        i = 0
        for _comune in _comuni:
            i += 1
            print("\n############# comune: " + str(i) + "/" + str(len(_comuni)) + " #############")
            self.scan(_comune[0])
        self.stop()
        
    def scan(self, id_comune):
        
        scan_points = self.generate_points(id_comune=id_comune)
        if scan_points is None:
            logging.critical("Non ci sono punti da interrogare, bona ci si!")
            return False

        if self.cursor is None:
            self.cursor = self.connection.cursor()

        start_time = time.time()
        queries_succeeded = 0
        queries_index = 0
        queries_tot = len(scan_points)
        logging.info("START TIME for comune: "+str(id_comune)+" "+str(datetime.datetime.fromtimestamp(start_time).strftime("%d/%m/%Y, %H:%M:%S")))
        logging.info("TOT POINTS: "+str(queries_tot))
        for p in scan_points:
            queries_index += 1
            if self.query_point(p[1], p[0]):
                queries_succeeded += 1
            if int(queries_index) % int(PRINT_UPDATES_EVERY_N_QUERY) == 0:
                self.connection.commit()
                errors = int(100 * (queries_index-queries_succeeded) / queries_index)
                progress = int(100 * (queries_index) / queries_tot)
                print(">> progress: "+str(progress)+"% - errors: "+str(errors)+"%")
                #calc_process_time(starttime=start_time, cur_iter=queries_succeeded, max_iter=len(scan_points))
        end_time = time.time()
        duration = end_time - start_time

        logging.info("For comune %s : %d QUERES LAST FOR  %s  seconds" % (str(id_comune), len(scan_points), str(duration)))

        self.cursor.execute("SELECT pg_size_pretty( pg_database_size('cadastredb'));")
        logging.info("For comune %s %s QUERIES SET DATABASE MEMORY TO %s" % (str(id_comune), len(scan_points), self.cursor.fetchall()))
        self.connection.commit()
        self.cursor.close()

        return True

    def stop(self):
        self.connection.close()

    def generate_points(self, id_comune,  _n_max=MAX_POINTS, _distance=DISTANCE_SAMPLING):
        if id_comune is None:
            logging.critical("Id comune not assigned in generate_point")
            sys.exit(1)
        logging.info("MAX NUMBER OF POINT BY CONFIG: "+str(_n_max))
        logging.info("PRECISION OF SAMPLING DISTANCE in meter: "+str(_distance))
        self.cursor.execute("SELECT  geom FROM comuni WHERE id='{0}';".format(str(id_comune)))
        _res = self.cursor.fetchall()
        if _res is None or _res[0][0] is None:
            logging.critical("comune "+str(id_comune)+" non esiste nel db, riprova scemo!")
            raise ValueError("comune id non esiste nel db, riprova scemo!")
        elif len(_res) > 1:
            logging.critical("comune " + str(id_comune) + " risulta duplicato, wwtf?!")
            raise ValueError("Codice comune risulta dublicato, wtf?")


        poly = wkb.loads(_res[0][0], hex=True)
        minx, miny, maxx, maxy = poly.bounds
        _x = np.arange(minx, maxx, meters2degrees(_distance))
        _y = np.arange(miny, maxy, meters2degrees(_distance))

        _xy = np.meshgrid(_x, _y)
        mat = np.array(_xy).transpose()
        p_array = np.reshape(mat, (1,-1,2))
        logging.info("PUNTI TOTALI NEL COMUNE " + str(id_comune)+" : " + str(len(p_array[0])))

        count = 0
        _scan_points = []
        while count < len(p_array[0]):
            _point = Point(p_array[0][count][0], p_array[0][count][1])
            if poly.contains(_point):
                _scan_points.append([p_array[0][count][0],p_array[0][count][1]])
            count += 1
        return _scan_points
        # print("PUNTI SCELTI: " + str(len(scan_points)))

    def reset(self):
        return True
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
        data_info = None
        timeout_start = time.time()
        while time.time() < timeout_start + QUERY_CONNECTION_TIMEOUT:
            data_info = self.wms.get_info(_bbox_hint)
            if data_info is not None:
                break
            time.sleep(0.5)

        if data_info is None:
            logging.error("Error: No data received for point: LAT %s - LON: %s" % (str(lat), str(lon)))
            return False

        comune, foglio, particella = parse_html_response(data_info)
        if comune == "na" and foglio == "na" and particella == "na":
            logging.error("Na na na found in parsing comune, foglio, particella for point: LAT %s - LON: %s " % (str(lat), str(lon)))
            return True
        if comune is None or foglio is None or particella is None:
            logging.error("Error: Issue parsing comune, foglio, particella for point: LAT %s - LON: %s with html data: + %s" % (str(lat), str(lon), str(data_info)))
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
        data_bbox = None
        timeout_start_bbox = time.time()
        while time.time() < timeout_start_bbox + QUERY_CONNECTION_TIMEOUT:
            data_bbox = self.wms.get_geom(_bbox_hint)
            if data_bbox is not None:
                break
            time.sleep(0.5)

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
        img_title = comune + "_" + foglio + "_" + particella + "_"+str(lat)+"_"+str(lon)
        xy = compute_shape_from_map_image(image, title=img_title)

        if xy is None:
            logging.error("Error: unable to get geometry for: %s, %s, %s with bbox: %s, with distance %s" % (comune, foglio, particella, str(_bbox_rcv), str(distance(_bbox_rcv.lat1, _bbox_rcv.lon1, _bbox_rcv.lat2, _bbox_rcv.lon2))) )
            return False
        coords = reproj(_xy=xy, _size=_new_size, _bbox=_bbox_rcv)

        if coords is None:
            logging.error("Error: unable to reproject coordiantes in ESRI system")
            return False

        poly = geometry.Polygon([[p.lat, p.lon] for p in coords])
        # print(poly.wkt)
        self.cursor.execute('UPDATE particelle SET geom=ST_GeomFromText(ST_AsText(%s),%s) WHERE  comune=%s AND foglio=%s AND particella=%s;', (poly.wkt, str(self.srs), comune, foglio, particella))
        #self.cursor.execute("UPDATE particelle SET geom VALUES (ST_AsEWKT(%s));", (poly.wkb,))
        #self.cursor.execute('INSERT INTO particelle (geom) VALUES (ST_GeomFromText(ST_AsText(%s),%s))', (poly.wkt, str(self.srs)))

        # print("save: %s, %s, %s " % (comune, foglio, particella))

        return True


if __name__ == '__main__':

    c = CatastoQueryTool()
    
    try:
        c.run()
    except KeyboardInterrupt:
        c.stop()
        print("STOOOOOPEEE!")
        sys.exit(1)

    #c.query_point(lat=43.72028570195715, lon=11.285161133038065)
