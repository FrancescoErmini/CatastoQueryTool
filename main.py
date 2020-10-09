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
from enum import Enum
from dataclasses import dataclass

from shapely import geometry, wkt, wkb
from shapely.geometry import shape, Point, LineString, Polygon

from shapely.geometry import Point



import psycopg2
from psycopg2.extensions import AsIs
from owslib.wms import WebMapService
import cv2


from math import cos, asin, sqrt, pi
import numpy as np

LOG_LEVEL = logging.INFO
log_file = "logs/log_"+str(datetime.datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p"))+".log"
#logging.basicConfig(filename=log_file, filemode='w+', format='%(message)s', level=logging.INFO)
logging.basicConfig(level=logging.INFO)

QUICK_MODE = True
RESET_DB = True
DEBUG_IMAGE = False
DEBUG_IMAGE_SAVE = False
DEBUG_IMAGE_LIVE = False

ITALIA_WMS_URL = 'https://wms.cartografia.agenziaentrate.gov.it/inspire/wms/ows01.php'
CATASTO_ITALIA_SRS = 'EPSG:4258'
CATASTO_ITALIA_LAYER_PARTICELLE = 'CP.CadastralParcel'
CATASTO_ITALIA_LAYER_FOGLI = 'CP.CadastralZoning'
MAX_CADASTRE_SCALE_THRESHOLD = 200.0 # metri oltre i quali il catasto mostra una immagine bianca (troppo zoom out)

DISTANCE_SAMPLING = 200 #meters between points

IMG_PIXEL_WIDTH = 200
PRINT_UPDATES_EVERY_N_QUERY = 100
QUERY_CONNECTION_TIMEOUT = 10


DBNAME='cadastredb'
DBUSER='biloba'
DBHOST='127.0.0.1'
DBPASSWORD='biloba'


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


class QueryResult(Enum):
    UNSET = 0  # never scan that point
    SUCCEED = 1  # scan of point succeed ( return bbox or geom )
    # errors
    NO_WMS_RESPONSE_GET_INFO = -1  # No response to the wms query
    NO_WMS_RESPONSE_GET_GEOM = -2  # No response to the wms query
    ERROR_VALIDATE_LATLON = -10
    ERROR_VALIDATE_BBOX = -11
    ERROR_PARSING_HTML = -21
    ERROR_PARSING_GEOM = -22
    # point to skip
    NOT_CADASTRAL = 3  # the query return a html respose but there is noting to capture ( river, or non particles ).
    DUPLICATED = 2     # the query return a parcel that is already known in db





def valid_bbox(_bbox):
    """
     new_bbox_coords = [
                [old_lon[0], old_lats[0]],
                [old_lon[1], old_lats[0]],
                [old_lon[1], old_lats[1]],
                [old_lon[0], old_lats[1]],
                [old_lon[0], old_lats[0]]
            ]
    Args:
        _bbox:

    Returns:

    """
    #_bbox[0][1] == _bbox[1][1]
    # check the leght of polygon is 5 points
    if len(_bbox) != 5:
        raise Exception("no a valid bbox")
    # check lon/lat ( array of 2 ) for each point
    for point in _bbox:
        if len(point) != 2:
            raise Exception("A point must be an array of lon/lat")
    # the last point is equal to the first point
    if not (_bbox[0] == _bbox[4]):
        raise Exception("no a valid bbox: first point and last point must be the same")
    # check the two lon are not inverted ( for the italy )
    if not (5.0 < _bbox[0][0] < 19.0 and 5.0 < _bbox[1][0] < 19.0):
        raise Exception("no a valid bbox: lon coords not inside italy bounds")
    if not (35.0 < _bbox[0][1] < 50.0 and 35.0 < _bbox[2][1] < 50.0):
        raise Exception("no a valid bbox: lat coords not inside italy bounds")
    # check  lons are ordered ( for the italy )
    if not (_bbox[0][0] == _bbox[3][0] and _bbox[1][0] == _bbox[2][0]):
        raise Exception("no a valid bbox: lon coords not orderd")
    # check  lons are ordered ( for the italy )
    if not (_bbox[0][1] == _bbox[1][1] and _bbox[2][1] == _bbox[3][1]):
        raise Exception("no a valid bbox: lon coords not orderd")
    return _bbox

def parse_html_response(html_string, layer):
    if html_string is None:
        return None
    #if html_string == 'b\'Content-Type: text/html\\r\\n\\r\\n\''
    if "<td>" not in html_string:
        return "na", "na", "na"
    try:
        if layer=="fogli":
            # <td>IT.AGE.MAP.A564_002800</td>
            res = re.findall(r'<td>IT.AGE.MAP.(.*?)</td>', html_string, re.M | re.I | re.S)
        else:
            res = re.findall(r'NationalCadastralReference</th><td>(.*?)</td>', html_string, re.M | re.I | re.S)
        codice = res[0]
        comune = codice[:4]
        foglio = codice[5:9]
        particella = "na" if layer == "fogli" else codice.split(".")[1]
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

    def __init__(self, base_url, srs, layer, version):
        self.wms = WebMapService(base_url, version=version)
        self.srs = srs
        self.layer = layer

    # def _sanity_check(self, table):
    #     if table not in self.wms.contents.keys():
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


class QueryPoint:
    def __init__(self, uuid, lon, lat):
        self.uuid = uuid
        self.lon = lon
        self.lat = lat



class CatastoQueryTool:

    def __init__(self, comune, table, level=5):
        if not (table == "fogli" or table == "particella"):
            logging.error("table puo essere foglio o particella non altri valori")
            sys.exit(1)
        self.srs = 4258
        self.wms = WMSTool(
            base_url=ITALIA_WMS_URL,
            srs=CATASTO_ITALIA_SRS,
            layer=CATASTO_ITALIA_LAYER_FOGLI if table == "fogli" else CATASTO_ITALIA_LAYER_PARTICELLE,
            version='1.3.0'
        )
        self.connection = None
        self.cursor = None
        self.comune = comune
        self.table = table
        self.points = []
        self.level = level

    def populate(self):
        """
        estrapola i punti dalla tabella grid
        """
        with psycopg2.connect(dbname=DBNAME, user=DBUSER, host=DBHOST, password=DBPASSWORD) as conn:
            with conn.cursor() as cur:
                #AND sampling_level<%s AND query_result<1;
                cur.execute("SELECT id,point FROM grid WHERE comune=%s", (self.comune,)) #str(self.level)))
                query_points = cur.fetchall()
                for uuid, point_wkt in query_points:
                    point_geom = wkb.loads(point_wkt, hex=True)
                    self.points.append(QueryPoint(uuid=uuid, lon=point_geom.x, lat=point_geom.y))

    def scan(self):
        self.connection = psycopg2.connect(dbname=DBNAME, user=DBUSER, host=DBHOST, password=DBPASSWORD)
        self.cursor = self.connection.cursor()
        start_time = time.time()
        queries_succeeded = 0
        queries_index = 0
        queries_tot = len(self.points)
        logging.info("START TIME: "+str(datetime.datetime.fromtimestamp(start_time).strftime("%d/%m/%Y, %H:%M:%S")))
        logging.info("TOT POINTS: "+str(queries_tot))

        if queries_tot == 0:
            logging.critical("Non ci sono punti da interrogare, bona ci si!")
            return False

        for p in self.points:
            queries_index += 1
            _result = 0
            try:
                _result = self.query_point(lon=p.lon, lat=p.lat)
                _result = _result.value
            except Exception as e:
                logging.error("Uncautch exception in query_point: "+str(e))
                pass
            print(_result)
            self.cursor.execute("UPDATE grid SET query_result=%s WHERE id=%s;", (_result, p.uuid))

            if _result > 0:
                queries_succeeded += 1
            if int(queries_index) % int(PRINT_UPDATES_EVERY_N_QUERY) == 0:
                #self.connection.commit()
                errors = int(100 * (queries_index-queries_succeeded) / queries_index)
                progress = int(100 * (queries_index) / queries_tot)
                #print(">> progress: "+str(progress)+"% - errors: "+str(errors)+"%")
                #calc_process_time(starttime=start_time, cur_iter=queries_succeeded, max_iter=len(scan_points))
        queries_error = queries_tot - queries_succeeded
        end_time = time.time()
        duration = end_time - start_time

        self.cursor.execute("UPDATE results SET queries=%s, errors=%s, duration=%s, last_scan=%s WHERE comune=%s;",
                            (queries_tot, queries_error, duration, datetime.datetime.now(), self.comune))
        self.connection.commit()
        self.cursor.close()
        self.connection.close()

        #self.cursor.execute("SELECT pg_size_pretty( pg_database_size('cadastredb'));")
        #logging.info("For comune %s %s QUERIES SET DATABASE MEMORY TO %s" % (str(self.comune), len(self.points), self.cursor.fetchall()))

        logging.info(
            "For comune %s : %d QUERES LAST FOR  %s  seconds" % (str(self.comune), len(self.points), str(duration)))

        return True

    def stop(self):
        self.cursor.close()
        self.connection.close()


    def create(self):
        if RESET_DB:
            pass
            #drop_particelle = "DROP TABLE IF EXISTS particelle;"

        create_particelle = "CREATE TABLE IF NOT EXISTS particelle (id SERIAL PRIMARY KEY, \
                                                    comune VARCHAR(64), foglio VARCHAR(64), particella VARCHAR(64),\
                                                    bbox GEOMETRY,\
                                                    geom GEOMETRY,\
                                                    updated TIMESTAMP DEFAULT NOW()\
                                                    );"
        create_fogli = "CREATE TABLE IF NOT EXISTS fogli (id SERIAL PRIMARY KEY, \
                                                            comune VARCHAR(64), foglio VARCHAR(64), particella VARCHAR(64),\
                                                            bbox GEOMETRY,\
                                                            geom GEOMETRY,\
                                                            updated TIMESTAMP DEFAULT NOW()\
                                                            );"
        with psycopg2.connect(dbname=DBNAME, user=DBUSER, host=DBHOST, password=DBPASSWORD) as conn:
            with conn.cursor() as cur:
                # self.cursor.execute(drop_particelle)
                cur.execute(create_particelle)
                cur.execute(create_fogli)

    def query_point(self, lat, lon, store=True):

        if 30.0 < lat < 54.0 and 4.0 < lon < 20.0:
            pass
        else:
            return QueryResult.ERROR_VALIDATE_LATLON

        if self.cursor is None:
            self.cursor = self.connection.cursor()

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
            return QueryResult.NO_WMS_RESPONSE_GET_INFO

        comune, foglio, particella = parse_html_response(data_info, self.table)
        if comune == "na" and foglio == "na" and particella == "na":
            logging.error("Na na na found in parsing comune, foglio, particella for point: LAT %s - LON: %s " % (str(lat), str(lon)))
            return QueryResult.NOT_CADASTRAL
        if comune is None or foglio is None or particella is None:
            logging.error("Error: Issue parsing comune, foglio, particella for point: LAT %s - LON: %s with html data: + %s" % (str(lat), str(lon), str(data_info)))
            return QueryResult.ERROR_PARSING_HTML

        self.cursor.execute('SELECT 1 from %s WHERE comune=%s AND foglio=%s AND particella=%s', (AsIs(self.table), comune, foglio, particella))
        if self.cursor.fetchone() is not None:
            logging.debug("particella duplicata: %s, %s, %s " % (comune, foglio, particella))
            return QueryResult.DUPLICATED

        logging.debug("NUOVA PARTICELLA: %s, %s, %s " % (comune, foglio, particella))
        self.cursor.execute('INSERT INTO %s (comune,foglio,particella) VALUES (%s, %s, %s)', (AsIs(self.table), comune, foglio, particella))

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
            return QueryResult.NO_WMS_RESPONSE_GET_GEOM

        _bbox_rcv = parse_geom_response(data_bbox)

        if _bbox_rcv is None:
            logging.error("Error: Issue parsing bbox data for comune %s, foglio %s, particella %s" % (comune, foglio, particella))
            return QueryResult.ERROR_PARSING_GEOM

        bbox_coords = [
            [_bbox_rcv.lon1, _bbox_rcv.lat1],
            [_bbox_rcv.lon2, _bbox_rcv.lat1],
            [_bbox_rcv.lon2, _bbox_rcv.lat2],
            [_bbox_rcv.lon1, _bbox_rcv.lat2],
            [_bbox_rcv.lon1, _bbox_rcv.lat1]
        ]

        try:
            valid_bbox(bbox_coords)
        except Exception:
            return QueryResult.ERROR_VALIDATE_BBOX

        bbox_poly = geometry.Polygon(bbox_coords)
        self.cursor.execute('UPDATE %s SET bbox=ST_GeomFromText(ST_AsText(%s),%s) WHERE comune=%s AND foglio=%s AND particella=%s;',
                            (AsIs(self.table), bbox_poly.wkt, str(self.srs), comune, foglio, particella))
        #print("saved bbox for: %s, %s, %s ", (comune, foglio, particella))

        if QUICK_MODE:
            return QueryResult.SUCCEED

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
        if len(coords) < 3:
            logging.error("xy coordinates must be at leat 3 to create a polygon")

        poly = geometry.Polygon([[p.lat, p.lon] for p in coords])
        # print(poly.wkt)
        self.cursor.execute('UPDATE particelle SET geom=ST_GeomFromText(ST_AsText(%s),%s) WHERE  comune=%s AND foglio=%s AND particella=%s;', (poly.wkt, str(self.srs), comune, foglio, particella))
        #self.cursor.execute("UPDATE particelle SET geom VALUES (ST_AsEWKT(%s));", (poly.wkb,))
        #self.cursor.execute('INSERT INTO particelle (geom) VALUES (ST_GeomFromText(ST_AsText(%s),%s))', (poly.wkt, str(self.srs)))

        # print("save: %s, %s, %s " % (comune, foglio, particella))

        return True
#
# from threading import Thread
#
# class CatastoThread(Thread):
#     def __init__(self, regione):
#         Thread.__init__(self)
#         self.catasto = CatastoQueryTool(regione=regione)
#
#     def run(self):
#         try:
#             self.catasto.run()
#         except KeyboardInterrupt:
#             self.catasto.stop()
#             print("STOOOOOPEEE!")
#             sys.exit(1)
#
#
# def dump_regioni():
#     connection = psycopg2.connect(dbname='cadastredb', user='biloba', host='127.0.0.1', password='biloba')
#     cursor = connection.cursor()
#     cursor.execute("SELECT DISTINCT regione FROM comuni;")
#     results = cursor.fetchall()
#     _regioni = [res[0] for res in results]
#     print("end get regioni")
#     cursor.close()
#     connection.close()
#     return _regioni

def grouper(n, iterable, fillvalue=None):
    from itertools import zip_longest
    args = [iter(iterable)]*n
    return zip_longest(*args, fillvalue=fillvalue)


if __name__ == '__main__':
    c = CatastoQueryTool(comune='048001', table='fogli', level=1)
    c.create()
    c.populate()
    c.scan()
    # all_regions = dump_regioni()
    #
    #
    # for grouped_regions in grouper(n=N_REGIONI_PARALLELO, iterable=all_regions):
    #     # avoid None in list of regions
    #     parallel_regions = [r for r in list(grouped_regions) if r is not None]
    #
    #     print("INIT-ora fo le regioni:")
    #     print(parallel_regions)
    #
    #     # crate parallels threads and wait to complete
    #     threads = []
    #     for region in parallel_regions:
    #         t = CatastoThread(regione=region)
    #         threads.append(t)
    #     print("creati "+str(len(threads))+" threads")
    #
    #     # start multi threading for N regioni
    #     for th in threads:
    #         th.start()
    #
    #     # active wait threads to complete before new parallel run
    #     for th in threads:
    #         th.join()
    #     print("finiti ithreads per regioni")
    #     print(parallel_regions)
