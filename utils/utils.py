import os
import time
import datetime
import random
import fiona
from dataclasses import dataclass
from shapely import geometry
from math import cos, asin, sqrt, pi
import numpy as np
import cv2

DEBUG_IMAGE = False

MAX_CADASTRE_SCALE_THRESHOLD = 200.0

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


def distance(lat1, lon1, lat2, lon2, unit='mt'):
    p = 0.017453292519943295  # Pi/180
    a = 0.5 - cos((lat2 - lat1) * p) / 2 + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
    res = 12742 * asin(sqrt(a)) # km
    if unit == 'mt':
        res = res * 1000
    return res


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
    if _d_v > MAX_CADASTRE_SCALE_THRESHOLD  or _d_h > MAX_CADASTRE_SCALE_THRESHOLD:
        return True
    return False


def meters2degrees(_meters):
    return round(_meters * 0.00001, 5)


def degrees2meters(_degrees):
    return _degrees * 100000


def shape_inspect(path):
    shapefile = fiona.open(path)
    # Make sure the dataset exists -- it would be None if we couldn't open it
    if not shapefile:
        print('Error: could not open shapefile')
    driver = shapefile.driver
    print('Dataset driver is: {n}\n'.format(n=driver))

    ### How many features are contained in this Shapefile?
    feature_count = len(shapefile)
    print('The shapefile has {n} feature(s)\n'.format(n=feature_count))

    ### What is the shapefiles's projection?
    # Get the spatial reference
    spatial_ref = shapefile.crs
    print('The shapefiles spatial ref is:\n', spatial_ref, '\n')

    # Let's pull out a specific feature from the shapefile
    feature = shapefile[0]

    ### What is the features's geometry? is it a point? a polyline? a polygon?
    geometry = feature['geometry']['type']
    print("The features's geometry is: {geom}\n".format(geom=geometry))

    ### How many properties are in the shapefile, and what are their names?
    properties = feature["properties"].keys()

    # How many fields
    field_count = len(properties)
    print('Layer has {n} fields'.format(n=field_count))

    # What are their names?
    print('Their names are: ')
    for prop in properties:
        print('\t{name}'.format(name=prop))
    for feature in shapefile:
        if feature['properties']['COMUNE'] == 'Bagno a Ripoli':
            print(feature)
"""
BBOX UTILS
"""


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

"""
OPENCV IMAGE UTILS
"""

def compute_shape_from_map_image(image):
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
            # cv2.rectangle(orig_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # cv2.imshow('Bounding rect', orig_image)
            if abs(w - size_x) < 10.0 and abs(h - size_y) < 10.0:
                new_contours.append(c)

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
                import datetime
                current_time = datetime.datetime.now()
                name = "img_fail_{}.png".format(current_time)
                img_path = os.path.join("../imgs", name)
                # cv2.imwrite(img_path, image);
                cv2.imshow(name, image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            return None
        # se esistono più poligoni, seleziona quello con area minore.
        if len(new_contours) > 1:
            area = cv2.contourArea(c_opt)
            for c in new_contours[1:]:
                new_area = cv2.contourArea(c)
                if new_area < area:
                    # print (" nuovo: " + str(new_area))
                    # print (" vecchio: " + str(area))
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
            cv2.drawContours(image, [approx], 0,
                             (random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255)), 2)
            # marca il punti 0,0  per riferimento.
            cv2.circle(image, (0, 0), 3, (155, 243, 198), -1)
            for i in approx:
                x, y = i.ravel()
                cv2.circle(image, (x, y), 3, (255, 0, 0), -1)
            cv2.imshow('Approx polyDP', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        """
        7. Ritorna una lista di tuple contenete le coordinate x y dei punti trovati.
        """

        return xy

    except Exception:
        return None

def calc_process_time(starttime, cur_iter, max_iter):
    telapsed = time.time() - starttime
    testimated = (telapsed/cur_iter)*(max_iter)

    finishtime = starttime + testimated
    finishtime = datetime.datetime.fromtimestamp(finishtime).strftime("%d/%m/%Y, %H:%M:%S")  # in time

    lefttime = testimated-telapsed  # in seconds
    print("time elapsed: %s(s), time left: %s(s), estimated finish time: %s" % (int(telapsed), int(lefttime), finishtime))
