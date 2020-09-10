
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


import psycopg2
from owslib.wms import WebMapService
import cv2


from math import cos, asin, sqrt, pi
import numpy as np


CATASTO_ITALIA_SRS = 4258



def fix():
    connection = psycopg2.connect(dbname='cadastredb', user='biloba', host='127.0.0.1', password='biloba')
    cursor = connection.cursor()
    cursor.execute(f"SELECT cod_catastale_comune FROM comuni;")
    comuni = cursor.fetchall()
    for comune in comuni:
        cod_comune = comune[0]
        cursor.execute("SELECT id, bbox FROM particelle WHERE comune='%s';" % (cod_comune,))
        parcels = cursor.fetchall()
        for parcel in parcels:
            """
             self.cursor.execute("SELECT  geom FROM comuni WHERE id='{0}';".format(str(id_comune)))
            _res = self.cursor.fetchall()
            if _res is None or _res[0][0] is None:
                logging.critical("comune "+str(id_comune)+" non esiste nel db, riprova scemo!")
                raise ValueError("comune id non esiste nel db, riprova scemo!")
            elif len(_res) > 1:
                logging.critical("comune " + str(id_comune) + " risulta duplicato, wwtf?!")
                raise ValueError("Codice comune risulta dublicato, wtf?")
    
    
            poly = wkb.loads(_res[0][0], hex=True)
            """
            bbox_geom = parcel[1]
            parcel_id = parcel[0]
            if bbox_geom is None:
                continue
            old_poly = wkb.loads(bbox_geom, hex=True)
            old_poly_coords = old_poly.exterior.coords.xy
            old_lats = old_poly.exterior.coords.xy[0]
            old_lon = old_poly.exterior.coords.xy[1]
            new_bbox_coords = [
                [old_lon[0], old_lats[0]],
                [old_lon[1], old_lats[0]],
                [old_lon[1], old_lats[1]],
                [old_lon[0], old_lats[1]],
                [old_lon[0], old_lats[0]]
            ]
            valid_bbox(_bbox=new_bbox_coords)

            bbox_poly = geometry.Polygon(new_bbox_coords)
            cursor.execute(
                'UPDATE particelle_new SET bbox=ST_GeomFromText(ST_AsText(%s),%s) WHERE id=%s;',
                (bbox_poly.wkt, str(CATASTO_ITALIA_SRS), parcel_id))
            # print("saved bbox for: %s, %s, %s ", (comune, foglio, parti
            connection.commit()
    cursor.close()
    connection.close()

if __name__ == '__main__':
	fix()