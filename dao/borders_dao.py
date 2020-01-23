#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Francesco Ermini'
import os
import csv

import fiona
from fiona.crs import from_epsg
import pyproj
from shapely import geometry
from shapely.ops import transform
from shapely.geometry import shape, Point, LineString, Polygon
import logging
import psycopg2
from shapely import wkb
import numpy as np
from utils.utils import meters2degrees
from psycopg2 import sql
import cv2
"""
BordersDAO e' la classe che si occupa di creazione, inserimento e cancellazione dei confini nazionali:


"""
"""
09;248;048; 001 ;048001;Bagno a Ripoli;Bagno a Ripoli;;3;Centro;Toscana;Firenze;0;FI;48001;48001;48001;48001;A564;25.403;ITI;ITI1;ITI14
"""

BORDERS_SHAPE_COMUNI = os.path.join("../data", "borders", "it", "comuni", "Com01012018_WGS84.shp")
BORDERS_CSV_COMUNI = os.path.join("../data", "borders", "it", "comuni", "comuni.csv")


class Borders:

	def __init__(self):
		self.srs = 4258

	def comuni(self):

		"""
		1. cancella e ricrea tabella comuni
		"""

		self.connection = psycopg2.connect(dbname='cadastredb', user='biloba', host='127.0.0.1', password='biloba')
		self.cursor = self.connection.cursor()

		create_comuni = "\
			CREATE TABLE comuni (\
				id CHAR(6) PRIMARY KEY,\
				comune VARCHAR(100),\
				cod_catastale_comune CHAR(4),\
				provincia VARCHAR(50),\
				sigla_provincia CHAR(2),\
				cod_istat_provincia CHAR(3),\
				regione VARCHAR(50),\
				cod_istat_regione CHAR(2),\
				geom GEOMETRY,\
				created TIMESTAMP DEFAULT NOW());\
			"
		self.cursor.execute("DROP TABLE IF EXISTS comuni;")
		self.cursor.execute(create_comuni)


		with open(BORDERS_CSV_COMUNI, mode='r', encoding='latin-1') as csv_file:
			csv_reader = csv.DictReader(csv_file,delimiter=";")
			for row in csv_reader:

				id = row["CodiceComuneAlfaNumerico"] #cod_istat_comune
				cod_catastale_comune = row["CodiceCatastale"]
				comune = row["DenominazioneComune"]

				provincia = row["DenominazioneProvincia"]
				cod_istat_provincia = row["CodiceProvincia"]
				sigla_provincia = row["SiglaAuto"]

				regione = row["DenominazioneRegione"]
				cod_istat_regione = row["CodiceRegione"]

				self.cursor.execute( '''\
					INSERT INTO comuni (id, comune, cod_catastale_comune, provincia, sigla_provincia, cod_istat_provincia, regione, cod_istat_regione) \
					VALUES (%s,%s,%s,%s,%s,%s,%s,%s);
					''', (id, comune, cod_catastale_comune, provincia, sigla_provincia, cod_istat_provincia, regione, cod_istat_regione)
				)

		self.connection.commit()


		project = pyproj.Transformer.from_proj(
		pyproj.Proj(init='epsg:32632'), # source coordinate system
		pyproj.Proj(init='epsg:4258')) # destination coordinate system


		with fiona.open(BORDERS_SHAPE_COMUNI) as input:


			for feature in input:
				shape_id = feature['properties']['PRO_COM_T'] 
				geom = shape(feature['geometry'])
				new_geom = transform(project.transform, geom)

				self.cursor.execute( '''\
					UPDATE comuni 
					SET geom=%s
					WHERE id=%s;
					''', (new_geom.wkt, shape_id))

		"""
		self.cursor.execute("\
			ALTER TABLE comuni \
			ALTER COLUMN geom \
			USING ST_Transform(ST_SetSRID( geom,32632 ), 4326 );\
		")
		"""
		self.connection.commit()

		self.cursor.close()
		self.connection.close()

	def generate_points(self, _distance):
		logging.info("PRECISION OF SAMPLING DISTANCE in meter: " + str(_distance))
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
		p_array = np.reshape(mat, (1, -1, 2))
		logging.info("PUNTI TOTALI NEL COMUNE: " + str(len(p_array[0])))

		count = 0
		while count < len(p_array[0]):
			_point = Point(p_array[0][count][0], p_array[0][count][1])
			if poly.contains(_point):
				self.points.append([p_array[0][count][0], p_array[0][count][1]])
			count += 1
		logging.info("PUNTI SCELTI: " + str(len(self.points)))



if __name__=='__main__':
	b = Borders()
	b.comuni()
