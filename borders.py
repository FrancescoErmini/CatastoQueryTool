#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Francesco Ermini'

import os, sys
import csv
import logging
import datetime
import fiona
from fiona.crs import from_epsg
import pyproj
from shapely import geometry
from shapely.ops import transform
from shapely import geometry, wkt, wkb
from shapely.geometry import shape, Point, LineString, Polygon

from math import cos, asin, sqrt, pi
import numpy as np
import psycopg2
from psycopg2 import sql
import cv2

LOG_LEVEL = logging.INFO
log_file = "logs/log_" + str(datetime.datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")) + ".log"
logging.basicConfig(filename=log_file, filemode='w+', format='%(message)s', level=logging.INFO)

"""
09;248;048; 001 ;048001;Bagno a Ripoli;Bagno a Ripoli;;3;Centro;Toscana;Firenze;0;FI;48001;48001;48001;48001;A564;25.403;ITI;ITI1;ITI14
"""
# Siglaautomobilistica;CodiceComune
BORDERS_SHAPE_COMUNI = os.path.join("data", "borders", "it", "comuni", "Com01012018_WGS84.shp")
BORDERS_CSV_COMUNI = os.path.join("data", "borders", "it", "comuni", "comuni.csv")


def meters2degrees(_meters):
	return round(_meters * 0.00001, 5)


class Borders:

	def __init__(self):
		self.srs = 4258

	def create(self):

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
		#self.cursor.execute("DROP TABLE IF EXISTS comuni;")
		self.cursor.execute(create_comuni)

		with open(BORDERS_CSV_COMUNI, mode='r', encoding='latin-1') as csv_file:
			csv_reader = csv.DictReader(csv_file, delimiter=";")
			for row in csv_reader:
				id = row["CodiceComuneAlfaNumerico"]  # cod_istat_comune
				cod_catastale_comune = row["CodiceCatastale"]
				comune = row["DenominazioneComune"]

				provincia = row["DenominazioneProvincia"]
				cod_istat_provincia = row["CodiceProvincia"]
				sigla_provincia = row["SiglaAuto"]

				regione = row["DenominazioneRegione"]
				cod_istat_regione = row["CodiceRegione"]

				self.cursor.execute('''\
					INSERT INTO comuni (id, comune, cod_catastale_comune, provincia, sigla_provincia, cod_istat_provincia, regione, cod_istat_regione) \
					VALUES (%s,%s,%s,%s,%s,%s,%s,%s);
					''', (id, comune, cod_catastale_comune, provincia, sigla_provincia, cod_istat_provincia, regione,
						  cod_istat_regione)
									)

		self.connection.commit()

		project = pyproj.Transformer.from_proj(
			pyproj.Proj(init='epsg:32632'),  # source coordinate system
			pyproj.Proj(init='epsg:4258'))  # destination coordinate system

		with fiona.open(BORDERS_SHAPE_COMUNI) as input:

			for feature in input:
				shape_id = feature['properties']['PRO_COM_T']
				geom = shape(feature['geometry'])
				new_geom = transform(project.transform, geom)

				self.cursor.execute('''\
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

	def create_grid(self, id_comune, _distance=1000):
		self.connection = psycopg2.connect(dbname='cadastredb', user='biloba', host='127.0.0.1', password='biloba')
		self.cursor = self.connection.cursor()

		create_grid = """CREATE TABLE IF NOT EXISTS grid (\
				id SERIAL PRIMARY KEY,
				comune VARCHAR(8),
				sampling_level INTEGER,
				query_result INTEGER,
				point GEOGRAPHY(POINT, 4326)
				
		);"""

		create_scan_results = """CREATE TABLE IF NOT EXISTS results (\
						id SERIAL PRIMARY KEY,
						comune VARCHAR(8),
						grid_points INTEGER,
						queries INTEGER,
						errors INTEGER,
						duration DECIMAL,
						last_scan TIMESTAMP
				);"""

		self.cursor.execute("DROP TABLE IF EXISTS grid;")
		self.cursor.execute(create_grid)
		self.cursor.execute("DROP TABLE IF EXISTS results;")
		self.cursor.execute(create_scan_results)


		if id_comune is None:
			logging.critical("Id comune not assigned in generate_point")
			sys.exit(1)
		logging.info("PRECISION OF SAMPLING DISTANCE in meter: " + str(_distance))
		self.cursor.execute("SELECT  geom FROM comuni WHERE id='{0}';".format(str(id_comune)))
		_res = self.cursor.fetchall()
		if _res is None or _res[0][0] is None:
			logging.critical("comune " + str(id_comune) + " non esiste nel db, riprova scemo!")
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
		p_array = np.reshape(mat, (1, -1, 2))
		logging.info("PUNTI TOTALI NEL COMUNE " + str(id_comune) + " : " + str(len(p_array[0])))

		count = 0
		_scan_points = []
		while count < len(p_array[0]):
			_point = Point(p_array[0][count][0], p_array[0][count][1])
			if poly.contains(_point):
				#_scan_points.append([p_array[0][count][0], p_array[0][count][1]])
				_scan_points.append(
					_point
				)
			count += 1

		sampling_level = 5
		i = 0
		for point in _scan_points:
			level = i % sampling_level
			i += 1
			self.cursor.execute('''\
								INSERT INTO grid (comune, sampling_level, query_result, point) \
								VALUES (%s, %s, %s, %s::geometry);
								''', (str(id_comune),level, 0, str(point))
								)

		self.cursor.execute('''\
												INSERT INTO results (comune, grid_points) \
												VALUES (%s, %s);
												''', (str(id_comune), len(_scan_points))
							)

		self.connection.commit()
		self.cursor.close()
		self.connection.close()







	def shape_inspect(self):
		shapefile = fiona.open(BORDERS_SHAPE_COMUNI)
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




	def dump_regioni(self):
		connection = psycopg2.connect(dbname='cadastredb', user='biloba', host='127.0.0.1', password='biloba')
		cursor = connection.cursor()
		cursor.execute("SELECT DISTINCT regione FROM comuni;")
		results = cursor.fetchall()
		_regioni = [res[0] for res in results]
		print("end get regioni")
		cursor.close()
		connection.close()
		return _regioni

	def dump_comuni(self, regione):
		if self.connection is None:
			self.connection = psycopg2.connect(dbname='cadastredb', user='biloba', host='127.0.0.1',
											   password='biloba')
		if self.cursor is None:
			self.cursor = self.connection.cursor()

		_query_str = f"SELECT id, comune FROM comuni WHERE regione='{regione}' AND geom is not NULL;"
		self.cursor.execute(_query_str)
		_comuni = self.cursor.fetchall()
		if _comuni is None:
			import sys
			logging.critical("Unable to read the db of comuni, annammo bene..")
			sys.exit(1)
		i = 0
		# for _comune in _comuni:
		# 	i += 1
		# 	print("\n############# comune: " + str(_comune[1]) + " - " + str(i) + "/" + str(
		# 		len(_comuni)) + "regione: " + str(self.regione) + " #############")
		# 	logging.info("\n############# comune: " + str(_comune[0]) + " #############")
		# 	self.scan(_comune[0])
		return _comuni


	# if __name__=='__main__':
b = Borders()
b.create_grid("048001")
