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

import psycopg2
from psycopg2 import sql
import cv2



"""
09;248;048; 001 ;048001;Bagno a Ripoli;Bagno a Ripoli;;3;Centro;Toscana;Firenze;0;FI;48001;48001;48001;48001;A564;25.403;ITI;ITI1;ITI14
"""
#Siglaautomobilistica;CodiceComune
BORDERS_SHAPE_COMUNI = os.path.join("data", "borders", "it", "comuni", "Com01012018_WGS84.shp")
BORDERS_CSV_COMUNI =   os.path.join("data", "borders", "it", "comuni", "comuni.csv")


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


if __name__=='__main__':
	b = Borders()
	b.comuni()
