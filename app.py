import psycopg2
from shapely import wkb
from utils.shapely_geojson import Feature, dumps
from psycopg2 import extensions, sql


import os
import json
from flask import Flask, request, jsonify, abort
from flask import render_template
app = Flask(__name__)



def get_particella(cadastral_parcel):
	# A564_0024.356

	connection = psycopg2.connect(dbname='cadastredb', user='biloba', host='127.0.0.1', password='biloba')
	with connection:
		with connection.cursor() as cursor:
			try:
				comune = cadastral_parcel[:4]
				foglio = cadastral_parcel[5:9]
				particella = cadastral_parcel.split(".")[1]
				print(comune+" "+foglio+" "+particella)
				#_sql2 = "SELECT bbox FROM particelle WHERE id=1;"
				_sql = "SELECT bbox FROM particelle_new WHERE comune='{comune}' AND foglio='{foglio}' AND particella='{particella}';".format(
					comune=comune,
					foglio=foglio,
					particella=particella)

				cursor.execute(_sql)
				res = cursor.fetchall()
				wkb_geom = res[0][0]
				geom = wkb.loads(wkb_geom, hex=True)
				feature = Feature(geom, properties={'comune': comune, 'foglio': foglio, 'particella': particella})
				return dumps(feature)

			except Exception:
				return None



@app.route('/parcel/<string:particella>', methods=['GET'])
def get_particle_geom(particella):

	geojson = get_particella(particella)
	#{'type': 'FeatureCollection', 'features': []}
	if geojon is None:
		abort(404, description="parcel not found")
	
	return jsonify(geojon)





#print(get_particella("H944_0001.662"))
if __name__ == "__main__":
	app.run(host='0.0.0.0')