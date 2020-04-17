import psycopg2
from shapely import wkb
from utils.shapely_geojson import Feature, dumps
from psycopg2 import extensions, sql

from config import BORDERS_CSV_COMUNI, BORDERS_SHAPE_COMUNI

query_particella = "SELECT jsonb_build_object(\
    'type',       'Feature',\
    'gid',         id,\
    'geometry',   ST_AsGeoJSON(geom)::jsonb,\
    'properties', to_jsonb(row) - 'id' - 'geom'\
) FROM (SELECT * FROM particelle WHERE id='1') row;"

query_particella2 = "SELECT ST_AsGeoJSON(t.*)\
                        FROM ()\
                    AS t(id, name, geom);\
                   "

EXECUTION_ERROR = "error"
EXECUTION_SUCCEED = "success"

class DAO:

    def __init__(self):
        self.connection = psycopg2.connect(dbname='cadastredb', user='biloba', host='127.0.0.1', password='biloba')
        autocommit = extensions.ISOLATION_LEVEL_AUTOCOMMIT
        print("ISOLATION_LEVEL_AUTOCOMMIT:", extensions.ISOLATION_LEVEL_AUTOCOMMIT)
        self.connection.set_isolation_level(autocommit)

    def __del__(self):
        self.connection.close()

    def __query_response(self, execution, value=None):
        return {"execution": EXECUTION_ERROR if execution else EXECUTION_SUCCEED,
                "value": value}

    def _query(self, _sql, _args=None):
        with self.connection:
            with self.connection.cursor() as cursor:
                try:
                    if _args is None:
                        cursor.execute(_sql)
                    else:
                        cursor.execute(_sql, _args)
                except Exception:
                    return self.__query_response(execution=False)

                _res = cursor.fetchall()
                # handle query that do not have response but
                # have been executed correctly
                if _res is None:
                    return self.__query_response(execution=True)
                # if there is a response
                else:
                    # more than one row found for result
                    if len(_res) > 1:
                        return self.__query_response(execution=True, value=_res)
                    else:
                        # more than one col asked for return
                        if len(_res[0]) > 1:
                            return self.__query_response(execution=True, value=_res[0])
                        else:
                            # only one row, one col.
                            return self.__query_response(execution=True, value=_res[0][0])



class CadastreDAO(DAO):

    def __init__(self):
        DAO.__init__(self)
        self.initialize()

    def initialize(self):
        create_particelle = "CREATE TABLE IF NOT EXISTS particelle (id SERIAL PRIMARY KEY, \
                                                            comune VARCHAR(64), foglio VARCHAR(64), particella VARCHAR(64),\
                                                            bbox GEOMETRY,\
                                                            geom GEOMETRY,\
                                                            updated TIMESTAMP DEFAULT NOW()\
                                                            );"
        self._query(create_particelle)

    def reset(self):
        drop_particelle = "DROP TABLE IF EXISTS particelle;"
        self._query(drop_particelle)

    def exists(self, comune, foglio, particella):

        _sql = "SELECT 1 from particelle WHERE comune=%(comune)s AND foglio=%(foglio)s AND particella=%(particella)s"
        _arg = {"comune": comune, "foglio": foglio, "particella": particella}
        if self._query(_sql, _arg):
            return True
        return False

    def insert(self, comune, foglio, particella):
        _sql = "INSERT INTO particelle (comune,foglio,particella) VALUES (%(comune)s, %(foglio)s, %(particella)s)"
        _arg = {"comune": comune, "foglio": foglio, "particella": particella}
        if self._query(_sql, _arg):
            return True
        return False

    def update_geom(self, comune, foglio, particella, key, poly, srs):
        """

        Args:
            comune:
            foglio:
            particella:
            key:
            poly: poly.wkt
            srs:

        Returns:

        """
        _sql = None
        if key == "geom":
            _sql = "UPDATE particelle SET geom=ST_GeomFromText(ST_AsText(%(poly)s),%(srs)s) \
            WHERE comune=%(comune)s AND foglio=%(foglio)s AND particella=%(particella)s;"
        if key == "bbox":
            _sql = "UPDATE particelle SET bbox=ST_GeomFromText(ST_AsText(%(poly)s),%(srs)s) \
                             WHERE comune=%(comune)s AND foglio=%(foglio)s AND particella=%(particella)s;"

        _arg = {"comune": comune, "foglio": foglio, "particella": particella, "poly": poly, "srs": srs}
        if _sql is None:
            return False

        if self._query(_sql, _arg) != ERROR_CODE:
            return True
        return False

    def get_particella(self, comune, foglio, particella):

        _sql = "SELECT geom FROM particelle WHERE comune='{comune}' AND foglio='{foglio}' AND particella='{particella}';".format(
            comune=comune,
            foglio=foglio,
            particella=particella)
        res = self._query_and_return(_sql)
        geom = wkb.loads(res, hex=True)
        feature = Feature(geom, properties={'comune': comune, 'foglio': foglio, 'particella': particella})
        return dumps(feature)

class QueryLoggerDAO(DAO):

    def __init__(self):
        DAO.__init__(self)

    def initialize(self):
        _sql = "CREATE TABLE IF NOT EXISTS logger(, status INT);"


class BordersDAO(DAO):

    def __init__(self):
        DAO.__init__(self)
        self.srs = 4258

    def initialize(self):
        import csv
        import fiona
        import pyproj
        from shapely.ops import transform
        from shapely.geometry import shape

        """
        1. cancella e ricrea tabella comuni
        """

        # self.connection = psycopg2.connect(dbname='cadastredb', user='biloba', host='127.0.0.1', password='biloba')
        # self.cursor = self.connection.cursor()

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
        created TIMESTAMP DEFAULT NOW());"

        # self.cursor.execute("DROP TABLE IF EXISTS comuni;")
        self._query("DROP TABLE IF EXISTS comuni;")
        # self.cursor.execute(create_comuni)
        self._query(create_comuni)

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

                self._query('''INSERT INTO comuni (id, comune, cod_catastale_comune, provincia,\
                 sigla_provincia, cod_istat_provincia, regione, cod_istat_regione) \
                 VALUES (%s,%s,%s,%s,%s,%s,%s,%s);''' % (id, comune, cod_catastale_comune, provincia, sigla_provincia, cod_istat_provincia,
                                     regione, cod_istat_regione)
                            )

        # from source tp destination coordinate system
        project = pyproj.Transformer.from_proj(
            pyproj.Proj(init='epsg:32632'),
            pyproj.Proj(init='epsg:4258')
        )

        with fiona.open(BORDERS_SHAPE_COMUNI) as input:

            for feature in input:
                shape_id = feature['properties']['PRO_COM_T']
                geom = shape(feature['geometry'])
                new_geom = transform(project.transform, geom)

                self._query('''UPDATE comuni SET geom=%s WHERE id=%s;''' % (new_geom.wkt, shape_id))

    def generate_points(self, _comune, _distance):
        from shapely.geometry import Point
        from shapely import wkb
        import numpy as np
        from utils.utils import meters2degrees
        import logging

        points = []

        _sql = "SELECT id FROM comuni WHERE comune=%(comune_name)s"
        _arg  = {'comune_name': _comune}
        id_comune = self._query(_sql, _arg)
        print(id_comune)
        print("ciao")


        logging.info("PRECISION OF SAMPLING DISTANCE in meter: " + str(_distance))
        _res = self._query_and_return("SELECT  geom FROM comuni WHERE id='%s';" % str(id_comune))

        if _res is None:
            raise ValueError("comune id non esiste nel db, riprova scemo!")
        #elif len(_res) > 1:
        #    raise ValueError("Codice comune risulta dublicato, wtf?")

        # check if _res[0] is ok with fetchone
        poly = wkb.loads(_res[0], hex=True)
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
                points.append([p_array[0][count][0], p_array[0][count][1]])
            count += 1
        logging.info("PUNTI SCELTI: " + str(len(points)))
        return points

    def get_comune(self, name_comune):
        _sql = "SELECT id FROM comuni WHERE comune=%s" % name_comune
        _res = self._query_and_return(name_comune)
        return _res[0]



# p = DAOParticella()
# p.get_particella('A564', '0038', '117')
