import psycopg2
from shapely import geometry, wkt, wkb
from shapely_geojson import Feature, FeatureCollection, dump, dumps

from psycopg2 import connect, extensions, sql

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
class DAO:

    def __init__(self):
        self.connection = psycopg2.connect(dbname='cadastredb', user='biloba', host='127.0.0.1', password='biloba')
        autocommit = extensions.ISOLATION_LEVEL_AUTOCOMMIT
        print("ISOLATION_LEVEL_AUTOCOMMIT:", extensions.ISOLATION_LEVEL_AUTOCOMMIT)
        self.connection.set_isolation_level(autocommit)

    def __del__(self):
        self.connection.close()

    def _query(self, _sql_obj):
        with self.connection:
            with self.connection.cursor() as cursor:
                cursor.execute(_sql_obj)

    def _query_and_return(self, _sql_obj):
        with self.connection:
            with self.connection.cursor() as cursor:
                cursor.execute(_sql_obj)
                return cursor.fetchone()[0]

        print("eror or not")



class DAOParticella(DAO):

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

        _sql = sql.SQL(
            "SELECT 1 from particelle WHERE comune={comune} AND foglio={foglio} AND particella={particella}"
        ).format(comune=sql.Identifier(comune),
                 foglio=sql.Identifier(foglio),
                 particella=sql.Identifier(particella))
        if self._query(_sql):
            return True
        return False

    def insert(self, comune, foglio, particella):
        _sql = sql.SQL(
            'INSERT INTO particelle (comune,foglio,particella) VALUES ({comune}, {foglio}, {particella})'
        ).format(comune=sql.Identifier(comune),
                 foglio=sql.Identifier(foglio),
                 particella=sql.Identifier(particella))
        if self._query(_sql):
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
        _sql = sql.SQL(
            'UPDATE particelle SET {key}=ST_GeomFromText(ST_AsText({poly}),{srs})\
            WHERE comune={comune} AND foglio={foglio} AND particella={particella};'
        ).format(comune=sql.Identifier(comune),
                 foglio=sql.Identifier(foglio),
                 particella=sql.Identifier(particella),
                 key=sql.Identifier(key),
                 poly=sql.Identifier(poly),
                 srs=sql.Identifier(srs)
                 )
        if self._query(_sql):
            return True
        return False

    def get_particella(self, comune, foglio, particella):

        _sql = "SELECT geom FROM particelle WHERE comune='{comune}' AND foglio='{foglio}' AND particella='{particella}';".format(comune=comune,
                foglio=foglio,
                particella=particella)
        res = self._query_and_return(_sql)
        geom = wkb.loads(res, hex=True)
        feature = Feature(geom, properties={'comune': comune, 'foglio': foglio, 'particella': particella})
        return dumps(feature)







p = DAOParticella()
p.get_particella('A564', '0038', '117')