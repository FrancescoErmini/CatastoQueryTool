import re
from owslib.wms import WebMapService
from utils import bbox


ITALIA_WMS_URL = 'https://wms.cartografia.agenziaentrate.gov.it/inspire/wms/ows01.php'
CATASTO_ITALIA_SRS = 'EPSG:4258'
CATASTO_ITALIA_LAYER_PARTICELLE = 'CP.CadastralParcel'
MAX_CADASTRE_SCALE_THRESHOLD = 200.0 # metri oltre i quali il catasto mostra una immagine bianca (troppo zoom out)
WMS_QUERY_GEOM_LABEL = 'application/vnd.ogc.gml'
WMS_QUERY_TEXT_LABEL = "text/html"


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

    def get_info(self, _bbox, size=(2, 2), info_format=WMS_QUERY_TEXT_LABEL):

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

    def get_geom(self, _bbox, size=(2, 2), info_format=WMS_QUERY_GEOM_LABEL):
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


def parse_html_response(html_string):
    if html_string is None:
        return None
    #if html_string == 'b\'Content-Type: text/html\\r\\n\\r\\n\''
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