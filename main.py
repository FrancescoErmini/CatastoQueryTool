"""
Query manager esegue procedura di creazione e aggiornamento caasto.

Procedura di creazione, prende in input un id comune e genera i punti ogni x metri e esegue query.

Procedra di aggiornamento prende in input un id comune, legge dalla tabella particella i bbox, crea il punto centroide esegue query.

"""

from dao.dao import CadastreDAO, BordersDAO

import argparse
from config import  DISTANCE_SAMPLING, MAX_POINTS, IMG_PIXEL_WIDTH, PRINT_UPDATES_EVERY_N_QUERY


class QueryManager:
    def __init__(self, mode, scale, places):
        self.mode = mode
        self.scale = scale
        self.places = places
        self.points = None
        self.bordersDAO = BordersDAO()
        self.comuni = None
        self.init()

    def init(self):
        #if self.scale == 'comune':
        self.comuni = self.places

        self._generate_points()

    def _generate_points(self):
        for comune in self.comuni:
            self.points = self.bordersDAO.generate_points(comune, _distance=DISTANCE_SAMPLING)


    def create(self):
        pass


    def update(self):
        pass


if __name__ == "__main__":
    import sys
    queryObj = QueryManager(mode="create", scale="comune", places=["Bagno a Ripoli", "Firenze"])
    sys.exit(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=['create', 'update'])
    parser.add_argument("places", nargs='*')
    parser.add_argument('--scale', action='store', default='comune', choices=['comune','regione','nazione'])
    args = parser.parse_args()
    print('mode: = %s' % args.mode)
    print('places: = %s' % args.places)
    print('scale = %s' % args.scale)
    if args.mode not in ['create','update'] or args.places is None:
        print("error.")
        sys.exit(1)
    queryObj = QueryManager(mode=args.mode, scale=args.scale, places=args.places)



