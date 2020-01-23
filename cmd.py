"""
Query manager esegue procedura di creazione e aggiornamento caasto.

Procedura di creazione, prende in input un id comune e genera i punti ogni x metri e esegue query.

Procedra di aggiornamento prende in input un id comune, legge dalla tabella particella i bbox, crea il punto centroide esegue query.

"""
from dao.cadastre_dao import CATASTO_DAO


class QueryManager:
    def __init__(self):
        self._dao = CATASTO_DAO

    def create(self):
        pass

    def update(self):
        pass