from flask import Flask, jsonify
from dao.cadastre_dao import DAOParticella

_cadastre_dao = DAOParticella()
app = Flask(__name__)


@app.route('/comune/<string:comune>/foglio/<string:foglio>/particella/<string:particella>', methods=['GET'])
def get_particle_geom(comune, foglio, particella):
    ret = _cadastre_dao.get_particella(comune=comune, foglio=foglio, particella=particella)
    return jsonify(ret)



if __name__ == "__main__":
    app.run()
