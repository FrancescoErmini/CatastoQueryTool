import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
#from descartes import PolygonPatch

#colors = [['b','g','r'],['c','m','y'],['c', 'k','w']]

def round2(num, ref=None):
    num = round(num, 6)

    return float(str(num)[-4:])

#https://stackoverflow.com/questions/26935701/ploting-filled-polygons-in-python
def plot_bboxes(_bboxes):
    fig, ax = plt.subplots()
    #ax.set_xlim(10.0, 12.0)
    #ax.set_ylim(40.0, 45.0)
    patches = []
    patches2 = []
    p_src = None
    cvect = []
    centri = []
    x_min = 10000
    y_min = 10000
    x_max = 0
    y_max = 0

    for (i, j), _bbox in np.ndenumerate(_bboxes):
        x_min = min(x_min, round2(_bbox.lon1))
        y_min = min(y_min, round2(_bbox.lat1))
        x_max = max(x_max, round2(_bbox.lon2))
        y_max = max(y_max, round2(_bbox.lat2))
        p_src = [
            (round2(_bbox.lat1, _bbox.lat1), round2(_bbox.lon1, _bbox.lon1)),
            (round2(_bbox.lat2, _bbox.lat1), round2(_bbox.lon1, _bbox.lon1)),
            (round2(_bbox.lat2, _bbox.lat1), round2(_bbox.lon2, _bbox.lon1)),
            (round2(_bbox.lat1, _bbox.lat1), round2(_bbox.lon2, _bbox.lon1))
            ]
        centri.append(([(round2(_bbox.lat2)+round2(_bbox.lat1))/2.0, (round2(_bbox.lon2)+round2(_bbox.lon1))/2.0],i,j))
        print(p_src)
        pol = Polygon(p_src, True)
        # polygon = Polygon(p_src)
        patches.append(pol)
        cvect.append((i * j) % _bboxes.size)

    #cmap = plt.cm.get_cmap('jet')
    #norm = plt.Normalize(min(cvect), max(cvect))

    p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.2)
    colors = 100 * np.random.rand(len(patches))
    p.set_array(np.array(colors))
    ax.add_collection(p)
    ax.set_xlim(y_min, y_max)
    ax.set_ylim(x_min, x_max)
    print(x_min, x_max)
    print(y_min, y_max)

    for c in centri:
        plt.text(c[0][0], c[0][1], "%d %d" % (c[1], c[2]))


    # clist = list(set(cvect))
    # handles = []
    # for col in clist:
    #     print(p.norm(col))
    #     handles.append(Polygon([(0, 0), (10, 0), (0, -10)], color=p.cmap(p.norm(col)),
    #                            label='bbox %i' % (col)))
    #
    # plt.legend(handles=handles)


    plt.show()