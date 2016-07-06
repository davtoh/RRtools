from RRtoolbox.lib.root import glob,lookinglob
from preamble import *
import warnings
script = os.path.basename(__file__)
script_name = script.split(".")[0]

def graph(context = None, save=True, gpath=root_path, include = None, exclude=None):

    if not gpath.endswith("/"):
        gpath += "/"

    # get all graphs
    if gpath:
        sys.path.append(gpath)

    graphs = glob(gpath + "graph*.py")
    # exclude to graphs
    if exclude is not None and exclude is "*":
        graphs = []
    elif exclude is not None:
        for i in exclude:
            cleared = False
            for j in lookinglob(i, path=gpath, raiseErr=True, ext="py",returnAll=True):
                if j in graphs:
                    graphs.remove(j)
                    cleared = True
            if not cleared:
                raise Exception("{} not in list".format(i))

    # include to graphs
    if include is not None:
        for j in [lookinglob(i, path=gpath, raiseErr=True, ext="py", returnAll=True) for i in include]:
            for i in j:
                if i.startswith(gpath) and i not in graphs:
                    graphs.append(i)

    # beging to process graphs
    for graph in graphs:
        try:
            print "Trying {}".format(graph)
            obj =__import__(os.path.basename(graph).split(".")[0])

            if hasattr(obj,"graph_data"):
                # do it for each one because it does not work outside in import.
                # it does work but importing as: from tests.graphs.preamble import *
                # not as from preamble import *
                if save: # to save
                    obj.graph_data.saves = True
                    obj.graph_data.shows = False
                else:
                    obj.graph_data.saves = False
                    obj.graph_data.shows = True

                if save and obj.graph_data.shows == True: # checks if bug
                    raise Exception("This should not be if save is {}".format(save))
            else:
                warnings.warn("{} does not have graph_data".format(graph))

            if hasattr(obj,"graph"):
                obj.graph(context)
            else:
                warnings.warn("{} does not have a graph function".format(graph))
        except Exception as e:
            raise Exception("Exception in {}:\n{}".format(graph, e))

if __name__ == "__main__":
    context = {"path":"/mnt/4E443F99443F82AF/Dropbox/thesis/images"}#,"ext":"svg"}
    graph(context, save=False) # show all graphs
    # example showing exclusion
    #graph(context, save=False,exclude = ["graph_polygonArea","graph_overlay"])
    # example showing inclusion. You can add specific graph paths
    #graph(context, save=False,include = ["graph_polygonArea","graph_overlay"], exclude="*")