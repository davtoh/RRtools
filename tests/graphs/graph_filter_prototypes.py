"""
run graphs for filter prototypes
"""
from preamble import *
from tests.tesisfunctions import filterFactory,graph_filter
from RRtoolbox.lib.arrayops import bandpass, highpass
script_name = os.path.basename(__file__).split(".")[0]

def graph(pytex =None, name = script_name, alpha=5, beta1=100, beta2=150, fclass = highpass):
    """
    :param pytex: (None) context from pythontex
    :param name: (script_name) base save name
    :param alpha:
    :param beta1:
    :param beta2:
    :param fclass: filter class to make custom filter
    :return: locals()
    """
    gd = graph_data(pytex)

    filters = [
        filterFactory(alpha, beta1), # highpass
        filterFactory(-1*alpha, beta1), # lowpass
        filterFactory(alpha, beta1, beta2), #bandpass
        filterFactory(-1*alpha, beta1, beta2), # bandstop
        filterFactory(-1*alpha, beta2, beta1), # invertedbandstop
        filterFactory(alpha, beta2, beta1)] # invertedbandpass

    fig = figure(figsize=(20,10))#figsize=(mm2inch(163,45)) # 163, 45 mm
    graph_filter(filters,single=False,cols=3,
                 legend=False,annotate=True,
                 show=False, win=fig)#,levels=np.linspace(-40, 40,81))
    gd.output(name)

    fig = figure()
    graph_filter(filters,single=True,cols=3,
                 legend=True,annotate=True,
                 show=False, win=fig,titles="")#,levels=np.linspace(-40, 40,81))
    gd.output(name+"_single")

    class fclass_mod(fclass):
        name = "{}*input".format(fclass.__name__)
        def __call__(self, values):
            return super(fclass_mod, self).__call__(values) * values

    try:
        fcustom = fclass(alpha, beta1)
        fcustom2 = fclass_mod(alpha, beta1)
    except:
        fcustom = fclass(alpha, beta1,beta2)
        fcustom2 = fclass_mod(alpha, beta1, beta2)

    filters = [fcustom,
               fcustom2]
    fig = figure()
    graph_filter(filters,single=True,cols=3,
                 legend=True,annotate=True,
                 show=False,win=fig,titles="")
    gd.output(name+"_custom")
    return locals()

if __name__ == "__main__":
    from RRtoolbox.lib.arrayops import bandpass, highpass
    graph(pytex.context, alpha=5, beta1=100, beta2=150, fclass = highpass)