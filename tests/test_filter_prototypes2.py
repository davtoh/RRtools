__author__ = 'Davtoh'

from tesisfunctions import histogram,graphmath,filterFactory,normsigmoid,graph_filter
import numpy as np

def enhancer(alfa1,alfa2,beta1,beta2=None):
    def filter(levels):
        #return np.log(levels)
        return 1/(np.exp((beta1-levels)/alfa1)+1)+1/(np.exp((beta2-levels)/alfa2)+1)
        #return (normsigmoid(levels,alfa,0+beta1)+normsigmoid(levels,alfa,255-beta2))/1.7

    return filter

#alfa1,alfa2,beta1,beta2 = 10,20,-100+20,255-10
#title="filter responce: alfa1,alfa2,beta1,beta2"+str((alfa1,alfa2,beta1,beta2))
#graph_filter([enhancer(alfa1,alfa2,beta1,beta2)],title=title)
until = 400
# lambda x: normsigmoid(x,100,200)
graph_filter([filterFactory(30, 30)], np.linspace(0, until, until))