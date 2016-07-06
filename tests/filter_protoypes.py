__author__ = 'Davtoh'

from tesisfunctions import histogram,graphmath,filterFactory,normsigmoid,graph_filter

def enhancer(alfa,beta1,beta2=None):
    def filter(levels):
        #return np.log(levels)
        return (normsigmoid(levels,alfa,0+beta1)+normsigmoid(levels,alfa,255-beta2))*255/2.0
    return filter

def replaceSigmoid(r,m=150,e=100.0):
    # see Digital Image Processing Using Matlab - Gonzalez Woods & Eddins_2 pag. 69
    return 1.0/(1.0+(float(m)/r)**e)

#graph_filter([enhancer(20,20,10)])
graph_filter([replaceSigmoid])
