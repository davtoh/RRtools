# encoding: utf-8
# module cv2.optflow
# from /home/davtoh/anaconda3/envs/rrtools/lib/python3.5/site-packages/cv2.cpython-35m-x86_64-linux-gnu.so
# by generator 1.144
# no doc
# no imports

# Variables with simple values

DISOpticalFlow_PRESET_FAST = 1
DISOpticalFlow_PRESET_MEDIUM = 2
DISOpticalFlow_PRESET_ULTRAFAST = 0

DISOPTICAL_FLOW_PRESET_FAST = 1
DISOPTICAL_FLOW_PRESET_MEDIUM = 2
DISOPTICAL_FLOW_PRESET_ULTRAFAST = 0

GPC_DESCRIPTOR_DCT = 0
GPC_DESCRIPTOR_WHT = 1

__loader__ = None

__spec__ = None

# functions


# real signature unknown; restored from __doc__
def calcOpticalFlowSF(from_, to, layers, averaging_block_size, max_flow, flow=None):
    """ calcOpticalFlowSF(from, to, layers, averaging_block_size, max_flow[, flow]) -> flow  or  calcOpticalFlowSF(from, to, layers, averaging_block_size, max_flow, sigma_dist, sigma_color, postprocess_window, sigma_dist_fix, sigma_color_fix, occ_thr, upscale_averaging_radius, upscale_sigma_dist, upscale_sigma_color, speed_up_thr[, flow]) -> flow """
    pass


# real signature unknown; restored from __doc__
def calcOpticalFlowSparseToDense(from_, to, flow=None, grid_step=None, k=None, sigma=None, use_post_proc=None, fgs_lambda=None, fgs_sigma=None):
    """ calcOpticalFlowSparseToDense(from, to[, flow[, grid_step[, k[, sigma[, use_post_proc[, fgs_lambda[, fgs_sigma]]]]]]]) -> flow """
    pass


def createOptFlow_DeepFlow():  # real signature unknown; restored from __doc__
    """ createOptFlow_DeepFlow() -> retval """
    pass


def createOptFlow_DIS(preset=None):  # real signature unknown; restored from __doc__
    """ createOptFlow_DIS([, preset]) -> retval """
    pass


def createOptFlow_Farneback():  # real signature unknown; restored from __doc__
    """ createOptFlow_Farneback() -> retval """
    pass


def createOptFlow_PCAFlow():  # real signature unknown; restored from __doc__
    """ createOptFlow_PCAFlow() -> retval """
    pass


def createOptFlow_SimpleFlow():  # real signature unknown; restored from __doc__
    """ createOptFlow_SimpleFlow() -> retval """
    pass


def createOptFlow_SparseToDense():  # real signature unknown; restored from __doc__
    """ createOptFlow_SparseToDense() -> retval """
    pass


def createVariationalFlowRefinement():  # real signature unknown; restored from __doc__
    """ createVariationalFlowRefinement() -> retval """
    pass


def readOpticalFlow(path):  # real signature unknown; restored from __doc__
    """ readOpticalFlow(path) -> retval """
    pass


def writeOpticalFlow(path, flow):  # real signature unknown; restored from __doc__
    """ writeOpticalFlow(path, flow) -> retval """
    pass

# no classes
