from sympy import *
from sympy import Matrix, MatrixSymbol, Eq, latex, symbols,eye
from sympy.functions import sin,cos
# another perspective http://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions/transforms/index.htm

n, m, i, j = symbols('n m i j', integer=True) # indexes and dimensions
x,y,z,c,u,v,w = symbols("x,y,z,c,u,v,w") # axis symbols
alpha_x,alpha_y,alpha_z,alpha_u,alpha_v,alpha_w = symbols("alpha_x,alpha_y,alpha_z,alpha_u,alpha_v,alpha_w") # axis scaling
theta_x, theta_y, theta_z, theta_u, theta_v, theta_w = symbols("theta_x,theta_y,theta_z,theta_u,theta_v,theta_w") # axis rotations
beta,theta,delta,gamma,phi = symbols("beta,theta,delta,gamma,phi") # angles
dims = [4,4] #

def fm_identity(i,j): # function of identity matrix
    if i == j: return 1
    else: return 0

def Identity(dim1 = dims[0],dim2 = dims[1]):
    """

    :param dim1: N - rows - Height
    :param dim2: M - cols - Width
    :return: Matrix
    """
    return Matrix(dim1, dim2, fm_identity)# Matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

def HXtranslate(x=x, dim1 = dims[0],dim2 = dims[1]):
    M = Identity(dim1,dim2)
    M[0,2]=x
    return M #Matrix([[1,0,x,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

def HYtranslate(y=y, dim1 = dims[0],dim2 = dims[1]):
    M = Identity(dim1,dim2)
    M[1,2]=y
    return M #Matrix([[1,0,0,0],[0,1,y,0],[0,0,1,0],[0,0,0,1]])

def HZtranslate(z=z, dim1 = dims[0],dim2 = dims[1]):
    M = Identity(dim1,dim2)
    M[2,2]=z
    return M #Matrix([[1,0,0,0],[0,1,0,0],[0,0,z,0],[0,0,0,1]])

def Htranslate(T=None,**axis):
    if not T: T = eye(4)
    return Matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0]])

def op(self,i,j,val):
    try:
        self[i,j]=val
    except IndexError:
        pass

def HXrotate(theta_x=theta_z, dim1 = dims[0],dim2 = dims[1]):
    """
    :param theta_x: in radians
    :return: H
    Matrix([[1, 0,            0,                0],
            [0, cos(theta_z), -sin(theta_z),    0],
            [0, sin(theta_z), cos(theta_z),     0],
            [0, 0,            0,                1]])
    """
    M = Identity(dim1,dim2)
    op(M,1,1,cos(theta_x))
    op(M,2,2,cos(theta_x))
    op(M,1,2,-sin(theta_x))
    op(M,2,1,sin(theta_x))
    return M

def HYrotate(theta_y=theta_z, dim1 = dims[0],dim2 = dims[1]):
    """
    :param theta_y: in radians
    :return: H
    Matrix([[cos(theta_z),  0, sin(theta_z), 0],
            [0,             1, 0,            0],
            [-sin(theta_z), 0, cos(theta_z), 0],
            [0,             0, 0,            1]])
    """
    M = Identity(dim1,dim2)
    op(M,0,0,cos(theta_y))
    op(M,2,2,cos(theta_y))
    op(M,0,2,sin(theta_y))
    op(M,2,0,-sin(theta_y))
    return M

def HZrotate(theta_z=theta_z, dim1 = dims[0],dim2 = dims[1]):
    """
    :param theta_z: in radians
    :return: H
    Matrix([[cos(theta_z), -sin(theta_z), 0, 0],
            [sin(theta_z), cos(theta_z),  0, 0],
            [0,            0,             1, 0],
            [0,            0,             0, 1]])
    """
    M = Identity(dim1,dim2)
    op(M,0,0,cos(theta_z))
    op(M,1,1,cos(theta_z))
    op(M,0,1,-sin(theta_z))
    op(M,1,0,sin(theta_z))
    return M

def Hrotate(theta_x=theta_x, theta_y=theta_y, theta_z=theta_z, dim1 = dims[0],dim2 = dims[1], xyz = False):
    """
        create any 3D rotation
    :param theta_x: rotation in axis x
    :param theta_y: rotation in axis y
    :param theta_z: rotation in axis z
    :param dim1:
    :param dim2:
    :param xyz:
    :return:
    """
    if xyz:
        M = HXrotate(theta_z)*HYrotate(theta_y)*HZrotate(theta_x) # R_XYZ
    else:
        M = HZrotate(theta_z)*HYrotate(theta_y)*HXrotate(theta_x) # R_ZYX
    return M

def HXscale(alpha_x=alpha_x):
    return Matrix([[alpha_x, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])

def HYscale(alpha_y=alpha_y):
    return Matrix([[1,0,0,0], [0, alpha_y, 0, 0], [0, 0, 1, 0]])

def HZscale(alpha_z=alpha_z):
    return Matrix([[1,0,0,0], [0,1,0,0], [0, 0, alpha_z, 0]])

def Hscale(x=1, y=1, z = 1):
    return Matrix([[x,0,0,0],[0,y,0,0],[0,0,z,0]])

def applyTransformations(trans,xyz=False,apply=None):
    if xyz: trans = trans[::-1]
    M = eye(4)
    if apply:
        for t in trans:
            M *= apply(t)
    else:
        for t in trans:
            M *= t
    return M

def getApplyCenter(W,H):
    go2center = HXtranslate(-W / 2.0) * HYtranslate(-H / 2.0)
    return2origin = HXtranslate(W / 2.0) * HYtranslate(H / 2.0)
    def wrap(M):
        return return2origin * M * go2center
    return wrap

def decompose(transform):
    # http://math.stackexchange.com/a/1463487
    pass

if False and __name__ == "__main__":
    """
    There are absolute and relative transformations.
     *Absolute transformations are with respect to the origin (e.g. x0,y0,z0) of the coordinate system (e.g. cartesian, polar)
     *Relative transformations are with respect to the previous position. It can be interpreted as if last position is the
        new origin, as if the coordinate system is moved with the object to the new position.
    WATCHED PROPERTIES:
        1. if all transformations are translations then all combinations of the same set give the same result.
    """
    from RRtoolbox.lib.image import loadcv, cv2, np
    from RRtoolbox.lib.config import MANAGER
    from RRtoolbox.lib.plotter import fastplt
    W,H = 400,300
    size = W,H # this is how functions receive size
    rows,cols = H,W # This is how shape works H,W == im.shape = rows,cols
    #plt = fastplt(im) # show loaded image
    func = None # getApplyCenter(W, H) # function to apply to each transform
    posX,posY,posZ,C = 0,0,1,1 # initialize position in each axis, posZ = 1 -> image is one dimension in z
    coors2D = np.array([posX,posY,C]) # array 2D coors
    coors3D = np.array([posX,posY,posZ,C]) # array 3D coors
    scoors2D = Matrix([x,y,c]) # 2D symbolic array
    scoors3D = Matrix([x,y,z,c]) # 3D symbolic array
    transformations = [HXtranslate(50),HYtranslate(50),HZrotate(3*3.14/10.0)]#[HZrotate(5*3.14/10.0),HXrotate(3.14/1000.0)]
    sM_rel_ = applyTransformations(transformations,False, apply=func) # apply relative transformations # symbolic transformation matrix
    scoo_rel_2D = sM_rel_[0:2,0:3] * scoors2D # get actual 2D symbolic coordinates from relative transformations
    scoo_rel_3D = sM_rel_ * scoors3D # get actual 3D symbolic coordinates from relative transformations
    M_rel_ = np.array(sM_rel_).astype(np.float) # array transformation matrix
    coo_rel_2D = M_rel_[0:2,0:3].dot(coors2D) # get actual 2D array coordinates from relative transformations
    coo_rel_3D = M_rel_.dot(coors3D) # get actual 3D array coordinates from relative transformations
    sM_abs_ = applyTransformations(transformations, True,apply=func) # apply absolute transformations # symbolic transformation matrix
    scoo_abs_2D = sM_abs_[0:2,0:3] * scoors2D # get actual 2D symbolic coordinates from absolute transformations
    scoo_abs_3D = sM_abs_ * scoors3D # get actual 2D symbolic coordinates from absolute transformations
    M_abs_ = np.array(sM_abs_).astype(np.float) # array transformation matrix
    coo_abs_2D = M_abs_[0:2,0:3].dot(coors2D) # get actual 2D array coordinates from absolute transformations
    coo_abs_3D = M_abs_.dot(coors3D) # get actual 3D array coordinates from absolute transformations
    # cv2.warpAffine(im,M[0:2,0:3],size) == cv2.warpPerspective(im,M[:3,:3],size)
    # TESTING
    im = loadcv(MANAGER.TESTPATH +"_good.jpg",1,size)
    fastplt(cv2.warpAffine(im,M_rel_[0:2,0:3],size),title="Affine relative (xyz=False)")
    fastplt(cv2.warpAffine(im,M_abs_[0:2,0:3],size),title="Affine absolute (xyz=True)")
    fastplt(cv2.warpPerspective(im,M_rel_[:3,:3],size),title="Perspective relative (xyz=False)")
    fastplt(cv2.warpPerspective(im,M_abs_[:3,:3],size),title="Perspective absolute (xyz=True)")
    assert np.allclose(M_rel_,M_abs_)
    #coors1 = np.array([[0,0],[200,50],[50,200]])
    #coors2 = coors1
    #M = cv2.getPerspectiveTransform(coors1,coors2) # get transformation matrix from coordinates
    #im2 = cv2.warpAffine(im,M[0:2,0:3],size)
    #im2 = cv2.warpPerspective(im,M[:3,:3],size)
    #im2 = cv2.transform(im,T) # color transform


if False and __name__ == "__main__":
    x,y,z,c = symbols("x,y,z,c")
    vector_c = Matrix([[x,y,z,c]]) # define a generic cartesian vector
    Hvector_1x4 = MatrixSymbol("V",1,4) # define a generic cartesian vector
    Vvector_1x4 = MatrixSymbol("V",4,1) # define a generic cartesian vector
    matrix1_4x4 = MatrixSymbol("M",4,4) # define a generic 4x4 matrix
    matrix2_4x4 = MatrixSymbol("M",4,4) # define a generic 4x4 matrix

    show(sBToB*sTM*oATsA)
    show(matrix2_4x4*Vvector_1x4)
    walkMatrix(matrix2_4x4)

    print getMatrixSymbol(matrix2_4x4*Vvector_1x4)
    print getMatrixSymbol((Hvector_1x4*matrix2_4x4))