from preamble import *
from HomogeniousTransform import *

def eq():
    # convert from A to B
    # where "o" stands for original and "s" for scaled
    Is,As,Bs,Io,Ao,Bo = symbols("I_s,A_s,B_s,I_o,A_o,B_o") # scaled and original symbols
    Aow,Aoh = symbols("Aw_o,Ah_o") # original A shape
    Asw,Ash = symbols("Aw_s,Ah_s") # scaled A shape
    Bow,Boh = symbols("Bw_o,Bh_o") # original B shape
    Bsw,Bsh = symbols("Bw_s,Bh_s") # scaled B shape
    oAToB,oATsA,sBToB = symbols("AB_oo,A_os,B_so")
    # oAToB: original A to original B
    # oATsA: original A to scaled A
    # sBToB: scaled B to original B
    sx, sy, ox, oy = symbols("Ax_os,Ay_os,Bx_so,By_so") # scaled x,y and original x,y
    # scaled transformation Matrix
    sTM_str = "M_s"
    sTM_sym = symbols(sTM_str) # symbol used between normal multiplications
    sTM = MatrixSymbol(sTM_str, 3, 3) # symbol used between matrices
    sTM_mat = Matrix(3, 3, lambda i,j:var('{}{}{}'.format(sTM_str,i+1,j+1)))#Matrix(sTM) # matrix
    # original to scaled A
    oATsA_mat = Matrix([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])
    #oATsA_mat = Matrix([[Asw/Aow,0,0],[0,Ash/Aoh,0],[0,0,1]])
    # scaled to original B
    sBToB_mat = Matrix([[ox, 0, 0], [0, oy, 0], [0, 0, 1]])
    #sBToB_mat = Matrix([[Bow/Bsw,0,0],[0,Boh/Bsh,0],[0,0,1]])
    # original transformation Matrix
    oTM_str = "M_o"
    oTM_sym = symbols(oTM_str) # symbol used between normal multiplications
    oTM = MatrixSymbol(oTM_str,3,3) # symbol used between matrices
    oTM_mat = Matrix(3, 3, lambda i,j:var('{}{}{}'.format(oTM_str,i+1,j+1)))#Matrix(oTM) # matrix
    # latex printing options
    opts = dict(mat_delim="[")#, mul_symbol=r"dot")
    # expressions
    # variables with order
    expr1 = "{a} {m} {b} {m} {c}".format(a=latex(sBToB),b=latex(sTM_sym),c=latex(oATsA),m="\cdot")
    # matrices multiplication
    expr1_mat = "{} {} {}".format(latex(sBToB_mat, **opts),latex(sTM_mat, **opts),latex(oATsA_mat, **opts))
    # solved matrix
    _expr2 = sBToB_mat * sTM_mat * oATsA_mat
    expr2 = latex(_expr2, mul_symbol=r"dot", **opts)
    substitute = {sx:Asw/Aow,sy:Ash/Aoh,ox:Bow/Bsw,oy:Boh/Bsh}
    substitutions = convert_substitutions(substitute,joining=r",\ ")
    # solved matrix with replacements
    expr3 = replaceIndexes(latex(_expr2, **opts),substitute,
                           usedelim=True, delim=lambda x:r" \left( {} \right) ".format(latex(x)),
                           strfy=latex).replace(r"\frac{",r"\dfrac{")
    #expr3 = latex(_expr2.subs(substitute), **opts) # ugly
    sTM_mat = latex(sTM_mat, **opts)

    return locals()

if __name__ == "__main__":
    globals().update(eq())