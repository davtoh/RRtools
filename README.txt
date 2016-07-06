RRToolbox (Retinal Restauration Toolbox)
Load convensions http://chimera.labs.oreilly.com/books/1230000000393/ch10.html#namespacepackage

Recommendations:
always import as relative imports and use hole modules:
- from . import mymodule #Ok
- import mymodule # sometimes import cannot find the path if the script is run from another source
- from Mypackage import mymodule #code is not portable, it is hardcoded
- from mymodule import myfunciton #this does not work with imp.reload(mymodule)
- import mymodule
    mymodule.myfunction #Ok #this works with imp.reload(mymodule)

always add relative paths, not absolute paths
import os
# add to main path of file lib folder to look for
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))