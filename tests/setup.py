from distutils.core import setup
import py2exe # only in windows
# see pyinstaller
from glob import glob
import sys

# https://mborgerson.com/creating-an-executable-from-a-python-script
#http://www.py2exe.org/index.cgi/Tutorial
#https://docs.python.org/2/distutils/setupscript.html

visualdll = r"C:\Program Files (x86)\Microsoft Visual Studio 11.0\VC\redist\x86\Microsoft.VC110.CRT"
script = [r'C:\Users\Davtoh\Dropbox\PYTHON\projects\Descriptors\asift.py']
sys.path.append(visualdll)
data_files = [("Microsoft.VC110.CRT", glob(visualdll+'\*.*'))]
setup(name = "asift",
        version = "0.1.0",
        #app=app,
        #options=options,
        #scripts = [example],
        windows=script,#console=script,
        data_files=data_files)