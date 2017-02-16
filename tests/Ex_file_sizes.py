from __future__ import print_function
# http://stackoverflow.com/a/5985/5288758
from future import standard_library
standard_library.install_aliases()
import urllib.request, urllib.parse, urllib.error, os, sys
link = "_tmp0000000.png"#"http://python.org"
#link = "https://raw.githubusercontent.com/davtoh/RRtoolbox/master/tests/_tmp0000000.png"
asfile = "out.txt"

print("opening url:", link)
site = urllib.request.urlopen(link)
meta = site.info()
print("Content-Length:", meta.getheaders("Content-Length")[0])

f = open(asfile, "wb")
f.write(site.read())
site.close()
f.close()

f = open(asfile, "rb")
readed = f.read()
print("File on disk after download:",len(readed))
f.close()

print("os.stat().st_size returns:", os.stat(asfile).st_size)