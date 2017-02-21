"""
this script profiles the imrestore code using graphviz
"""
from __future__ import print_function
from __future__ import absolute_import


# This script has been made to test imrestore.py.

from .RRtoolbox.lib.inspector import GraphTraceOutput,GraphTrace

graphviz1 = GraphTraceOutput()
graphviz1.output_file = 'run_imrestore.png'
with GraphTrace(output=[graphviz1]) as tracer:
    # it seems that if pickle is used then this profiling results in an error
    from .imrestore import shell
    shell()
tracer.saveSource("runSource_imrestore")
print(tracer.source)