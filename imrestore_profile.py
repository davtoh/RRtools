"""
this script profiles the imrestore code using graphviz
"""
__author__ = 'Davtoh'

# This script has been made to test imrestore.py.

from RRtoolbox.lib.inspector import graphTraceOutput,graphTrace

graphviz1 = graphTraceOutput()
graphviz1.output_file = 'run_imrestore.png'
with graphTrace(output=[graphviz1]) as tracer:
    # it seems that if pickle is used then this profiling results in an error
    from imrestore import shell
    shell()
tracer.saveSource("runSource_imrestore")
print tracer.source