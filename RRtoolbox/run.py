"""
(Development state) This module is intended to provide managers to run code
which can provide statistics, debugging and facilities to run tools from the
RRtoolbox package.
"""
from __future__ import print_function
from __future__ import absolute_import


if __name__=="__main__":

    #from tools.restoration import asif_demo

    #asif_demo()

    # This script has been made to test and run the core.py and shell.py modules.

    def Test():
        return "that is a test"

    #from pycallgraph import PyCallGraph
    from .lib.inspector import GraphTraceOutput,GraphTrace

    graphviz1 = GraphTraceOutput()
    graphviz1.output_file = 'run.png'
    #@memoize("tracer")
    def do():
        with GraphTrace(output=[graphviz1]) as tracer:
            """
            from core import rrbox
            a = rrbox()
            tool = 'restoration'
            a.tools[tool].asif_demo()
            a.tools[tool].ASIFT.clear()
            a.tools[tool].MATCH.clear()
            """
            x = Test()
            x = x.replace("is","was")
            print(x)
        return tracer

    tracer = do()
    tracer.saveSource("runSource")
    print(tracer.source)