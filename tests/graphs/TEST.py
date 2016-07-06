from tests.graphs.preamble import *
graph_data.saves = True
graph_data.shows = False

if False and __name__ == "__main__":

    from tests.graphs.graph_smooth_hist_max_min import graph
    #windows = 'flat', 'hanning', 'hamming', 'bartlett', 'blackman','savgol'
    wl1,wl2 = 11,51
    savgol = 'savgol'
    myfilter = 'hanning'
    s1 = graph(windows =  [savgol], window_len=wl1)['gd'].savelog[0][1]
    s2 = graph(windows =  [savgol], window_len=wl2)['gd'].savelog[0][1]

    print(r"""
    \begin{figure}[h]
        \centering
        \captionbox{savgol filter}{
        \begin{subfigure}[b]{0.4\textwidth}
            \includegraphics[width=\textwidth]{%s}
            \caption{(a) $window_{len} = %s$}
        \end{subfigure}
        ~
        \begin{subfigure}[b]{0.4\textwidth}
            \includegraphics[width=\textwidth]{%s}
            \caption{(b) $window_{len} = %s$}
        \end{subfigure}
        }
    \end{figure}
    """ % (s1,wl1,s2,wl2))

if True:
    from tests.graphs.graph_bilateralFilter import graph
    data = graph(shapes = ((50,50),(100,100)),
            noise = 's&p', useTitle = True, split = True)
    gd = data["gd"] # get graphic data
    fns = [i[1] for i in gd.savelog] # file names
    ts = [gd.captions[fn] for fn in fns] # captions

    print(r"""
    \begin{figure}[h]
        \centering
        \captionbox{bilateral filter exposed to diferent shapes\label{fig:graph_bilateralFilter}}{""")

    pairs = []
    for i in range(0,len(fns),2):
        pairs.append(
        r"""
        \begin{subfigure}[t]{0.4\textwidth}
            \includegraphics[width=\textwidth]{%s}
            \caption{%s}
        \end{subfigure}
        ~
        \begin{subfigure}[t]{0.4\textwidth}
            \includegraphics[width=\textwidth]{%s}
            \caption{%s}
        \end{subfigure}
        """ % (fns[i],ts[i],fns[i+1],ts[i+1]))

    print(r" \vfil ".join(pairs))

    print(r"""
        }
    \end{figure}
    """)
