from glob import glob
import os

path = ""
pdg = False
more = ""
if pdg: more = "--export-latex"

def svg2pdf(path="*.svg",pdg=False):
    files = glob(path)
    for file in files:
        parts = file.split(".")
        if len(parts)>1:
            ext = "."+parts[-1]
            body = ".".join(parts[:-1])
        else:
            body = parts[0]
            ext = ""
        os.system("inkscape -D -z --file={body}{ext} --export-pdf={body}.pdf {more}".format(body=body,ext=ext,more=more))

if __name__ == "__main__":
    svg2pdf()