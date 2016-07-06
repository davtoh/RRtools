"""
reads a document headers of the form:
    header ......
and returns formats the new documents as:
    header
"""
import re

findline = re.compile(r"([\.| ]*)(.*)",re.I)

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def process(text):
    lines = text.splitlines()
    all = [j.groups() for j in [findline.match(i[::-1]) for i in lines] if j]
    new_lines = []
    for ind, header in all:
        if header and not is_number(header): # to
            if ind:
                new_lines.append("    "+header[::-1])
            else:
                new_lines.append(header[::-1])
    return new_lines

if __name__ == "__main__":

    with open("headers","r") as t:
        text = t.read()
        processed =process(text)
        #ascii_letters += " \n"
        #processed = [j for j in text if j in ascii_letters]
        with open("headers_new","w") as t2:
            t2.write("\n".join(processed))