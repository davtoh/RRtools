# http://stackoverflow.com/a/22664404/5288758
import collections

def move_element(odict, thekey, newpos):
    odict[thekey] = odict.pop(thekey)
    i = 0
    for key, value in odict.items():
        if key != thekey and i >= newpos:
            odict[key] = odict.pop(key)
        i += 1
    return odict

queue = collections.OrderedDict()

queue["animals"] = ["cat", "dog", "fish"]
queue["food"] = ["cake", "cheese", "bread"]
queue["people"] = ["john", "henry", "mike"]
queue["drinks"] = ["water", "coke", "juice"]
queue["cars"] = ["astra", "focus", "fiesta"]

print queue

queue = move_element(queue, "people", 1)

print queue