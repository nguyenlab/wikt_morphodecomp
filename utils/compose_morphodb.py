import json
import sys

morphodb = None
plurals = None
morphodb_plurals = None

with open("enmorphodb.json") as morphodb_file:
    morphodb = json.load(morphodb_file)

with open("enplurals.json") as plurals_file:
    plurals = json.load(plurals_file)


morpho_intersection = [word for word in plurals if (plurals[word]["morphemes"]["seq"][0] in morphodb)]

morphodb_plurals = dict([(k, v) for (k, v) in morphodb.items()])
for word in morpho_intersection:
    morphodb_plurals[word] = plurals[word]

with open(sys.argv[1], "w") as morphodb_plurals_file:
    json.dump(morphodb_plurals, morphodb_plurals_file)



