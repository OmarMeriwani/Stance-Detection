import re

sources = 'Omar Meriwani H124 235'
numbers = re.findall(r'\d+', sources)
for n in numbers:
    print(str(sources[sources.find(n) - 1]).isalpha())
isNumberExist = False if len(numbers) == 0 else True
