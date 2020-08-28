import sys
import time
string = {} #list of words

start = time.time()

for input_line in sys.stdin:
    key, value = input_line.split()
    value = int(value)
    if key in string.keys(): #the word is in the list already
        string[key] += value
    else: #the word inserted into the list first time
        string[key] = 1
for s in string:
    print("{0}\t{1}".format(s, string[s]))
    

end = time.time()

print("{:.4f}\n".format(end-start))
