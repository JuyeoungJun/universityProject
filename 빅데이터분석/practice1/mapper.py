import sys
for line in sys.stdin:
    line = line.strip() #strip the carrage return (by default)
    keys = line.split() #split line at blanks (by default)
    for key in keys:
        value = 1
        print('{0}\t{1}'.format(key, value))
        #note that the Hadoop default is 'tab' separates key from the value
