import numpy as np
import random
inputnum = 18748
outputnum = 7455
a = list(range(inputnum))

b = random.sample(a, outputnum)

cell_lines = 'HUVEC'
filename1 = cell_lines + '/promoters.fasta'
filename2 = cell_lines + '/enhancers.fasta'
outputfile = cell_lines + '/data.fasta'
f1 = open(filename1, 'r')
f2 = open(filename2, 'r')
out = open(outputfile, 'w')
for i in f1.readlines():
    out.writelines(i)

count = 0
for i in f2.readlines():
    if int(count/2) in b:
        out.writelines(i)
    count +=1
f1.close()
f2.close()
out.close()