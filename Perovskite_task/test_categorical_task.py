
from turboworks.gp_opt import gp_minimize
# string1dim = ["good","bad","terrible","awful","bad1","terrible1","awful1","bad2","terrible2","awful2","bad3","terrible3","awful3"]
# string2dim = ["garbage","trash","junk","answer","garbage1","trash1","junk1","garbage2","trash2","junk2","garbage3","trash3","junk3"]

string1dim = ["good","bad","awful"]
string2dim = ["garbage","answer","trash"]
def BBfun(strings):
    score = 100
    string1 = strings[0]
    string2 = strings[1]
    if string1 == "good":
        if string2 == "answer":
            score = 1
        elif string2 == "junk":
            score = 15
        else:
            score = 50
    elif string1 == "bad":
        if string2 == "answer":
            score = 5
        elif string2 == "junk":
            score = 20
    else:
        score = 100

    print "'",string1, string2,"'", "got score:", score
    return score

my_input = [["awful","answer"]]
my_output = [BBfun(my_input[0])]
dimensions = [string1dim, string2dim]

run_num = 2

for run in range(run_num):
    ans = gp_minimize(my_input, my_output,[string1dim,string2dim])
    my_input.append(list(ans))
    my_output.append(BBfun(ans))

name_index = ['Li', 'Be', 'B', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu',
              'Zn', 'Ga', 'Ge',
              'As', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'Cs', 'Ba',
              'La', 'Hf','Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Ti', 'Pb', 'Bi']

sub = ['Li', 'Be', 'B', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Sc','Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu',
       'Zn', 'Ga', 'Ge','As', 'Rb', 'Sr','Y', 'Zr', 'Nb', 'Mo', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
       'Sb', 'Te', 'Cs', 'Ba', 'La', 'Hf','Ta', 'W', 'Re', 'Os','Ir', 'Pt', 'Au', 'Hg', 'Pb',]

import timeit
start_time = timeit.default_timer()

print "---------------------------"
theinput = [['Li', 'Al', 'O3']]
theoutput = [3.0951368517076494]
thedimensions = [sub,sub, ['O3', 'O2N', 'ON2', 'N3', 'O2F', 'OFN', 'O2S']]
print "INPUT:", theinput
print "OUTPUT:", theoutput
print "DIM:", thedimensions
print"NEXT_X:",gp_minimize(theinput,theoutput,thedimensions)

elapsed = timeit.default_timer() - start_time
print(elapsed)