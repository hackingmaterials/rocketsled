from turboworks.gp_opt import gp_minimize


"""
Proof of mixed optimization ability for skopt on arbitrary bb fun
"""

dim = [(1,10), (2.4, 4.8),("red","blue","green","orange")]
dim2 = [(1,10), (2.4, 4.8),(1,5)]
dim3 = [(1,10),(2,20),("red","blue","green","orange")]
my_input = [[9, 2,"green"]]
my_output = [12]

def BBfun (x):
    "highest score is 10, 4.8, red"
    score = 0
    score+= x[0]*x[1]
    if x[2] == "red":
        score+=10
    elif x[2]=="orange":
        score+=5
    elif x[2]=="green":
        score+=1
    else:
        score-=5
    return score*-1

for i in range(1):
    guess = gp_minimize(my_input,my_output,dim)
    my_input.append(guess)

    # convert0 = int(guess[0])
    # convert1 = float(guess[1])
    # guess = [convert0, convert1, guess[2]]
    my_output.append(BBfun(guess))
    # print "Guess:", guess