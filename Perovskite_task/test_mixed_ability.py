


"""
Proof of mixed optimization ability for skopt on arbitrary bb fun
"""

dim = [(1,2),(5,6)]
my_input = [[1,6]]
my_output = [6]

def BBfun (x):
    "highest score 210 with [10, 20, 'red']"
    score = x[0]*x[1]
    # if x[2] == "red":
    #     score+=10
    # elif x[2]=="orange":
    #     score+=5
    # elif x[2]=="green":
    #     score+=1
    # else:
    #     score-=5
    return score*-1


from turboworks.gp_opt import gp_minimize

for i in range(20):
    guess = gp_minimize(my_input,my_output,dim)
    print "iteration", i, "input:", my_input
    print "iteration", i, "guess:", guess
    my_input.append(guess)
    my_output.append(BBfun(guess))




