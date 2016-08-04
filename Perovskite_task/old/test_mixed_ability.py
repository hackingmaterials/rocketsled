


"""
Proof of mixed optimization ability for skopt on arbitrary bb fun
"""

dim = [(1,3),(5,7),["red","orange","green","blue","turquoise"]]
my_input = [[1, 6, "blue"]]
my_output = [1]

def BBfun (x):
    score = x[0]*x[1]
    if x[2] == "red":
        score+=10
    elif x[2]=="orange":
        score+=5
    elif x[2]=="green":
        score+=1
    else:
        score-=5
    return score*-1


from turboworks.gp_opt import gp_minimize

for i in range(130):
    guess = gp_minimize(my_input,my_output,dim)
    my_input.append(guess)
    my_output.append(BBfun(guess))
    # if [5,10,"red"] in my_input:
    #     break
    print i, guess




