# create a program capable of displaying questions to the user like kbc
# use list data types to store the questions and their correct answers
#  display the final amount the person is taking home after playing the game


levels=5

amount_money=0

questions={
    "q1":1, "q2":2, "q3":3, "q4":4, #level1
    "q5":2,"q6":4,"q7":3,"q8":1,    #level2
    "q9":1, "q10":4,"q11":2,        #level3
    "q12":3,"q13":1,                #level4
    "q14":4                         #level5
}

options={
    "q1":{1:"a1",2:"b1",3:"c1",4:"d1"},
    "q2":{1:"a2",2:"b2",3:"c2",4:"d2"},
    "q3":{1:"a3",2:"b3",3:"c3",4:"d3"},
    "q4":{1:"a4",2:"b4",3:"c4",4:"d4"},
    "q5":{1:"a5",2:"b5",3:"c5",4:"d5"},
    "q6":{1:"a6",2:"b6",3:"c6",4:"d6"},
    "q7":{1:"a7",2:"b7",3:"c7",4:"d7"},
    "q8":{1:"a8",2:"b8",3:"c8",4:"d8"},
    "q9":{1:"a9",2:"b9",3:"c9",4:"d9"},
    "q10":{1:"a10",2:"b10",3:"c10",4:"d10"},
    "q11":{1:"a11",2:"b11",3:"c11",4:"d11"},
    "q12":{1:"a12",2:"b12",3:"c12",4:"d12"},
    "q13":{1:"a13",2:"b13",3:"c13",4:"d13"},
    "q14":{1:"a14",2:"b14",3:"c14",4:"d14"},

}
def display_options(key):
    print(options.get(key))




def user_answer():
    option = None
    for attempt in range(3):
        try:
            enter = int(input("what's your answer ?"))
            if enter in range(1,5):
                option=enter
                break

            else:
                print("invalid choice !")
        except ValueError:
            print("That's not a valid number! Please enter digits only.")
        if attempt==2:
            print("all attempts exhausted! GET LOST")
            break
    return option



def check_ans(ans,key):
    if ans==questions[key]:
        return True
    return False

def question_evaluation(level,start,stop):
    global amount_money

    level_start_money=amount_money
    game_valid=True

    for i,key in enumerate(list(questions.keys())[start:stop]):
        print(key)
        display_options(key)
        ans=user_answer()
        if ans is None:
            game_valid=False
            break

        answer= check_ans(ans,key) #true/false

        if not answer:
            amount_money=level_start_money
            print(f"Wrong answer. earned Rs {amount_money} go back to level {level-1}")
            game_valid=False
            break
        else :
            amount_money= amount_money+100
            print(f"correct answer. earned Rs {amount_money}")

    if game_valid:
        print(f"level {level} finished. earned Rs {amount_money}")
    else:
        print(f"GAME OVER. earned Rs {amount_money}")


    return game_valid


def display_question(level):

    match level:
        case 1:
            success=question_evaluation(level,0,4)
        case 2:
            success=question_evaluation(level,4,9)
        case 3:
            success=question_evaluation(level,9,12)
        case 4:
            success=question_evaluation(level,12,14)
        case 5:
            success=question_evaluation(level,14,55)
        case _:
            print("end of questions")
            print(f"you earned Rs {amount_money}")

    return success




for i in range(1,levels+1):
    print(f"level {i}")
    if not display_question(i):
        break
    if i==levels:
        print("CONGRATULATIONS!!! you completed all the levels")
