# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

# from OneOnOne import PretrainedModel
from OneOnOne import Sampling
# from OneOnOne import Classification
from os import path


s=Sampling(file_input_params=True, filepath="testfile.txt")
print(s.model_params)
s.initial_training()
s.get_iterations()



# def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    # print(path.dirname(__file__))
    # c=Sampling("random")
    # c.initial_training()
    # c=ContextDecider()
    # c.decide_context()
    # c.model.summary()

    # print(f'Hi, hi')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
