import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


FILE = './data.csv' 


def import_data():  
    data = pd.read_csv(FILE, header=None)
    return data

def find_transpose(A):
    return A.transpose()

def multiply(A, B): 
    return A.dot(B)


def create_XY(data_x, data_y):  
    r = len(data_x)
    A = np.zeros((r, 2), dtype=np.float64)
    B = np.zeros((r, 1), dtype=np.float64)
    for i in range(r):
        A[i][0], A[i][1] = 1, data_x[i]
        B[i][0] = data_y[i]
    return A, B



def create_line(V):  
    return f'Y = ({V[1][0]})X  + {V[0][0]}'
   



def solve_equation(A, B): 
    return np.linalg.solve(A, B)




def draw_chart(x, y, a, b, equation):
    plt.plot(x, y, 'ro')
    plt.plot(x, a*x+b)
    plt.title(equation)
    plt.xlabel('Blood Cell Concentration')
    plt.ylabel('Probability of Getting Blood Cancer')
    plt.show()


def execute():  # Program starts
    data = import_data()
    observed_x = [data[0][i] for i in range(int(0.95*len(data[0])))]
    observed_y = [data[1][i] for i in range(int(0.95*len(data[1])))]
    X, Y = create_XY(observed_x, observed_y)
    XT = find_transpose(X)
    XTX = multiply(XT, X)
    XTY = multiply(XT, Y)
    result = solve_equation(XTX, XTY)
    line_equation = create_line(result)
    print(line_equation)

    for i in range(int(0.95*len(data[0])), len(data[0])):
        print("\nX = ", data[0][i])
        print("Read Value = ", data[1][i])
        print("Estimated Value = ", data[0][i]*result[1][0]+result[0][0])
        print('\033[31m', "Error = ", data[1][i] -
              (data[0][i]*result[1][0]+result[0][0]), '\033[0m')
        print()

    draw_chart(observed_x, observed_y, result[1], result[0], line_equation)


if __name__ == "__main__":
    execute()
