import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as mse
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def func(y, params):
    assert params.ndim == 1 and params.shape[0] == 4
    x = (params[0] * y ** params[1]) + (params[2] * y ** params[3])
    return x

def pso_algorithm(y, x,
                  limit_min : np.array,
                  limit_max : np.array,
                  inertia = 0.8,
                  c1: float = 2,
                  c2: float = 2,
                  swarm_size = 200,
                  epochs = 1000):
    """
    Particle swarm optimization algorithm for fitting a function.
    """
    # Initialize the swarm and its velocity
    swarms = np.random.rand(swarm_size, 4) * (limit_max - limit_min) + limit_min

    v = (np.random.rand(swarm_size, 4)  - 0.5) * (limit_max - limit_min) * 0.01

    p = np.zeros_like(swarms)   # record best solution for each swarm member
    pv = np.ones(swarm_size) * 1e10  #  minimum error  of the swarm members

    best_p = np.zeros(4)       # record the best solution of the swarm
    best_err = 1e10            # record the best error of the swarm

    err_arr = np.ones(swarm_size) * 1e10  # record the error of the swarm members

    # Calculate the initial error of the swarm members
    for epoch in range(epochs):
        for i in range(swarm_size) :
            # firstly, calculate prediction of x for each swarm member
            member = swarms[i,:]
            x_pred = func(y, member)  # predict the value of x
            x_pred = np.clip(x_pred, 1e-5 * np.ones_like(x_pred), np.inf)
            err = mse(np.log10(x), np.log10(x_pred))

            err_arr[i] = err   # record the error of the swarm member

            # update the best solution point
            if err < pv[i]:
                pv[i] = err
                p[i,:] = member  # update the best solution point

        if np.min(pv) < best_err:
            best_err = np.min(pv)  # update the best error
            best_p = p[np.argmin(pv),:]  # update the best solution

        if epoch < epochs - 1:
            # update the swarm velocity and position
            for j in range(swarm_size) :
                # Start the optimization process
                v[j, :] = (inertia * v[j,:]
                           + c1 * np.random.rand() * (p[j, :] - swarms[j, :])
                           + c2 * np.random.rand() * (best_p - swarms[j, :]))
                swarms[j,:] = v[j,:] + swarms[j,:]  # update the swarm position

            # clip the swarm position to the limit
            swarms = np.clip(swarms, limit_min, limit_max)  # limit the swarm position

        if epoch % 100 == 0:
            print("-" * 50)
            print(f"Epoch {epoch}, Average Error: {np.mean(err_arr):.6f}, Best Error in Group: {np.min(err_arr):.6f}")
            print(f"Best Error so far: {best_err:.6f}, Best Parameters: {best_p}")

    return best_p, best_err

def main():
    # Reading the data from the csv file
    data = pd.read_csv('./data.csv', header=None)
    xp = np.array(data[0])
    yp = np.array(data[1])

    lim_min = np.array([0, -2, 0, -5])
    lim_max = np.array([0.1, 0, 1, 0])
    best_p, best_err = pso_algorithm(yp,xp,
                                     lim_min,
                                     lim_max,
                                     inertia=0.8,
                                     c1=2,
                                     c2=2,
                                     swarm_size=500,
                                     epochs=1000)
    print("=" * 50)  # print the best parameters and error
    print(f"Best Parameters: {best_p}, Best Error: {best_err:.4f}")

    y_test = np.linspace(0.9 * min(yp), 1.1 * max(yp), 2000)
    x_test = func(y_test, best_p)  # predict the value of x

    plt.figure(figsize=(10, 8))
    plt.scatter(yp, xp, label='Data')
    plt.plot(y_test, x_test, label='Fitted Curve')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('y')
    plt.ylabel('x')
    plt.legend()
    plt.title(f"Fitting Curve with PSO Algorithm")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()