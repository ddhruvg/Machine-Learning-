#import libraries 
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.animation import FuncAnimation
import matplotlib.colors as colors


#Note -> B is substitution for Thetha Parameters 

#import data 
x_train  = np.loadtxt("../data/Q1/linearX.csv", delimiter=",")
y_train = np.loadtxt("../data/Q1/linearY.csv", delimiter=",")

#plot train data
def plot_train(x_train,y_train):
    # plt.clf() 
    plt.scatter(x_train, y_train, color="lightgray", s=10, label="All data")

    
    indices = np.random.choice(len(x_train), size=100, replace=False) #some random points taken to avoid congestion 
    x_sample = x_train[indices]
    y_sample = y_train[indices]

    plt.scatter(x_sample, y_sample, color="red", s=30, label="Random sample")


    plt.xlabel("X value")
    plt.ylabel("Y value")
    plt.title("Training Data")
    plt.legend()
    plt.show()

plot_train(x_train,y_train) #to plot train data


#data handling 
X = np.ones((2,1000))
for i in range(1000):
    X[1][i] = x_train[i]
X = X.T
B = np.zeros((1, 2))
Y = y_train.reshape(1,1000)

#gadient descent functions and other utilites '
def hypothesis(X,B):
    return X@B.T
def loss_fun(X,Y,B):
    m = len(Y)
    H = hypothesis(X,B)
    error = (Y.T - H)
    return (error.T@ error )/(2*m)
    
def gradient_descent(alpha,conv_param,B,Y,X): 
    m = len(X)
    delta = float('inf')
    loss =  loss_fun(X,Y,B)
    cnt  = 0
    while abs(delta) >= conv_param:
        H = hypothesis(X,B)
        grad = np.dot((Y.T - H).T, X) / len(X)
        B = B + alpha * grad
        
        delta = loss - loss_fun(X,Y,B) 
        loss = loss_fun(X,Y,B)
        
        cnt += 1
    # print(f"B = {B[0]} " , f"Loss = {loss[0][0]}")  
    print(f"B = {B[0]} " , f"Loss = {loss}" ,f"Delta = {delta}")  
    print(f"total iterations = {cnt}")  
    return B  



#predict 
B = np.zeros((1, 2)) # intial assumption 
alpha = 0.1 #learnign rate
convg_param = 1e-10 #covergence value
B = gradient_descent(alpha,convg_param,B,Y,X)
predict_y = hypothesis(X,B)         



#new line plot with original points in background 
# plt.clf()  
plt.scatter(x_train, y_train, color="lightgray", s=12, label="Train data")



indices = np.random.choice(len(x_train), size=100, replace=False)
x_sample = x_train[indices]
y_sample = y_train[indices]
plt.scatter(x_sample, y_sample, color="red", s=30, label="Random sample")

plt.plot(x_train, predict_y, color="blue", linewidth=2, label="Learned line")


plt.xlabel("X value")
plt.ylabel("Y value")
plt.title("Train Data vs Learned Line")

plt.legend()

plt.show()

#fucntions for 3-D plot and contours 
def gradient_descent_plot(alpha,conv_param,B,Y,X,dist): 
    m = len(X)
    delta = float('inf')
    loss =  loss_fun(X,Y,B)
    cnt  = 0
    J  = [loss[0][0]]
    B0 = [0.]
    B1 = [0.]
    while abs(delta) > conv_param:
        H = hypothesis(X,B)
          
        grad = np.dot((Y.T - H).T, X) / len(X)
        B = B + alpha * grad
        
        delta = loss - loss_fun(X,Y,B) 
        loss = loss_fun(X,Y,B)
        cnt += 1
        
        # if cnt%dist == 0  and len(J) < 50:  
        if cnt%dist == 0:     #to specify gap between consecutive J points
            J.append(float(loss[0][0]))
            B0.append(float(B[0][0]))  
            B1.append(float(B[0][1])) 
            
    return J, B0, B1


def show_3d(X,Y,alpha,convg_param,lr):
    # plt.clf()
    B0_range = np.linspace(-20, 30, 200)
    B1_range = np.linspace(0, 60, 200)
    B0, B1 = np.meshgrid(B0_range, B1_range)


    Z = np.zeros_like(B0, dtype=float)
    for i in range(B0.shape[0]):
        for j in range(B0.shape[1]):
            Bb = np.array([[B0[i, j], B1[i, j]]])
            Z[i, j] = loss_fun(X, Y, Bb).item()

    dist = 1 #to specify gap between consecutive J points
    J_points,B0_points,B1_points = gradient_descent_plot(alpha,convg_param,np.zeros((1, 2)),Y,X,dist)

    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(B0, B1, Z, cmap="viridis", alpha=0.7, edgecolor="none")
    ax.view_init(elev=30, azim=45)
    ax.scatter(B0_points,B1_points, J_points, color='red', s=20, label="My Points")
    
    ax.set_xlabel("Thetha 0")
    ax.set_ylabel("Thetha 1")
    ax.set_zlabel("Loss")
    ax.set_title(f"Gradient Descent Path LR = {lr}")
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

def show_contor(X,Y,alpha,convg_param):
    # plt.clf()
    B0_range = np.linspace(-20, 30, 200)
    B1_range = np.linspace(0, 60, 200)
    B0, B1 = np.meshgrid(B0_range, B1_range)


    J = np.zeros_like(B0, dtype=float)
    for i in range(B0.shape[0]):
        for j in range(B0.shape[1]):
            Bb = np.array([[B0[i, j], B1[i, j]]])
            J[i, j] = loss_fun(X, Y, Bb).item()
    dist =1 #to specify gap between consecutive J points
    J_points,B0_points,B1_points = gradient_descent_plot(alpha,convg_param,np.zeros((1, 2)),Y,X,dist)
    
    plt.figure()
    plt.clf()
    contour = plt.contourf(B0, B1, J, levels=10 ,  cmap='viridis')  
    plt.colorbar(contour, label="Error Value")
    plt.scatter(B0_points, B1_points, c='red', s=15, label="My Points")

    plt.xlabel("θ0")
    plt.ylabel("θ1")
    plt.legend()
    plt.show()

#Calls for different Learning Rates 

alpha = 0.001 #learing rate
convg_param = 1e-10
show_3d(X,Y,alpha,convg_param,alpha)
show_contor(X,Y,alpha,convg_param)

alpha = 0.025
convg_param = 1e-10
show_3d(X,Y,alpha,convg_param,alpha)
show_contor(X,Y,alpha,convg_param)

alpha = 0.1
convg_param = 1e-10
show_3d(X,Y,alpha,convg_param,alpha)
show_contor(X,Y,alpha,convg_param)


#3-D Animations fucntions 

def gradient_descent_plot_ani(alpha,conv_param,B,Y,X,dist): 
    m = len(X)
    delta = float('inf')
    loss =  loss_fun(X,Y,B)
    cnt  = 0
    J  = [loss[0][0]]
    B0 = [0.]
    B1 = [0.]
    while abs(delta) > conv_param:
        H = hypothesis(X,B)
          
        grad = np.dot((Y.T - H).T, X) / len(X)
        B = B + alpha * grad
        
        delta = loss - loss_fun(X,Y,B) 
        loss = loss_fun(X,Y,B)
        cnt += 1
        
        if cnt%dist == 0  and len(J) < 100:  
        # if cnt%dist == 0:     
            J.append(float(loss[0][0]))
            B0.append(float(B[0][0]))  
            B1.append(float(B[0][1])) 
            
    return J, B0, B1
def show_3d_animation(X, Y, alpha, convg_param,lr):
    # plt.clf()
    B0_range = np.linspace(-20, 30, 200)
    B1_range = np.linspace(0, 60, 200)
    B0, B1 = np.meshgrid(B0_range, B1_range)

    Z = np.zeros_like(B0, dtype=float)
    for i in range(B0.shape[0]):
        for j in range(B0.shape[1]):
            Bb = np.array([[B0[i, j], B1[i, j]]])
            Z[i, j] = loss_fun(X, Y, Bb).item()
    dist = 1
    J_points, B0_points, B1_points = gradient_descent_plot_ani(alpha, convg_param, np.zeros((1, 2)), Y, X, dist)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    
    surf = ax.plot_surface(B0, B1, Z, cmap="viridis", alpha=0.7, edgecolor="none")

   
    ax.set_xlabel("Theta 0")
    ax.set_ylabel("Theta 1")
    ax.set_zlabel("Loss")
    ax.set_title(f"Gradient Descent Path, LR = {lr}")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    
    scatter = ax.scatter([], [], [], color='red', s=30, label="Loss Value")

    ax.legend()
    ax.view_init(elev=30, azim=45)
 
    def update(frame):
        if frame < len(B0_points):
           
            scatter._offsets3d = (B0_points[:frame+1], B1_points[:frame+1], J_points[:frame+1])
            
            ax.view_init(elev=30, azim=45 + frame/2)
        
        return scatter
    
    
    ani = FuncAnimation(fig, update, frames=len(B0_points)+20, 
                        interval=200, blit=False, repeat=False) #interval specify time delay 

    ani.save(f'gradient_descent_animation_{lr}.gif', writer='pillow', fps=5, dpi=100) #to save 3-d gif 
    
    plt.show()
    
    return ani


def create_contour_animation(X, Y, alpha, convg_param,lr):
    
    B0_range = np.linspace(-20, 30, 200)
    B1_range = np.linspace(0, 60, 200)
    B0, B1 = np.meshgrid(B0_range, B1_range)

    J = np.zeros_like(B0, dtype=float)
    for i in range(B0.shape[0]):
        for j in range(B0.shape[1]):
            Bb = np.array([[B0[i, j], B1[i, j]]])
            J[i, j] = loss_fun(X, Y, Bb).item()
    
    dist = 1
    J_points, B0_points, B1_points = gradient_descent_plot_ani(alpha, convg_param, np.zeros((1, 2)), Y, X, dist)
    
    # Create figure and contour
    fig, ax = plt.subplots()
    contour = ax.contourf(B0, B1, J, levels=10, cmap='viridis')  
    plt.colorbar(contour, label="Error Value")
    
    
    scatter = ax.scatter([], [], c='red', s=15, label=f"Gradient Descent Path , LR = {lr}")
    ax.set_xlabel("θ0")
    ax.set_ylabel("θ1")
    ax.legend()
    
    
    def update(frame):
        
        scatter.set_offsets(np.c_[B0_points[:frame+1], B1_points[:frame+1]])
        return scatter,
    
    
    ani = FuncAnimation(fig, update, frames=len(B0_points), 
                        interval=200, blit=True, repeat=False)
    
    
    ani.save(f'gradient_descent_animation_contour_{lr}.gif', writer='pillow', fps=5)
    
    plt.show()
    return ani

#Animations call
#Note -> for animation to be resonable, i have only taken 50-consecutive points for lr = 0.1, for lower lr the gap between taken points should be increaed like 100 for LR = 0.001 beacue otherwise it would be too congested and no proper view will be present.  
alpha = 0.1
convg_param = 1e-10
show_3d_animation(X,Y,alpha,convg_param,alpha)
create_contour_animation(X,Y,alpha,convg_param,alpha)

#After the call 3-D gif is saved 

#for the last part 
#make dist = 100 in create_contour_animation for better anmation other wise points will be very congested 
alpha = 0.001
create_contour_animation(X,Y,alpha,convg_param,alpha)

alpha = 0.025
create_contour_animation(X,Y,alpha,convg_param,alpha)

