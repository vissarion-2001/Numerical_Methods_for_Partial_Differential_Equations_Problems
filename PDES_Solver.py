# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 14:30:14 2024

@author: velis
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.animation as animation 

def oned_diffusion_ftcs(xo=0, xn=1, dx=0.1, to=0, tn=2000, dt=0.1, a=1e-4):
    """ This function uses forward time, central space to solve
    the  1d diffusion equation. 
    Arguments
    xo --> Initial point of the space grid(default == 0)
    xn --> Final point of the space grid(default == 1)
    sp_step --> spatial resolution(default == 0.1)
    to --> Initial point of time grid(default == 0
    tn --> Final point of time grid(default == 2000)
    t_step --> time step (default == 0.1)
    a --> diffusion equation coefficient (default == 1e-4)
    """
  
    # Initializing the space and time grid

    space_grid = np.arange(xo, xn+dx, dx)
    time_grid = np.arange(to, tn+dt, dt)

    # Initialize the time
    global field_matrix
    field_matrix = np.zeros((len(time_grid), len(space_grid)))
    
    # Imposing Boundary and initial conditions
    
    field_matrix[0,:] = 0
    field_matrix[:,0] = 10
    field_matrix[:,-1] = 10
    
    # Vectorized Numerical scheme
    
    s = (a*dt)/(dt**2)
    start = time.time()
    j1 = range(2, len(space_grid))
    j2 = range(1, len(space_grid)-1)
    j3 = range(0, len(space_grid)-2)
    
    for i in range(1, len(time_grid)):
        field_matrix[i, 1:-1] = s*field_matrix[i-1, j1] + (1-2*s)*field_matrix[i-1, j2] + s*field_matrix[i-1, j3]
    end = time.time()
    
    # Plotting the solution
    
    # Plotting Results 
    l_time = [100, 500, 5000, 20000] 
    fig1, ax1 = plt.subplots(figsize=(20, 10)) 
    legend2 = [] 
    for i in l_time: 
        ax1.plot(space_grid, field_matrix[i-1, :], lw=5) 
        legend2.append(str(i)) 
        
    plt.title("T-X Distribution over time", fontsize=20) 
    plt.xlabel("Space Domain(m)", fontsize=20) 
    plt.ylabel("Temperature", fontsize=20) 
    plt.legend(legend2, bbox_to_anchor=(1.0, 1.0), fontsize=20, loc="upper center") 
    plt.show() 
    
    fig2, ax2 = plt.subplots(2, 2, figsize=(15, 10)) 
    for j in range(2): 
        w = 0 
        if j == 0: 
            for i in [l_time[0], l_time[1]]: 
                ax2[j][w].plot(space_grid, field_matrix[i-1, :], lw=5, ls="--", label=str(i))
                ax2[j][w].set_xlabel("Space Domain(m)", fontsize=20)
                ax2[j][w].set_ylabel("Temperature", fontsize=20)
                ax2[j][w].set_title("T-X Distribution", fontsize=20)
                ax2[j][w].legend(fontsize=20, loc="upper center")
                w = w+1
        else:
            for i in [l_time[2], l_time[3]]: 
                ax2[j][w].plot(space_grid, field_matrix[i-1, :], lw=5, ls="--", label=str(i))
                ax2[j][w].set_xlabel("Space Domain(m)", fontsize=20)
                ax2[j][w].set_ylabel("Temperature", fontsize=20)
                ax2[j][w].set_title("T-X Distribution", fontsize=20)
                ax2[j][w].legend(fontsize=20, loc="upper center")
                w = w+1 
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.xlim([0, 1])
    plt.ylim([0, 10])
    plt.show()

def twod_poisson_equation(dx=0.02, dy=0.02, xo=0, xn=1, yo=0,yn=1):
  
    """ This function solves the Poisson Equation
    Arguments
    xo --> Initial point of the space grid(default == 0) at x direction
    xn --> Final point of the space grid(default == 1) at x direction
    dx --> Descritization step(default=0.02) at x direction
    yo --> Initial point of the space grid(default == 0) at y direction
    yn --> Final point of the space grid(default == 1) at y direction
    dy --> Descritization step(default=0.02) at x direction
    """
    # Spatial Discretization
    
    x = np.arange(xo, xn+dx, dx)
    y = np.arange(yo, yn+dy, dy)
    
    # Source term 
    
    S = -1*np.ones((len(x), len(y)))
    
    # Matrices Definition
    
    Temp_old_J = np.zeros((len(x), len(y))) 
    Temp_old_G = np.zeros((len(x), len(y))) 
    Temp_old_S1 = np.zeros((len(x), len(y))) 
    Temp_old_S2 = np.zeros((len(x), len(y))) 
    Temp_old_S3 = np.zeros((len(x), len(y)))
    
    # Boundary Conditions Initialization 
    
    Temp_old_J[:, 0], Temp_old_J[:, -1], Temp_old_J[0, :], Temp_old_J[-1, :] = 0, 0, 0, 0 
    Temp_old_G[:, 0], Temp_old_G[:, -1], Temp_old_G[0, :], Temp_old_G[-1, :] = 0, 0, 0, 0 
    Temp_old_S1[:, 0], Temp_old_S1[:, -1], Temp_old_S1[0, :], Temp_old_S1[-1, :] = 0, 0, 0, 0 
    Temp_old_S2[:, 0], Temp_old_S2[:, -1], Temp_old_S2[0, :], Temp_old_S2[-1, :] = 0, 0, 0, 0 
    Temp_old_S3[:, 0], Temp_old_S3[:, -1], Temp_old_S3[0, :], Temp_old_S3[-1, :] = 0, 0, 0, 0
    
    Temp_new_J = Temp_old_J.copy() 
    Temp_new_G = Temp_old_G.copy() 
    Temp_new_S1 = Temp_old_S1.copy() 
    Temp_new_S2 = Temp_old_S2.copy() 
    Temp_new_S3 = Temp_old_S3.copy()
    
    # Numerical Scheme 
    
    rj = 0  # Repetition Counter for Jacob 
    rg = 0  # Repetion Counter for Gauss-Seidel 
    rs1 = 0  # Repetion Counter for SOR L1 
    rs2 = 0  # Repetion Counter for SOR L2 
    rs3 = 0  # Repetition Counter for SOR L3 
    l_c = [0.5, 1.3, 1.8]
    
    j1 = range(1, len(y)-1)
    j2 = range(2, len(y))
    j3 = range(0, len(y)-2)
    
    while True: 
        # Jacobi Scheme 
        while True: 
            rj = rj+1 
            for i in range(1, len(x)-1): 
                Temp_new_J[i, j1] = ((-S[i, j1]*(dx**2))/4) + ((Temp_old_J[i+1, j1] + Temp_old_J[i-1, j1] + Temp_old_J[i, j2] + Temp_old_J[i, j3])/4) 
            if np.max((np.abs(Temp_new_J[1:len(x)-1, 1:len(y)-1]-Temp_old_J[1:len(x)-1, 1:len(y)-1])/Temp_new_J[1:len(x)-1, 1:len(y)-1])) < 1e-5: 
                print(f'Convergence Iterations Jacobi:{rj}') 
                break
            Temp_old_J = Temp_new_J.copy() 
        
        # Gauss-Seidel Numerical Scheme 
        
        while True: 
            rg = rg+1 
            for i in range(1, len(x)-1):  
                Temp_new_G[i, j1] = ((-S[i, j1]*(dx**2))/4) + ((Temp_old_G[i+1, j1] + Temp_new_G[i-1, j1] + Temp_old_G[i, j2] + Temp_new_G[i, j3])/4) 
            if np.max((np.abs(Temp_new_G[1:len(x)-1, 1:len(y)-1]-Temp_old_G[1:len(x)-1, 1:len(y)-1])/Temp_new_G[1:len(x)-1, 1:len(y)-1])) < 1e-5:  
                print(f'Convergence Iterations Gauss-Seidel:{rg}')
                break
            Temp_old_G = Temp_new_G.copy() 
        
        # SOR for l1 
        
        while True: 
            rs1 = rs1+1 
            for i in range(1, len(x)-1): 
                Temp_new_S1[i, j1] = (((-S[i, j1]*(dx**2))/4) + ((Temp_old_S1[i+1, j1] + Temp_new_S1[i-1, j1] + Temp_old_S1[i, j2] + Temp_new_S1[i, j3])/4))*l_c[0]+(1-l_c[0])*(Temp_old_S1[i, j1]) 
            if np.max((np.abs(Temp_new_S1[1:len(x)-1, 1:len(y)-1]-Temp_old_S1[1:len(x)-1, 1:len(y)-1])/Temp_new_S1[1:len(x)-1, 1:len(y)-1])) < 1e-5: 
                print(f'Convergence Iterations SOR for lamda = {l_c[0]}: {rs1}')
                break
            Temp_old_S1 = Temp_new_S1.copy() 
        # SOR for l2 
        
        while True: 
            rs2 = rs2+1 
            for i in range(1, len(x)-1): 
                Temp_new_S2[i, j1] = (((-S[i, j1]*(dx**2))/4) + ((Temp_old_S2[i+1, j1] + Temp_new_S2[i-1, j1] + Temp_old_S2[i, j2] + Temp_new_S2[i, j3])/4))*l_c[1]+(1-l_c[1])*(Temp_old_S2[i, j1]) 
            if np.max((np.abs(Temp_new_S2[1:len(x)-1, 1:len(y)-1]-Temp_old_S2[1:len(x)-1, 1:len(y)-1])/Temp_new_S2[1:len(x)-1, 1:len(y)-1])) < 1e-5: 
                print(f'Convergence Iterations SOR for lamda = {l_c[1]}: {rs2}')
                break
            Temp_old_S2 = Temp_new_S2.copy() 
        while True: 
            rs3 = rs3+1 
            for i in range(1, len(x)-1): 
                Temp_new_S3[i, j1] = (((-S[i, j1]*(dx**2))/4) + ((Temp_old_S3[i+1, j1] + Temp_new_S3[i-1, j1] + Temp_old_S3[i, j2] + Temp_new_S3[i, j3])/4))*l_c[2]+(1-l_c[2])*(Temp_old_S3[i, j1]) 
            if np.max((np.abs(Temp_new_S3[1:len(x)-1, 1:len(y)-1]-Temp_old_S3[1:len(x)-1, 1:len(y)-1])/Temp_new_S3[1:len(x)-1, 1:len(y)-1])) < 1e-5: 
                print(f'Convergence Iterations SOR for lamda = {l_c[2]}: {rs3}')
                break
            Temp_old_S3 = Temp_new_S3.copy() 
        break 
    
    fig, ax = plt.subplots(1, 5, figsize=(30, 5), subplot_kw={"projection": "3d"}) 
    X, Y = np.meshgrid(x, y) 
    T = [Temp_new_J, Temp_new_G, Temp_new_S1, Temp_new_S2, Temp_new_S3] 
    r = [rj, rg, rs1, rs2, rs3] 
    ns = ["Jacobi", "Gauss-Seidel", "SOR-0.5", "SOR-1.3", "SOR-1.8"] 
    
    for i in range(len(T)): 
        ax[i].plot_surface(X, Y, T[i], cmap="coolwarm")
        ax[i].set_xlabel("x(m)"), ax[i].set_ylabel("y(m)"), ax[i].set_zlabel("Temperature")
        ax[i].set_title(f'{ns[i]}:{r[i]} iterations') 
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()

def twod_diffusion_equation(dx=0.01, dy=0.01, xo=0, xn=1, yo=0, yn=1, a=1e-4):
    
    """ 2D Diffusion Equation  using Forward Time Central Space Scheme 
    Arguments
    xo --> Initial point of the space grid(default == 0) at x direction
    xn --> Final point of the space grid(default == 1) at x direction
    dx --> Descritization step(default=0.01) at x direction
    yo --> Initial point of the space grid(default == 0) at y direction
    yn --> Final point of the space grid(default == 1) at y direction
    dy --> Descritization step(default=0.01) at x direction 
    a --> diffusion equation coefficient (default == 1e-4)
    """
  
    # Spatial Discretization
    
    x = np.arange(xo, xn+dx, dx)
    y = np.arange(yo, yn+dy, dy)
    
    # Time Descritization
    
    dt = (dx ** 2) / (12 * a)
    
    # Source Term Matrix 
    
    S = 1e-4 * np.ones((len(x), len(y)))
    
    # Initial Conditions and Boundary Conditions
    
    To = np.zeros((len(x), len(y))) 
    Tn = To.copy()
    
    # # Defining Numerical Scheme Elements 
    
    sx = (a * dt) / (dx ** 2) 
    sy = (a * dt) / (dy ** 2)
    
    # Defining Numerical Scheme Elements 
    
    sx = (a * dt) / (dx ** 2) 
    sy = (a * dt) / (dy ** 2)
    
    error_Convergence = [] 
    error = 1  # Error Initialization
    
    i = 0  # Repetion Counter
    
    X, Y = np.meshgrid(x, y) 
    fig1, ax1 = plt.subplots(1, 3, figsize=(20, 20), subplot_kw={"projection": "3d"})
    
    #  Numerical Scheme
    
    while error > 1e-5: 
        
        i = i + 1

        j1 = range(1, len(y) - 1)
        j2 = range(2, len(y))
        j3 = range(0, len(y) - 2)
        
        for i in range(1, len(x) - 1):
            Tn[i, j1] = To[i, j1] + sx * (To[i + 1, j1]-2*To[i, j1]+To[i-1, j1])+ sy*(To[i, j2]-2*To[i, j1]+To[i, j3])+dt*S[i, j1]
            
        Tn[-1, :] = Tn[-2, :] 
        error = np.max(np.abs((Tn[1:len(x)-1, 1:len(y)-1] - To[1:len(x)-1, 1:len(y)-1])/(Tn[1:len(x)-1, 1:len(y)-1]))) 
        To = Tn.copy()
        
        if i == 12: 
            ax1[0].plot_surface(X, Y, Tn, cmap="viridis"), ax1[0].set_xlabel("x(m)"), ax1[0].set_ylabel("y(m)"), ax1[0].set_zlabel("Temperature")
            ax1[0].set_title("Iteration 12-1s") 
        elif i == 120: 
            ax1[1].plot_surface(X, Y, Tn, cmap="viridis"), ax1[1].set_xlabel("x(m)"), ax1[1].set_ylabel("y(m)"), ax1[1].set_zlabel("Temperature")
            ax1[1].set_title("Iteration 120-10s") 
        elif i == 1200: 
            ax1[2].plot_surface(X, Y, Tn, cmap="viridis"), ax1[2].set_xlabel("x(m)"), ax1[2].set_ylabel("y(m)"), ax1[2].set_zlabel("Temperature")
            ax1[2].set_title("Iteration 1200-100s")
            
        error_Convergence.append(error)
        
    fig3, ax3 = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(20, 20)) 
    ax3.plot_surface(X, Y, Tn, cmap="viridis") 
    ax3.set_xlabel("x(m)"), ax3.set_ylabel("y(m)"), ax3.set_zlabel("Temperature") 
    ax3.set_title("Convergence Temperature Distribution-Iteration 24877-2072.3s")
    
    fig4, ax4 = plt.subplots(figsize=(20, 20)) 
    ax4.plot(error_Convergence, marker="*") 
    ax4.set_xlabel("Iteration") 
    ax4.set_ylabel("Relative Error") 
    ax4.set_title("Relative Error-Iteration")
    
def ftcs_1d_hyberbolic(xo=0, xn=1, dx=0.01, u=1, to = 0, dt = 0.008): 

  """ 1D Hyperbolic Equation using Forward Time, Central space Scheme
      Arguments 
      xo --> Initial point of the space grid(default == 0) at x direction
      xn --> Final point of the space grid(default == 1) at x direction 
      dx --> Descritization step(default=0.01) at x direction 
      to --> Initial point of time grid(default == 0
      t_step --> time step (default == 0.008) 
      u --> Wave Velocity(default ==1)
      """
    
    tn = xn/u
    
    x = np.arange(xo, xn+dx, dx)
    time_grid = np.arange(to, tn+dt, dt)
    
    # Old Solution Vector for Upwind 
    
    Tup_old = np.zeros((len(x), 1)) 
    
    # Initial and Boundary Conditions for Upwind 
    
    Tup_old[0] = 10 
    Tup_old[1:] = 0 
    
    # Old Solution vector for other methods 
    
    Tlf_old = Tup_old.copy() 
    Tlw_old = Tup_old.copy() 
    Tftcs_old = Tup_old.copy() 
    
    # New Solution vectors for all methods 
    
    Tup_new = Tup_old.copy() 
    Tlf_new = Tup_old.copy() 
    Tlw_new = Tup_old.copy() 
    Tftcs_new = Tup_old.copy() 
    
    # Numerical SchemeS Parameters 
    
    C = (u*dt)/dx  # Courant Number
    
    # Upwind Numerical Algorithm 
    
    fig1 = plt.figure()
    
    j1 = range(1, len(x))
    j2 = range(0, len(x)-1)
    
    for i in range(1, len(time_grid)):
        Tup_new[j1] = (1-C)*Tup_old[j1]+C*Tup_old[j2] 
        Tup_old = Tup_new.copy()
        plt.clf(), plt.title(f"Upwind-dt={dt}")
        plt.plot(x[0:-1], Tup_new[0:-1], marker="o", linestyle="-", color="blue")
        plt.xlabel("x(m)")
        plt.ylabel("Temperature")
        plt.pause(0.1) 
    plt.show() 
    
    # Lax-Friedrichs Algorithm 
    
    fig2 = plt.figure() 
    
    j3 = range(1, len(x)-1)
    j4 = range(2, len(x))
    j5 = range(0, len(x)-2)
    
    
    for i in range(1, len(time_grid)): 
        Tlf_new[j3] = ((1/2)-(C/2))*Tlf_old[j4]+((C/2)+(1/2))*Tlf_old[j5] 
        Tlf_old = Tlf_new.copy()
        plt.clf(), plt.title(f"Lax-Friedrichs-dt={dt}")
        plt.plot(x[0:-1], Tlf_new[0:-1], marker="o", linestyle="-", color="red")
        plt.xlabel("x(m)")
        plt.ylabel("Temperature")
        plt.pause(0.1)
    plt.show() 
    
    # Lax-Wendroff Algorithm 
    
    fig3 = plt.figure() 
    
    j6 = range(1, len(x)-1)
    j7 = range(2, len(x))
    j8 = range(0, len(x)-2)
    
    for i in range(1, len(time_grid)): 
        Tlw_new[j6] = Tlw_old[j6]-(C/2)*(Tlw_old[j7]-Tlw_old[j8])+((C**2)/2)*(Tlw_old[j7]-2*Tlw_old[j6]+Tlw_old[j8])
        Tlw_old = Tlw_new.copy()
        plt.clf(), plt.title(f"Lax-Wendroff-dt={dt}")
        plt.plot(x[0:-1], Tlw_new[0:-1], marker="o", linestyle="-", color="green")
        plt.xlabel("x(m)")
        plt.ylabel("Temperature")
        plt.pause(0.1)
    plt.show() 
    
    # FTCS 
    
    fig4 = plt.figure() 
    
    j9 = range(1, len(x)-1)
    j10 = range(2, len(x))
    j11 = range(0, len(x)-2)
    
    for i in range(1, len(time_grid)): 
        Tftcs_new[j9] = Tftcs_old[j9]+(C/2)*(Tftcs_old[j10]-Tftcs_old[j11])
        Tftcs_old = Tftcs_new.copy()
        plt.clf(), plt.title(f"FTCS-dt={dt}")
        plt.plot(x[0:-1], Tftcs_new[0:-1], marker="o", linestyle="-", color="orange")
        plt.xlabel("x(m)")
        plt.ylabel("Temperature")
        plt.pause(0.1) 
    plt.show()
    

def convection_diffusion(xo = 0, xn=1, dx=0.01, a=1e-4, u=0.001):

  """ Convenction Diffusion
      Arguments 
      xo --> Initial point of the space grid(default == 0) at x direction
      xn --> Final point of the space grid(default == 1) at x direction 
      dx --> Descritization step(default=0.01) at x direction 
      u --> Wave Velocity(default == 0.001)
      a --> diffusion equation coefficient (default == 1e-4) 
  """
  
    x = np.arange(xo, xn+dx, dx)
    
    Told = np.zeros((len(x), 1)) 
    Told[0] = 10 
    Told[-1] = 0
    
    Tnew = Told.copy()
    
    # Numerical Scheme Selection 
    
    
    choice = int(input("Choose Numerical Scheme: 0 for CD-Convection/CD Diffurion or 1 for Upwind-Convection/CD-Diffustion:"))
    
    
    if choice == 0: 
        
        # Time Discretization 
        # Minimum Time Step  
        
        if u != 0: 
            dt1 = (dx**2)/(2*a) 
            dt2 = np.sqrt((dx**2)/abs(u**2)) 
            max_dt = min(dt1, dt2) 
        else: 
            max_dt = (dx**2) / (2 * a) 
        
        dt_c = float(input(f"The maximum time step should be {max_dt:.4f}. Please select this or a lower time step: ")) 
        
        # Courant and s Number 
        
        C = (u*dt_c)/dx
        s = (a*dt_c)/(dx**2) 
        
        # Model Coefficients 
        ac_e = -C/2
        ad_e = s
        ac_p = 0
        ad_p = -2*s
        ac_w = C/2
        ad_w = s 
    
    if choice == 1: 
        
        # Time Discretization 
        # Minimum Time Step 
        
        max_dt = (dx**2)/(2*a+(abs(u)*dx))
        dt_c = float(input(f"The minimum time step should be {max_dt:.4f}. Please select this or a lower time step: ")) 
        
        # Model Coefficients
        # Courant and s Number

        C = (u*dt_c)/dx
        s = (a*dt_c)/(dx**2)

        # Model Coefficients 
        
        if u >= 0: 
            ac_e = 0
            ad_e = s
            ac_p = -C
            ad_p = -2*s
            ac_w = C
            ad_w = s 
        else:
            ac_e = -C
            ad_e = s
            ac_p = C
            ad_p = -2*s
            ac_w = 0
            ad_w = s
        
        # Explicit Numerical Scheme Coefficients for Convection-Diffusion Problems 
        
        ae = ac_e+ad_e
        ap = 1+ac_p+ad_p
        aw = ac_w+ad_w

       # Numerical Algorithm

        error = 1  # error initalization for Convergence
        j = 0  # Counter for Repetitions
        k = 0  # Plot Counter
        times = np.array([100, 200, 500])  # Time in (s) for Visualization
        times_in_rep = [int(np.round(times[0]/dt_c)), int(np.round(times[1]/dt_c)), int(np.round(times[2]/dt_c))]

        fig1, ax1 = plt.subplots(1, 1, figsize=(20, 20)) 
       
        while error > 1e-5: 
            j = j+1 
           
            for i in range(1, len(x)-1): 
                Tnew[i] = ae*Told[i+1]+ap*Told[i]+aw*Told[i-1] 
            error = np.nanmax(np.abs((Tnew[1:len(x) - 1] - Told[1:len(x) - 1]) / (Tnew[1:len(x) - 1])))
            Told = Tnew.copy()
            # Visualization
            # Time in Repetitions
            if j in times_in_rep: 
                ax1.plot(x, Tnew, label=f"{times[k]}") 
                ax1.set_xlabel("X(m") 
                ax1.set_ylabel("Temperature") 
                k = k+1 
        
        total_rep = j 
        total_time = total_rep*dt_c 
        ax1.plot(x, Tnew, label=f"Convergence-{total_time}s") 
        ax1.legend() 
        
        if choice == 0: 
            ax1.set_title("FTCS") 
        else: 
            ax1.set_title("UpWind") 
        
        print(f"The total repetitions need it for convergence is: {total_rep}")
        print(f"The total time need it for convergence is: {total_time} s")

        plt.show()  
    
if __name__ == "__main__" :  
    
    oned_diffusion_ftcs()
    twod_diffusion_equation()
    ftcs_1d_hyberbolic()
    convection_diffusion()
    twod_poisson_equation()
