import numpy as np
import matplotlib.pyplot as plt

#CONFIGURAÇÃO 
np.random.seed(42)
dt = 0.1
t = np.arange(0, 20, dt)
g = 9.80665

x = np.sin(3*t)
y = np.cos(2*t)
vx = np.gradient(x, dt)
vy = np.gradient(y, dt)
ax_w = np.gradient(vx, dt)
ay_w = np.gradient(vy, dt)

psi = 0.6*np.sin(0.7*t)
psi_dot = np.gradient(psi, dt)

def Rz(psi):
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])

def normalize(v, eps=1e-12):
    n = np.linalg.norm(v)
    return v if n < eps else v/n

#IMU 
acc_noise_std = 0.08 * g * 0.01
gyro_noise_std = np.deg2rad(0.01)
mag_noise_std  = np.deg2rad(2.0)

a_world_3d = np.vstack([ax_w, ay_w, np.zeros_like(t)])
gravity = np.array([0, 0, g]).reshape(3,1)

acc_body = np.zeros((3, len(t)))
for k in range(len(t)):
    Rbw = Rz(psi[k]).T
    acc_body[:, k] = (Rbw @ (a_world_3d[:, k].reshape(3,1) + gravity)).ravel()
acc_body += np.random.normal(0, acc_noise_std, size=acc_body.shape)
gyro_z = psi_dot + np.random.normal(0, gyro_noise_std, size=len(t))

gps_noise_std = 0.3
x_gps = x + np.random.normal(0, gps_noise_std, size=len(t))
y_gps = y + np.random.normal(0, gps_noise_std, size=len(t))

#QUATERNION 
def quat_mul(q1, q2):
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quat_from_gyro(wz, dt):
    #Rotação em torno de z
    dpsi = wz*dt
    return np.array([np.cos(dpsi/2), 0.0, 0.0, np.sin(dpsi/2)])

def R_from_quat(q):
    w,x,y,z = q
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1-2*(x*x+z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1-2*(x*x+y*y)]
    ])
    return R

#MODELO NÃO LINEAR 
def f_nonlin(xk, uk):
    """
    xk = [x, y, vx, vy, qw, qx, qy, qz]
    uk = [ax_body, ay_body, az_body, wz]
    """
    x, y, vx, vy, qw, qx, qy, qz = xk
    q = normalize(np.array([qw, qx, qy, qz]))
    Rwb = R_from_quat(q)  
    ab = np.array(uk[:3]).reshape(3,1)
    a_world = (Rwb @ ab).ravel() - np.array([0,0,g])


    vxn = vx + a_world[0]*dt
    vyn = vy + a_world[1]*dt
    xn  = x  + vx*dt + 0.5*a_world[0]*dt*dt
    yn  = y  + vy*dt + 0.5*a_world[1]*dt*dt

    wz = uk[3]
    dq = quat_from_gyro(wz, dt)
    qn = normalize(quat_mul(q, dq))

    return np.array([xn, yn, vxn, vyn, qn[0], qn[1], qn[2], qn[3]])

def h_meas(xk):
    #mede posição (GPS)
    return np.array([xk[0], xk[1]])

#Jacobianos 
def numerical_jacobian_f(xk, uk, eps=1e-6):
    n = len(xk)
    F = np.zeros((n, n))
    fx = f_nonlin(xk, uk)
    for i in range(n):
        dx = np.zeros(n); dx[i] = eps
        F[:, i] = (f_nonlin(xk+dx, uk) - fx)/eps
    return F

def numerical_jacobian_h(xk, eps=1e-6):
    n = len(xk)
    m = 2
    H = np.zeros((m, n))
    hx = h_meas(xk)
    for i in range(n):
        dx = np.zeros(n); dx[i] = eps
        H[:, i] = (h_meas(xk+dx) - hx)/eps
    return H

#EKF
xk = np.array([x_gps[0], y_gps[0], 0., 0., 1., 0., 0., 0.])
P  = np.diag([10,10, 10,10,  0.1,0.1,0.1,0.1]) * 10.0
Q  = np.diag([0.05,0.05, 0.1,0.1, 0.001,0.001,0.001,0.001]) 
R  = np.eye(2) * (gps_noise_std**2)                         

x_est, y_est, vx_est, vy_est = [], [], [], []

for k in range(len(t)):
    uk = [acc_body[0, k], acc_body[1, k], acc_body[2, k], gyro_z[k]]

    #Predição
    Fk = numerical_jacobian_f(xk, uk)
    x_pred = f_nonlin(xk, uk)
    P_pred = Fk @ P @ Fk.T + Q

    #Atualização (GPS)
    zk = np.array([x_gps[k], y_gps[k]])
    Hk = numerical_jacobian_h(x_pred)
    yk = zk - h_meas(x_pred)
    Sk = Hk @ P_pred @ Hk.T + R
    K  = P_pred @ Hk.T @ np.linalg.inv(Sk)
    xk = x_pred + K @ yk
    #normaliza quaternion
    xk[4:8] = normalize(xk[4:8])
    P  = (np.eye(len(xk)) - K @ Hk) @ P_pred

    x_est.append(xk[0]); y_est.append(xk[1])
    vx_est.append(xk[2]); vy_est.append(xk[3])

x_est = np.array(x_est); y_est = np.array(y_est)
vx_est = np.array(vx_est); vy_est = np.array(vy_est)

#Acelerações estimadas pela derivada da velocidade
ax_est = np.gradient(vx_est, dt)
ay_est = np.gradient(vy_est, dt)

#PLOTS
plt.figure(figsize=(7,6))
plt.plot(x, y, 'k-', label="Real")
plt.scatter(x_gps, y_gps, s=10, c='r', alpha=0.5, label="GPS (ruidoso)")
plt.plot(x_est, y_est, 'b-', label="EKF")
plt.xlabel("x [m]"); plt.ylabel("y [m]")
plt.legend(); plt.grid(); plt.show()

fig, axs = plt.subplots(2, 1, figsize=(8,6), sharex=True)
axs[0].plot(t, vx, color="blue", label="Vx Real")
axs[0].plot(t, vx_est, color="red", label="Vx Estimado")
axs[0].legend(); axs[0].set_ylabel("Vx [m/s]"); axs[0].grid()

axs[1].plot(t, vy, color="blue", label="Vy Real")
axs[1].plot(t, vy_est, color="red", label="Vy Estimado")
axs[1].legend(); axs[1].set_ylabel("Vy [m/s]"); axs[1].set_xlabel("Tempo [s]"); axs[1].grid()
plt.show()

fig, axs = plt.subplots(2, 1, figsize=(8,6), sharex=True)
axs[0].plot(t, ax_w, color="blue", label="Ax Real")
axs[0].plot(t, ax_est, color="red", label="Ax Estimado")
axs[0].legend(); axs[0].set_ylabel("Ax [m/s²]"); axs[0].grid()

axs[1].plot(t, ay_w, color="blue", label="Ay Real")
axs[1].plot(t, ay_est, color="red", label="Ay Estimado")
axs[1].legend(); axs[1].set_ylabel("Ay [m/s²]"); axs[1].set_xlabel("Tempo [s]"); axs[1].grid()
plt.show()

#MÉTRICAS e TABELA Latex
def mae(a,b): return np.mean(np.abs(a-b))
def rmse(a,b): return np.sqrt(np.mean((a-b)**2))

mae_vx = mae(vx, vx_est); mae_vy = mae(vy, vy_est)
rmse_vx = rmse(vx, vx_est); rmse_vy = rmse(vy, vy_est)
mae_ax = mae(ax_w, ax_est); mae_ay = mae(ay_w, ay_est)
rmse_ax = rmse(ax_w, ax_est); rmse_ay = rmse(ay_w, ay_est)

print("\n--- Métricas de Erro (EKF) ---")
print(f"MAE Vx: {mae_vx:.4f} m/s | MAE Vy: {mae_vy:.4f} m/s")
print(f"RMSE Vx: {rmse_vx:.4f} m/s | RMSE Vy: {rmse_vy:.4f} m/s")
print(f"MAE Ax: {mae_ax:.4f} m/s² | MAE Ay: {mae_ay:.4f} m/s²")
print(f"RMSE Ax: {rmse_ax:.4f} m/s² | RMSE Ay: {rmse_ay:.4f} m/s²")

latex_table = f"""
\\begin{{table}}[h]
\\centering
\\caption{{Erros de velocidade e aceleração nos eixos $x$ e $y$ - EKF}}
\\label{{tab:erros_ekf}}
\\begin{{tabular}}{{lcccc}}
\\hline
 & $\\mathrm{{MAE}}_x$ & $\\mathrm{{MAE}}_y$ & $\\mathrm{{RMSE}}_x$ & $\\mathrm{{RMSE}}_y$ \\\\
\\hline
Velocidade (m/s)     & {mae_vx:.3f} & {mae_vy:.3f} & {rmse_vx:.3f} & {rmse_vy:.3f} \\\\
Aceleração (m/s$^2$) & {mae_ax:.3f} & {mae_ay:.3f} & {rmse_ax:.3f} & {rmse_ay:.3f} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""
print("\n--- Tabela LaTeX ---")
print(latex_table)