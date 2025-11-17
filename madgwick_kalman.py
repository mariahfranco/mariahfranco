import numpy as np
import matplotlib.pyplot as plt

#CONFIGURAÇÃO 
np.random.seed(42)
dt = 0.1
t = np.arange(0, 20, dt)
g = 9.80665

#Trajetória verdadeira 
x = np.sin(3*t)
y = np.cos(2*t)

#Velocidade e aceleração verdadeiras
vx = np.gradient(x, dt)
vy = np.gradient(y, dt)
ax_w = np.gradient(vx, dt)
ay_w = np.gradient(vy, dt)

#Yaw verdadeiro e sua derivada
psi = 0.6*np.sin(0.7*t)                # rad
psi_dot = np.gradient(psi, dt)         # rad/s

#Funções úteis 
def Rz(psi):
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])

def normalize(v, eps=1e-12):
    n = np.linalg.norm(v)
    return v if n < eps else v/n

#Aceleração específica (linear + gravidade)
a_world_3d = np.vstack([ax_w, ay_w, np.zeros_like(t)])  
gravity = np.array([0, 0, g]).reshape(3,1)

#Rotaciona para o corpo e adiciona ruído 
acc_noise_std = 0.08 * g * 0.01   
gyro_noise_std = np.deg2rad(0.3)  
mag_noise_std  = np.deg2rad(2.0)  

acc_body = np.zeros((3, len(t)))
gyro_z   = psi_dot + np.random.normal(0, gyro_noise_std, size=len(t))  # giroscópio z
mag_yaw  = psi + np.random.normal(0, mag_noise_std, size=len(t))       # magnetômetro 

for k in range(len(t)):
    Rbw = Rz(psi[k]).T  # corpo <- mundo
    acc_body[:, k] = Rbw @ (a_world_3d[:, k].reshape(3,1) + gravity).ravel()
acc_body += np.random.normal(0, acc_noise_std, size=acc_body.shape)

#GPS 
gps_noise_std = 0.3
x_gps = x + np.random.normal(0, gps_noise_std, size=len(t))
y_gps = y + np.random.normal(0, gps_noise_std, size=len(t))

#MADGWICK 

def quat_from_yaw(psi):
    # Rotação em torno de Z
    cz = np.cos(psi/2)
    sz = np.sin(psi/2)
    return np.array([cz, 0.0, 0.0, sz])  # [w, x, y, z]

def yaw_from_quat(q):
    w, x, y, z = q
    return 2*np.arctan2(z, w)

#misturando giroscópio e magnetômetro (com ganho beta)
beta = 0.15
q_est = np.zeros((4, len(t)))
q = quat_from_yaw(mag_yaw[0])  
q_est[:, 0] = q

for k in range(1, len(t)):
    # Predição por gyro 
    wz = gyro_z[k-1]
    dpsi = wz * dt
    dq = quat_from_yaw(dpsi)              

    w,x,y,z = q
    W,X,Y,Z = dq
    q_pred = np.array([
        w*W - x*X - y*Y - z*Z,
        w*X + x*W + y*Z - z*Y,
        w*Y - x*Z + y*W + z*X,
        w*Z + x*Y - y*X + z*W
    ])
    q_pred = normalize(q_pred)

    #Correção pelo magnetômetro 
    psi_meas = mag_yaw[k]
    q_meas = quat_from_yaw(psi_meas)

    #interpolação "tipo madgwick"
    q = normalize((1-beta)*q_pred + beta*q_meas)
    q_est[:, k] = q

#Aceleração no mundo estimada com a orientação estimada
a_world_est = np.zeros((3, len(t)))
for k in range(len(t)):
    psi_hat = yaw_from_quat(q_est[:, k])
    Rwb = Rz(psi_hat)           
    a_world_est[:, k] = (Rwb @ acc_body[:, k].reshape(3,1)).ravel() - gravity.ravel()

ax_hat_in = a_world_est[0, :]
ay_hat_in = a_world_est[1, :]

#KALMAN LINEAR (com entrada = aceleração no mundo) 
#Estado: [x, y, vx, vy]
F = np.array([[1, 0, dt, 0],
              [0, 1, 0, dt],
              [0, 0, 1,  0],
              [0, 0, 0,  1]])
B = np.array([[0.5*dt*dt, 0],
              [0, 0.5*dt*dt],
              [dt, 0],
              [0, dt]])
H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])

xk = np.array([x_gps[0], y_gps[0], 0., 0.])
P  = np.eye(4)*1000
Q  = np.eye(4)*0.001
R  = np.eye(2)*(gps_noise_std**2)

x_est, y_est, vx_est, vy_est = [], [], [], []
for k in range(len(t)):
    uk = np.array([ax_hat_in[k], ay_hat_in[k]])  
    #Predição
    xk = F @ xk + B @ uk
    P  = F @ P @ F.T + Q
    #Atualização
    zk = np.array([x_gps[k], y_gps[k]])
    yk = zk - H @ xk
    S  = H @ P @ H.T + R
    K  = P @ H.T @ np.linalg.inv(S)
    xk = xk + K @ yk
    P  = (np.eye(4) - K @ H) @ P
 
    x_est.append(xk[0]); y_est.append(xk[1])
    vx_est.append(xk[2]); vy_est.append(xk[3])

x_est = np.array(x_est); y_est = np.array(y_est)
vx_est = np.array(vx_est); vy_est = np.array(vy_est)

#Aceleração estimada a partir da velocidade filtrada
ax_est = np.gradient(vx_est, dt)
ay_est = np.gradient(vy_est, dt)

#PLOTS 
plt.figure(figsize=(7,6))
plt.plot(x, y, 'k-', label="Real")
plt.scatter(x_gps, y_gps, s=10, c='r', alpha=0.5, label="GPS (ruidoso)")
plt.plot(x_est, y_est, 'b-', label="Madgwick + LKF")
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

print("\n--- Métricas de Erro (Madgwick + LKF) ---")
print(f"MAE Vx: {mae_vx:.4f} m/s | MAE Vy: {mae_vy:.4f} m/s")
print(f"RMSE Vx: {rmse_vx:.4f} m/s | RMSE Vy: {rmse_vy:.4f} m/s")
print(f"MAE Ax: {mae_ax:.4f} m/s² | MAE Ay: {mae_ay:.4f} m/s²")
print(f"RMSE Ax: {rmse_ax:.4f} m/s² | RMSE Ay: {rmse_ay:.4f} m/s²")

latex_table = f"""
\\begin{{table}}[h]
\\centering
\\caption{{Erros de velocidade e aceleração nos eixos $x$ e $y$ - Madgwick + LKF}}
\\label{{tab:erros_madgwick_lkf}}
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