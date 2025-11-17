import numpy as np
import matplotlib.pyplot as plt
import quaternion
from filterpy.kalman import KalmanFilter, ExtendedKalmanFilter

plt.style.use('seaborn-v0_8')


# 1. GERAÇÃO DO GROUND TRUTH
np.random.seed(42)
dt = 0.1
t = np.arange(0, 60, dt)
g = 9.80665
pathsize = 15
traj_choice = 4 # Escolha entre 1,2,3,4

def gerar_trajetoria(path, t):
    if path==1:
        w=0.25; x=pathsize*np.cos(w*t); y=pathsize*np.sin(w*t); vx=-w*pathsize*np.sin(w*t); vy=w*pathsize*np.cos(w*t); ax=-(w**2)*pathsize*np.cos(w*t); ay=-(w**2)*pathsize*np.sin(w*t)
    elif path==2:
        w1,w2=0.5,1.0; x=pathsize*np.cos(w1*t); y=pathsize*np.sin(w2*t); vx=-pathsize*w1*np.sin(w1*t); vy=pathsize*w2*np.cos(w2*t); ax=-pathsize*(w1**2)*np.cos(w1*t); ay=-pathsize*(w2**2)*np.sin(w2*t)
    elif path==3:
        w1,w2=0.125,0.25; x=pathsize*np.cos(w1*t); y=pathsize*np.sin(w2*t); vx=-pathsize*w1*np.sin(w1*t); vy=pathsize*w2*np.cos(w2*t); ax=-pathsize*(w1**2)*np.cos(w1*t); ay=-pathsize*(w2**2)*np.sin(w2*t)
    elif path==4:
        w1,w2=1.5,1.0; x=pathsize*np.sin(w1*t); y=pathsize*np.cos(w2*t); vx=pathsize*w1*np.cos(w1*t); vy=-pathsize*w2*np.sin(w2*t); ax=-pathsize*(w1**2)*np.sin(w1*t); ay=-pathsize*(w2**2)*np.cos(w2*t)
    return x,y,vx,vy,ax,ay

x_true, y_true, vx_true, vy_true, ax_true, ay_true = gerar_trajetoria(traj_choice, t)
psi_true = np.arctan2(vy_true, vx_true)


# 2. SIMULAÇÃO DOS SENSORES RUIDOSOS

# Parâmetros de ruído
acc_noise_std = 0.05
gyro_noise_std = np.deg2rad(0.01)
mag_noise_std = np.deg2rad(2.0)
gps_noise_std = 0.2

# Simulação da IMU
psi_dot_true = np.gradient(np.unwrap(psi_true), dt)
gyro_z_noisy = psi_dot_true + np.random.normal(0, gyro_noise_std, size=len(t))
mag_yaw_noisy = psi_true + np.random.normal(0, mag_noise_std, size=len(t))

def Rz_3d(psi):
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0],[s, c, 0],[0,0,1]])

a_world_3d = np.vstack([ax_true, ay_true, np.zeros_like(t)])
gravity = np.array([0,0,g]).reshape(3,1)
acc_body_noisy = np.zeros((3, len(t)))
for k in range(len(t)):
    Rbw = Rz_3d(psi_true[k]).T
    acc_body_noisy[:,k] = (Rbw @ (a_world_3d[:,k].reshape(3,1)+gravity)).ravel()
acc_body_noisy += np.random.normal(0, acc_noise_std, size=acc_body_noisy.shape)

# Simulação do GPS
x_gps = x_true + np.random.normal(0, gps_noise_std, size=len(t))
y_gps = y_true + np.random.normal(0, gps_noise_std, size=len(t))



# 3. MÉTODO 1: HÍBRIDO (MK)

# Estimação de Yaw com Filtro Complementar
beta = 0.15
q_est_mk = quaternion.quaternion(1,0,0,0)
yaw_est_mk = np.zeros(len(t))
for i in range(len(t)):
    q_pred = q_est_mk * quaternion.from_euler_angles(0,0, gyro_z_noisy[i-1]*dt if i>0 else 0)
    q_meas = quaternion.from_euler_angles(0,0, mag_yaw_noisy[i])
    q_est_mk = quaternion.slerp(q_pred, q_meas, 0, 1, beta).normalized()
    yaw_est_mk[i] = quaternion.as_euler_angles(q_est_mk)[2]

# KF Linear para Posição/Velocidade
a_world_est_mk = np.zeros((2, len(t)))
for i in range(len(t)):
    Rwb = Rz_3d(yaw_est_mk[i])[:2,:2] 
    a_body_2d = Rwb @ acc_body_noisy[:2, i]
    a_world_est_mk[:,i] = a_body_2d

kf_mk = KalmanFilter(dim_x=4, dim_z=2)
kf_mk.x = np.array([x_gps[0], y_gps[0], vx_true[0], vy_true[0]])
kf_mk.F = np.array([[1,0,dt,0], [0,1,0,dt], [0,0,1,0], [0,0,0,1]])
kf_mk.H = np.array([[1,0,0,0], [0,1,0,0]])
kf_mk.B = np.array([[0.5*dt**2, 0], [0, 0.5*dt**2], [dt, 0], [0, dt]])
kf_mk.P *= 10
kf_mk.R = np.diag([gps_noise_std**2, gps_noise_std**2])
kf_mk.Q = np.diag([0.01, 0.01, 0.1, 0.1])

mk_states, _, _, _ = kf_mk.batch_filter(zs=np.vstack([x_gps, y_gps]).T, us=a_world_est_mk.T)
x_est_mk, y_est_mk, vx_est_mk, vy_est_mk = mk_states.T


# 4. MÉTODO 2: EKF 

def f_ekf(x, dt, u_acc_body, u_gyro_z):
    px, py, vx, vy, yaw = x
    ax_body, ay_body = u_acc_body
    c, s = np.cos(yaw), np.sin(yaw)
    ax_world, ay_world = c*ax_body - s*ay_body, s*ax_body + c*ay_body
    
    px_new = px + vx*dt + 0.5*ax_world*dt**2
    py_new = py + vy*dt + 0.5*ay_world*dt**2
    vx_new = vx + ax_world*dt
    vy_new = vy + ay_world*dt
    yaw_new = yaw + u_gyro_z*dt
    return np.array([px_new, py_new, vx_new, vy_new, np.arctan2(np.sin(yaw_new), np.cos(yaw_new))])

def F_jac(x, dt, u_acc_body):
    vx, vy, yaw = x[2], x[3], x[4]
    ax_body, ay_body = u_acc_body
    c, s = np.cos(yaw), np.sin(yaw)
    
    d_ax_d_yaw = -s*ax_body - c*ay_body
    d_ay_d_yaw =  c*ax_body - s*ay_body

    F = np.eye(5)
    F[0,2], F[1,3] = dt, dt
    F[0,4] = 0.5 * d_ax_d_yaw * dt**2
    F[1,4] = 0.5 * d_ay_d_yaw * dt**2
    F[2,4] = d_ax_d_yaw * dt
    F[3,4] = d_ay_d_yaw * dt
    return F

ekf = ExtendedKalmanFilter(dim_x=5, dim_z=2) # [x, y, vx, vy, yaw]
ekf.x = np.array([x_gps[0], y_gps[0], vx_true[0], vy_true[0], psi_true[0]])
ekf.P = np.diag([gps_noise_std**2, gps_noise_std**2, 0.1, 0.1, np.deg2rad(1)**2])
ekf.R = np.diag([gps_noise_std**2, gps_noise_std**2])
q_pos, q_vel, q_yaw = 0.05, 0.1, np.deg2rad(0.5)
ekf.Q = np.diag([q_pos**2, q_pos**2, q_vel**2, q_vel**2, q_yaw**2])

H_ekf = np.array([[1,0,0,0,0], [0,1,0,0,0]])

ekf_states = []
for i in range(len(t)):
    u_acc = acc_body_noisy[:2, i]
    u_gyro = gyro_z_noisy[i]
    z = np.array([x_gps[i], y_gps[i]])

    # Guarda o estado atual antes da predição
    x_prior = ekf.x.copy()

    # 1. Calcula a Jacobiana com base no estado atual
    Fk = F_jac(x_prior, dt, u_acc)

    # 2. Atribui a Jacobiana ao filtro
    ekf.F = Fk

    # 3. Chama a predição da biblioteca (sem argumentos) para atualizar a covariância
    ekf.predict()

    # 4. Predição NÃO-LINEAR do estado
    ekf.x = f_ekf(x_prior, dt, u_acc, u_gyro)

    # 5. Passo de atualização 
    ekf.update(z, HJacobian=lambda x: H_ekf, Hx=lambda x: H_ekf @ x)
    ekf_states.append(ekf.x.copy())
    

ekf_states = np.array(ekf_states)
x_est_ekf, y_est_ekf, vx_est_ekf, vy_est_ekf, _ = ekf_states.T


# 5. CÁLCULO DE ERROS E PLOTS

# Cálculo de Erros
def calculate_errors(x_est, y_est, vx_est, vy_est, x_true, y_true, vx_true, vy_true, dt):
    pos_err = np.sqrt((x_est - x_true)**2 + (y_est - y_true)**2)
    vel_err = np.sqrt((vx_est - vx_true)**2 + (vy_est - vy_true)**2)
    ax_est, ay_est = np.gradient(vx_est, dt), np.gradient(vy_est, dt)
    ax_true, ay_true = np.gradient(vx_true, dt), np.gradient(vy_true, dt)
    acc_err = np.sqrt((ax_est - ax_true)**2 + (ay_est - ay_true)**2)
    
    mae_pos, rmse_pos = np.mean(pos_err), np.sqrt(np.mean(pos_err**2))
    mae_vel, rmse_vel = np.mean(vel_err), np.sqrt(np.mean(vel_err**2))
    mae_acc, rmse_acc = np.mean(acc_err), np.sqrt(np.mean(acc_err**2))
    
    return (pos_err, vel_err, acc_err), (mae_pos, rmse_pos, mae_vel, rmse_vel, mae_acc, rmse_acc)

(err_pos_mk, err_vel_mk, err_acc_mk), (mae_pos_mk, rmse_pos_mk, mae_vel_mk, rmse_vel_mk, mae_acc_mk, rmse_acc_mk) = \
    calculate_errors(x_est_mk, y_est_mk, vx_est_mk, vy_est_mk, x_true, y_true, vx_true, vy_true, dt)
(err_pos_ekf, err_vel_ekf, err_acc_ekf), (mae_pos_ekf, rmse_pos_ekf, mae_vel_ekf, rmse_vel_ekf, mae_acc_ekf, rmse_acc_ekf) = \
    calculate_errors(x_est_ekf, y_est_ekf, vx_est_ekf, vy_est_ekf, x_true, y_true, vx_true, vy_true, dt)
    
# Plots
plt.figure(figsize=(8,8))
plt.plot(x_true, y_true, 'k-', label="Ground Truth", linewidth=2)
plt.scatter(x_gps, y_gps, s=10, c='g', alpha=0.3, label="GPS Ruidoso")
plt.plot(x_est_mk, y_est_mk, 'b-', label="MK")
plt.plot(x_est_ekf, y_est_ekf, 'r-', label="EKF")
plt.xlabel("x [m]"); plt.ylabel("y [m]"); plt.grid(); plt.legend(); plt.axis('equal'); plt.show()

def plot_error(t, err_mk, err_ekf, title, ylabel):
    plt.figure(figsize=(10, 5))
    plt.plot(t, err_mk, 'b-', label="MK")
    plt.plot(t, err_ekf, 'r-', label="EKF")
    plt.title(title); plt.xlabel("Tempo [s]"); plt.ylabel(ylabel)
    plt.grid(); plt.legend(); plt.show()

plot_error(t, err_pos_mk, err_pos_ekf, "Erro de Posição", "Erro [m]")
plot_error(t, err_vel_mk, err_vel_ekf, "Erro de Velocidade", "Erro [m/s]")
plot_error(t, err_acc_mk, err_acc_ekf, "Erro de Aceleração", "Erro [m/s²]")


# 6. GERAÇÃO DA TABELA LATEX

def reduction_pct(mk_error, ekf_error):
    if mk_error == 0:
        return 0.0
    return 100.0 * (mk_error - ekf_error) / mk_error

# Cálculo das reduções percentuais para cada métrica
red_mae_pos = reduction_pct(mae_pos_mk, mae_pos_ekf)
red_rmse_pos = reduction_pct(rmse_pos_mk, rmse_pos_ekf)
red_mae_vel = reduction_pct(mae_vel_mk, mae_vel_ekf)
red_rmse_vel = reduction_pct(rmse_vel_mk, rmse_vel_ekf)
red_mae_acc = reduction_pct(mae_acc_mk, mae_acc_ekf)
red_rmse_acc = reduction_pct(rmse_acc_mk, rmse_acc_ekf)

latex_table_string = f"""
\\begin{{table}}[H]
\\centering
\\caption{{Comparação de métodos - Trajetória Simulada {traj_choice}}}
\\rowcolors{{2}}{{gray!10}}{{white}}
\\begin{{tabular}}{{lcccccc}}
\\toprule
 & \\multicolumn{{2}}{{c}}{{MAE}} & \\multicolumn{{2}}{{c}}{{RMSE}} & \\multicolumn{{2}}{{c}}{{Redução \\%}} \\\\
\\cmidrule(r){{2-3}} \\cmidrule(r){{4-5}} \\cmidrule(r){{6-7}}
\\textbf{{Variável}} & \\textbf{{MK}} & \\textbf{{EKF}} & \\textbf{{MK}} & \\textbf{{EKF}} & \\textbf{{MAE}} & \\textbf{{RMSE}} \\\\
\\hline
Posição [m] & {mae_pos_mk:.3f} & {mae_pos_ekf:.3f} & {rmse_pos_mk:.3f} & {rmse_pos_ekf:.3f} & {red_mae_pos:+.1f}\\% & {red_rmse_pos:+.1f}\\% \\\\
Velocidade [m/s] & {mae_vel_mk:.3f} & {mae_vel_ekf:.3f} & {rmse_vel_mk:.3f} & {rmse_vel_ekf:.3f} & {red_mae_vel:+.1f}\\% & {red_rmse_vel:+.1f}\\% \\\\
Aceleração [m/s²] & {mae_acc_mk:.3f} & {mae_acc_ekf:.3f} & {rmse_acc_mk:.3f} & {rmse_acc_ekf:.3f} & {red_mae_acc:+.1f}\\% & {red_rmse_acc:+.1f}\\% \\\\
\\bottomrule
\\end{{tabular}}
\\label{{tab:sim_path_{traj_choice}}}
%\\fonte{{Autora (2025).}} % Descomente se precisar
\\end{{table}}
"""

print("--- Tabela LaTeX Gerada ---")
print(latex_table_string)