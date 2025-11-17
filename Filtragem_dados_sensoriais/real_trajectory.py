import numpy as np
import matplotlib.pyplot as plt
import quaternion
from filterpy.kalman import KalmanFilter, ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise

plt.style.use('seaborn-v0_8')


# 1. CARREGAR DADOS DE REFERÊNCIA DO ARQUIVO .DAT

try:
    trajeto = np.loadtxt("trajeto7.dat", unpack=True, delimiter=',')
    dN_ref, dE_ref, vN_ref, vE_ref, aN_ref, aE_ref, _, _, _, _, yaw_ref = trajeto
except FileNotFoundError:
    print("Erro: Arquivo não encontrado.")
    exit()

n = len(dN_ref)
dt = 1.0  

# 2. PRÉ-PROCESSAMENTO DAS ENTRADAS DA IMU

def rotz_2d(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])

a_body = np.zeros((n, 2))
for i in range(n):
    a_body[i, :] = rotz_2d(yaw_ref[i]).T @ np.array([aN_ref[i], aE_ref[i]])

gyro_z = np.gradient(np.unwrap(yaw_ref), dt)


# 3. MÉTODO 1: MADGWICK + KALMAN LINEAR (MK)

class ComplementaryFilter:
    def __init__(self, beta=0.1):
        self.beta = beta
        self.q = quaternion.quaternion(1,0,0,0)
    def update(self, gyro_z, mag_yaw, dt):
        d_psi = gyro_z * dt
        dq = quaternion.from_euler_angles(0,0,d_psi)
        q_pred = self.q * dq
        q_meas = quaternion.from_euler_angles(0,0,mag_yaw)
        self.q = quaternion.slerp(q_pred, q_meas, 0, 1, self.beta).normalized()
    def get_yaw(self):
        return quaternion.as_euler_angles(self.q)[0]

comp_filter = ComplementaryFilter(beta=0.15)
yaw_mk = np.zeros(n)
for i in range(n):
    comp_filter.update(gyro_z[i], yaw_ref[i], dt)
    yaw_mk[i] = comp_filter.get_yaw()

a_nav_mk_input = np.zeros((n, 2))
for i in range(n):
    a_nav_mk_input[i] = rotz_2d(yaw_mk[i]) @ a_body[i]

kf_mk = KalmanFilter(dim_x=4, dim_z=2)
kf_mk.x = np.array([dN_ref[0], dE_ref[0], vN_ref[0], vE_ref[0]])
kf_mk.F = np.array([[1,0,dt,0], [0,1,0,dt], [0,0,1,0], [0,0,0,1]])
kf_mk.H = np.array([[1,0,0,0], [0,1,0,0]])
kf_mk.B = np.array([[0.5*dt**2, 0], [0, 0.5*dt**2], [dt, 0], [0, dt]])
kf_mk.P *= 10
kf_mk.R = np.diag([0.5**2, 0.5**2])
kf_mk.Q = np.diag([0.01, 0.01, 0.1, 0.1])

mk_states = []
mk_predictions = []
for i in range(n):
    u = a_nav_mk_input[i]
    z = np.array([dN_ref[i], dE_ref[i]])
    kf_mk.predict(u=u)
    mk_predictions.append(kf_mk.x.copy())
    kf_mk.update(z=z)
    mk_states.append(kf_mk.x.copy())

mk_states = np.array(mk_states)
mk_predictions = np.array(mk_predictions)
dN_mk, dE_mk, vN_mk, vE_mk = mk_states.T
aN_mk, aE_mk = np.gradient(vN_mk, dt), np.gradient(vE_mk, dt)


# 4. MÉTODO 2: EKF (PROCESSANDO DADOS REAIS)

ekf = ExtendedKalmanFilter(dim_x=5, dim_z=2)
ekf.x = np.array([dN_ref[0], dE_ref[0], vN_ref[0], vE_ref[0], yaw_ref[0]])
ekf.P = np.diag([0.5**2, 0.5**2, 1.0, 1.0, np.deg2rad(5)**2])
ekf.R = np.diag([0.5**2, 0.5**2])
q_pos, q_vel, q_yaw = 0.5, 1.0, np.deg2rad(1.0)
ekf.Q = np.diag([q_pos**2, q_pos**2, q_vel**2, q_vel**2, q_yaw**2])

def f_ekf(x, dt, u):
    yaw = x[4]; ax, ay = u[0], u[1]; gyro_z = u[2]
    c,s = np.cos(yaw), np.sin(yaw)
    aN, aE = c*ax - s*ay, s*ax + c*ay
    x_new = np.zeros(5)
    x_new[0] = x[0] + x[2]*dt + 0.5*aN*dt**2
    x_new[1] = x[1] + x[3]*dt + 0.5*aE*dt**2
    x_new[2] = x[2] + aN*dt
    x_new[3] = x[3] + aE*dt
    x_new[4] = x[4] + gyro_z*dt
    x_new[4] = np.arctan2(np.sin(x_new[4]), np.cos(x_new[4]))
    return x_new

def F_jac(x, dt, u):
    yaw = x[4]; ax, ay = u[0], u[1]
    c,s = np.cos(yaw), np.sin(yaw)
    daN_dyaw = -s*ax - c*ay; daE_dyaw = c*ax - s*ay
    F = np.eye(5)
    F[0,2]=dt; F[1,3]=dt
    F[0,4]=0.5*daN_dyaw*dt**2; F[1,4]=0.5*daE_dyaw*dt**2
    F[2,4]=daN_dyaw*dt; F[3,4]=daE_dyaw*dt
    return F

H_ekf = np.array([[1,0,0,0,0],[0,1,0,0,0]])
def hx_ekf(x): return H_ekf @ x

ekf_states = [ekf.x.copy()]
ekf_predictions = []
for i in range(n - 1):
    u_k = np.array([a_body[i,0], a_body[i,1], gyro_z[i]])
    ekf.F = F_jac(ekf.x, dt, u_k)
    ekf.predict()
    predicted_x = f_ekf(ekf.x, dt, u_k)
    ekf_predictions.append(predicted_x.copy())
    ekf.x = predicted_x
    z_k = np.array([dN_ref[i+1], dE_ref[i+1]])
    ekf.update(z_k, HJacobian=lambda x: H_ekf, Hx=hx_ekf)
    ekf_states.append(ekf.x.copy())

ekf_states = np.array(ekf_states)
ekf_predictions = np.array(ekf_predictions)
dN_ekf, dE_ekf, vN_ekf, vE_ekf, _ = ekf_states.T
aN_ekf = np.gradient(vN_ekf, dt)
aE_ekf = np.gradient(vE_ekf, dt)

# 5. ANÁLISE ESTATÍSTICA DAS INOVAÇÕES 

innov_mk_N = dN_ref - mk_predictions[:, 0]
innov_mk_E = dE_ref - mk_predictions[:, 1]
innov_mk_mag = np.sqrt(innov_mk_N**2 + innov_mk_E**2)
innov_ekf_N = dN_ref[1:] - ekf_predictions[:, 0]
innov_ekf_E = dE_ref[1:] - ekf_predictions[:, 1]
innov_ekf_mag = np.sqrt(innov_ekf_N**2 + innov_ekf_E**2)

mean_inn_mk_n = np.mean(innov_mk_N); std_inn_mk_n = np.std(innov_mk_N)
mean_inn_mk_e = np.mean(innov_mk_E); std_inn_mk_e = np.std(innov_mk_E)
rmse_inn_mk = np.sqrt(np.mean(innov_mk_mag**2))
mean_inn_ekf_n = np.mean(innov_ekf_N); std_inn_ekf_n = np.std(innov_ekf_N)
mean_inn_ekf_e = np.mean(innov_ekf_E); std_inn_ekf_e = np.std(innov_ekf_E)
rmse_inn_ekf = np.sqrt(np.mean(innov_ekf_mag**2))


# 6. PLOTS

tempo_eixo = np.arange(n) * dt

# Gráfico da Trajetória 
plt.figure(figsize=(8, 8))
plt.plot(dE_ref, dN_ref, 'k--', label='Referência', alpha=0.5)
plt.plot(dE_mk, dN_mk, 'b-', label='MK', linewidth=2)
plt.plot(dE_ekf, dN_ekf, 'r-', label='EKF', linewidth=2)
plt.xlabel('Leste [m]'); plt.ylabel('Norte [m]'); plt.title('Comparação Qualitativa das Trajetórias Estimadas')
plt.legend(); plt.grid(True); plt.axis('equal'); plt.show()

# Gráfico das Inovações 
plt.figure(figsize=(12, 6))
plt.plot(tempo_eixo, innov_mk_mag, 'b-', label='MK')
plt.plot(tempo_eixo[1:], innov_ekf_mag, 'r-', label='EKF')
plt.title('Análise de Inovações (Erro de Previsão da Posição)')
plt.xlabel('Tempo [s]'); plt.ylabel('Magnitude do Erro de Previsão [m]')
plt.legend(); plt.grid(True); plt.show()

# Gráfico Comparativo de Velocidade 
fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
fig.suptitle('Comparação das Estimativas de Velocidade', fontsize=16)
axs[0].plot(tempo_eixo, vN_ref, 'k--', label='Referência', alpha=0.7)
axs[0].plot(tempo_eixo, vN_mk, 'b-', label='MK')
axs[0].plot(tempo_eixo, vN_ekf, 'r-', label='EKF')
axs[0].set_ylabel('Velocidade Norte [m/s]'); axs[0].legend(); axs[0].grid(True)

axs[1].plot(tempo_eixo, vE_ref, 'k--', label='Referência', alpha=0.7)
axs[1].plot(tempo_eixo, vE_mk, 'b-', label='MK')
axs[1].plot(tempo_eixo, vE_ekf, 'r-', label='EKF')
axs[1].set_ylabel('Velocidade Leste [m/s]'); axs[1].set_xlabel('Tempo [s]'); axs[1].legend(); axs[1].grid(True)
plt.tight_layout(rect=[0, 0.03, 1, 0.96]); plt.show()

# Gráfico Comparativo de Aceleração 
fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
fig.suptitle('Comparação das Estimativas de Aceleração', fontsize=16)
axs[0].plot(tempo_eixo, aN_ref, 'k--', label='Referência', alpha=0.7)
axs[0].plot(tempo_eixo, aN_mk, 'b-', label='MK')
axs[0].plot(tempo_eixo, aN_ekf, 'r-', label='EKF')
axs[0].set_ylabel('Aceleração Norte [m/s²]'); axs[0].legend(); axs[0].grid(True)

axs[1].plot(tempo_eixo, aE_ref, 'k--', label='Referência', alpha=0.7)
axs[1].plot(tempo_eixo, aE_mk, 'b-', label='MK')
axs[1].plot(tempo_eixo, aE_ekf, 'r-', label='EKF')
axs[1].set_ylabel('Aceleração Leste [m/s²]'); axs[1].set_xlabel('Tempo [s]'); axs[1].legend(); axs[1].grid(True)
plt.tight_layout(rect=[0, 0.03, 1, 0.96]); plt.show()


# 7. TABELA LATEX

print("\n--- Tabela LaTeX (Análise Estatística das Inovações de Posição) ---")
print(f"""
\\begin{{table}}[h]
\\centering
\\small
\\caption{{Análise Estatística do Erro de Previsão (Inovações) de cada filtro}}
\\label{{tab:innovation_analysis}}
\\rowcolors{{2}}{{gray!10}}{{white}}
\\begin{{tabular}}{{lcc}}
\\hline
\\textbf{{Métrica de Inovação}} & \\textbf{{Filtro Híbrido (MK)}} & \\textbf{{Filtro EKF}} \\\\
\\hline
Média (Viés) - Eixo Norte [m] & {mean_inn_mk_n:+.4f} & {mean_inn_ekf_n:+.4f} \\\\
Média (Viés) - Eixo Leste [m] & {mean_inn_mk_e:+.4f} & {mean_inn_ekf_e:+.4f} \\\\
\\hline
Desvio Padrão - Eixo Norte [m] & {std_inn_mk_n:.4f} & {std_inn_ekf_n:.4f} \\\\
Desvio Padrão - Eixo Leste [m] & {std_inn_mk_e:.4f} & {std_inn_ekf_e:.4f} \\\\
\\hline
\\textbf{{RMSE (Geral) [m]}} & \\textbf{{{rmse_inn_mk:.4f}}} & \\textbf{{{rmse_inn_ekf:.4f}}} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
""")