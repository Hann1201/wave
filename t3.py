import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

st.title("平面波传播仿真平台")

# ==================== 侧边栏：参数控制 ====================
with st.sidebar:
    st.header("仿真参数")

    # 波源参数
    wavelength = st.slider("真空波长 λ₀ (m)", 0.5, 5.0, 2.0, 0.1)
    amplitude = st.slider("电磁波振幅", 0.5, 3.0, 1.0, 0.1)

    # 介质参数
    epsilon_r = st.slider("相对介电常数 εᵣ", 1.0, 10.0, 4.0, 0.1)
    mu_r = st.slider("相对磁导率 μᵣ", 0.5, 5.0, 1.0, 0.1)

    # 电导率参数（用于计算衰减常数）
    sigma = st.slider("电导率 σ (S/m)", 0.0, 5.0, 0.0, 0.01)

    # 分界面位置
    interface_pos = st.slider("分界面位置", 2.0, 8.0, 5.0, 0.1)

    # 显示控制
    show_boundary = st.checkbox("显示真空/介质边界", value=True)
    show_grid = st.checkbox("显示网格线", value=True)

    refresh_btn = st.button("刷新图像", type="primary")


# ==================== 平面波计算函数 ====================
def calculate_plane_wave(wavelength, amplitude, epsilon_r, mu_r, sigma, interface_pos, x_range=(0, 10),
                         num_points=1000):
    """计算平面波在真空-介质分界面的传播"""

    # 物理常数
    c = 3e8  # 光速
    omega = 2 * np.pi * c / wavelength  # 角频率
    epsilon0 = 8.854e-12  # 真空介电常数
    mu0 = 4 * np.pi * 1e-7  # 真空磁导率

    # 真空区域参数
    k0 = 2 * np.pi / wavelength  # 真空波数
    eta0 = np.sqrt(mu0 / epsilon0)  # 真空本征阻抗 ≈ 377Ω

    # 介质区域参数
    epsilon = epsilon_r * epsilon0  # 介质介电常数
    mu = mu_r * mu0  # 介质磁导率

    # 传播常数 γ = α + jβ
    # γ = jω√(με) * √(1 - jσ/(ωε))
    omega_epsilon = omega * epsilon
    loss_tan = sigma / omega_epsilon  # 损耗角正切

    # 计算传播常数（有损耗介质）
    temp = np.sqrt(1 - 1j * sigma / (omega * epsilon))
    gamma = 1j * omega * np.sqrt(mu * epsilon) * temp

    alpha = np.real(gamma)  # 衰减常数 (Np/m)
    beta = np.imag(gamma)  # 相位常数 (rad/m)

    # 介质中波长
    lambda_medium = 2 * np.pi / beta if beta > 0 else np.inf

    # 相速
    vp = omega / beta if beta > 0 else np.inf

    # 本征阻抗
    eta_c = np.sqrt(1j * omega * mu / (sigma + 1j * omega * epsilon))
    eta_magnitude = np.abs(eta_c)
    eta_phase = np.angle(eta_c) * 180 / np.pi

    # 折射率
    n = np.sqrt(epsilon_r * mu_r)

    # 真空波数（用于传播）
    k1 = beta  # 使用相位常数作为波数

    # 生成空间坐标
    x = np.linspace(x_range[0], x_range[1], num_points)

    # 计算电场分布（稳态，固定时刻 t=0）
    E = np.zeros_like(x)

    # 透射系数（用于匹配边界）
    eta1 = eta_c
    Gamma = (eta1 - eta0) / (eta1 + eta0)  # 反射系数
    T = 2 * eta1 / (eta1 + eta0)  # 透射系数

    for i, xi in enumerate(x):
        if xi < interface_pos:
            # 真空区域：入射波 + 反射波（忽略反射，只显示入射波）
            E_incident = amplitude * np.cos(k0 * xi)
            # 如果有损耗，添加衰减项
            E[i] = E_incident
        else:
            # 介质区域：透射波（考虑衰减）
            if alpha > 0:
                E[i] = amplitude * T * np.exp(-alpha * (xi - interface_pos)) * np.cos(beta * (xi - interface_pos))
            else:
                E[i] = amplitude * T * np.cos(beta * (xi - interface_pos))

    return x, E, k0, beta, alpha, lambda_medium, vp, eta_magnitude, eta_phase, n, epsilon_r, mu_r, sigma


def create_2d_wave_plot(x, E, interface_pos, amplitude, show_boundary, show_grid, epsilon_r, mu_r, n, lambda_medium,
                        sigma):
    """创建二维波形图"""
    fig, ax = plt.subplots(figsize=(10, 5))

    # 绘制波形
    ax.plot(x, E, 'b-', linewidth=2, label='电场强度 E')

    # 标记分界面
    if show_boundary:
        ax.axvline(x=interface_pos, color='red', linewidth=2, linestyle='--', label='真空/介质边界')

    # 添加区域标注
    ax.text(interface_pos / 2, amplitude * 1.3, '真空区', ha='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    loss_text = f" (σ={sigma} S/m)" if sigma > 0 else ""
    ax.text(interface_pos + (10 - interface_pos) / 2, amplitude * 1.3,
            f'介质区 (εᵣ={epsilon_r}, μᵣ={mu_r}){loss_text}', ha='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # 设置坐标轴
    ax.set_xlabel('传播方向 x', fontsize=12)
    ax.set_ylabel('电场强度 E', fontsize=12)
    ax.set_title('电磁波二维传播波形', fontsize=14, fontweight='bold')

    if show_grid:
        ax.grid(True, alpha=0.3)

    ax.legend(loc='upper right')
    ax.set_xlim(0, 10)
    ax.set_ylim(-amplitude * 1.5, amplitude * 1.5)

    plt.tight_layout()
    return fig


def create_3d_wireframe(x, E, interface_pos, amplitude):
    """创建三维线条网格图"""
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # 创建网格
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x[::10], y)  # 降采样以优化显示

    # 扩展电场到二维平面
    E_2d = np.tile(E[::10], (len(y), 1))

    # 绘制三维线条网格
    for i in range(0, len(y), 3):
        ax.plot(x[::10], E_2d[i, :] * 0.5 + y[i], zs=y[i], zdir='y',
                color='steelblue', alpha=0.7, linewidth=1)

    for i in range(0, len(x[::10]), 5):
        ax.plot([x[::10][i]] * len(y), E_2d[:, i] * 0.5 + y, zs=y, zdir='y',
                color='steelblue', alpha=0.5, linewidth=0.8)

    ax.set_xlabel('传播方向 X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.set_zlabel('电场 E', fontsize=10)
    ax.set_title('电磁波三维线条网格图像', fontsize=14, fontweight='bold')

    # 设置视角
    ax.view_init(elev=25, azim=45)

    plt.tight_layout()
    return fig


# ==================== 主界面：结果展示 ====================
if refresh_btn:
    with st.spinner("计算中..."):
        # 计算平面波
        x, E, k0, beta, alpha, lambda_medium, vp, eta_magnitude, eta_phase, n, epsilon_r, mu_r, sigma = calculate_plane_wave(
            wavelength, amplitude, epsilon_r, mu_r, sigma, interface_pos
        )

    # 显示电磁波参数
    st.subheader("电磁波传播参数")

    # 第一行参数
    col_param1, col_param2, col_param3, col_param4 = st.columns(4)
    with col_param1:
        st.metric("相位常数 β", f"{beta:.4f} rad/m")
    with col_param2:
        st.metric("衰减常数 α", f"{alpha:.6f} Np/m" if alpha > 0 else "0 (无损耗)")
    with col_param3:
        st.metric("传播常数 γ", f"{alpha:.4f} + j{beta:.4f}")
    with col_param4:
        st.metric("本征阻抗 |η|", f"{eta_magnitude:.2f} Ω")

    # 第二行参数
    col_param5, col_param6, col_param7, col_param8 = st.columns(4)
    with col_param5:
        st.metric("本征阻抗相位", f"{eta_phase:.2f}°")
    with col_param6:
        st.metric("相速 vₚ", f"{vp / 1e8:.4f}×10⁸ m/s")
    with col_param7:
        st.metric("介质中波长 λ", f"{lambda_medium:.4f} m")
    with col_param8:
        st.metric("折射率 n", f"{n:.4f}")

    # 如果有损耗，显示额外信息
    if sigma > 0:
        st.info(f"⚠️ 介质有损耗 (σ={sigma} S/m)，电磁波在传播过程中会衰减")

    # 创建两列布局显示图像
    col_left, col_right = st.columns(2)

    with col_left:
        # 二维波形图
        fig_2d = create_2d_wave_plot(x, E, interface_pos, amplitude,
                                     show_boundary, show_grid,
                                     epsilon_r, mu_r, n, lambda_medium, sigma)
        st.pyplot(fig_2d)
        plt.close(fig_2d)

    with col_right:
        # 三维线条网格图
        fig_3d = create_3d_wireframe(x, E, interface_pos, amplitude)
        st.pyplot(fig_3d)
        plt.close(fig_3d)

    # 参数详细说明
    with st.expander("查看详细参数说明"):
        st.write(f"""
        ### 电磁波传播参数详解

        #### 基本参数
        - **真空波长 λ₀**: {wavelength} m
        - **真空中波数 k₀**: {k0:.4f} rad/m
        - **角频率 ω**: {2 * np.pi * 3e8 / wavelength:.3e} rad/s

        #### 介质参数
        - **相对介电常数 εᵣ**: {epsilon_r}
        - **相对磁导率 μᵣ**: {mu_r}
        - **电导率 σ**: {sigma} S/m

        #### 传播常数 (γ = α + jβ)
        - **衰减常数 α**: {alpha:.6f} Np/m (表示波幅衰减的快慢)
        - **相位常数 β**: {beta:.4f} rad/m (表示相位变化的快慢)
        - **传播常数**: {alpha:.4f} + j{beta:.4f}

        #### 本征阻抗
        - **阻抗模值 |η|**: {eta_magnitude:.2f} Ω
        - **阻抗相位 θ**: {eta_phase:.2f}° (电场滞后于磁场的角度)

        #### 传播特性
        - **相速 vₚ**: {vp:.3e} m/s = ω/β
        - **介质中波长 λ**: {lambda_medium:.4f} m = 2π/β
        - **折射率 n**: {n:.4f} = √(εᵣμᵣ)

        #### 损耗特性
        - **损耗角正切 tanδ**: {sigma / (2 * np.pi * 3e8 / wavelength * 8.854e-12 * epsilon_r):.6f}
        - **趋肤深度 δ**: {1 / alpha:.4f} m (若α>0)

        > 注：当 σ > 0 时，电磁波在介质中会有衰减，振幅按 e^(-αx) 规律减小
        """)

    # 结果导出
    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:
        # 导出数据
        data_dict = {
            'x': x.tolist(),
            'E': E.tolist(),
            'wavelength': wavelength,
            'epsilon_r': epsilon_r,
            'mu_r': mu_r,
            'sigma': sigma,
            'interface_pos': interface_pos,
            'beta': beta,
            'alpha': alpha,
            'vp': vp,
            'eta_magnitude': eta_magnitude,
            'n': n
        }
        import pickle

        buf_data = io.BytesIO()
        pickle.dump(data_dict, buf_data)
        buf_data.seek(0)
        st.download_button(
            label="下载仿真数据",
            data=buf_data,
            file_name=f"plane_wave_e{epsilon_r}_u{mu_r}_s{sigma}.pkl",
            mime="application/octet-stream"
        )

    with col_dl2:
        # 导出图像
        fig_export, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # 复制2D图
        ax1.plot(x, E, 'b-', linewidth=2)
        ax1.axvline(x=interface_pos, color='red', linewidth=2, linestyle='--')
        ax1.set_xlabel('传播方向 x')
        ax1.set_ylabel('电场强度 E')
        ax1.set_title('电磁波二维传播波形')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 10)

        # 简单的3D示意
        ax2.remove()
        ax2 = fig_export.add_subplot(122, projection='3d')
        y_simple = np.linspace(-1, 1, 20)
        X_s, Y_s = np.meshgrid(x[::20], y_simple)
        E_s = np.tile(E[::20], (len(y_simple), 1))
        ax2.plot_surface(X_s, Y_s, E_s * 0.3, cmap='coolwarm', alpha=0.8)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('E')
        ax2.set_title('电磁波三维分布')

        plt.tight_layout()
        buf_img = io.BytesIO()
        fig_export.savefig(buf_img, format='png', dpi=150, bbox_inches='tight')
        buf_img.seek(0)
        st.download_button(
            label="下载结果图像",
            data=buf_img,
            file_name=f"plane_wave_e{epsilon_r}_u{mu_r}_s{sigma}.png",
            mime="image/png"
        )
        plt.close(fig_export)

else:
    # 初始状态提示
    st.info("👈 请在侧边栏设置参数，点击「刷新图像」查看仿真结果")

    # 显示默认示意图
    col_demo1, col_demo2 = st.columns(2)
    with col_demo1:
        st.markdown("""
        **二维波形图预览**

        显示平面波在真空-介质分界面的传播：
        - 左侧：真空区，波长较长
        - 右侧：介质区，波长缩短、振幅变化
        - 红色虚线：分界面位置
        """)
    with col_demo2:
        st.markdown("""
        **三维线条网格图预览**

        显示电磁波的三维空间分布：
        - X轴：传播方向
        - Y轴：横向扩展
        - Z轴：电场强度
        """)