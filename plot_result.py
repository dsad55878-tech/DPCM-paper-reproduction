import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_power_curve():
    # 1. 读取数据
    csv_file = 'simulation_results.csv'
    if not os.path.exists(csv_file):
        print(f"错误：找不到 {csv_file}。")
        print("请先运行 main.py 生成数据！")
        return

    try:
        df = pd.read_csv(csv_file)
        print("成功读取数据，准备绘图...")
    except Exception as e:
        print(f"读取 CSV 出错: {e}")
        return

    # 2. 筛选数据 (为了复现论文 Figure 3 的条件)
    # 论文中 Figure 3 是 N=500, Reliability (alpha)=0.80
    target_N = 500
    target_alpha = 0.80
    
    subset = df[(df['N'] == target_N) & (df['Reliability'] == target_alpha)]
    
    if subset.empty:
        print(f"警告：结果中没有找到 N={target_N} 且 Alpha={target_alpha} 的数据。")
        print("正在绘制所有数据的平均值作为替代...")
        subset = df # 如果找不到特定条件，就画整体趋势
        title_text = 'Average Power across all conditions'
    else:
        title_text = f'Replication of Figure 3 (N={target_N}, α={target_alpha})'

    # 3. 绘图设置
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # 绘制折线图
    # x轴: Delta_t (采样间隔), y轴: Statistical_Power (统计效力)
    sns.lineplot(x='Delta_t', y='Statistical_Power', data=subset, 
                 marker='o', linewidth=3, markersize=10)
    
    # 4. 设置标签和标题
    plt.title(title_text, fontsize=16, fontweight='bold')
    plt.xlabel('Sampling Interval Δt (Weeks)', fontsize=14)
    plt.ylabel('Statistical Power (%)', fontsize=14)
    plt.ylim(0, 105) # y轴范围 0到100%
    plt.xticks([1, 2, 5], ['1 (Weekly)', '2 (Biweekly)', '5 (Pre-Post)']) # 设置x轴刻度标签
    
    # 5. 添加文字标注 (显示下降幅度)
    # 获取起点和终点的值
    try:
        power_at_1 = subset[subset['Delta_t'] == 1]['Statistical_Power'].mean()
        power_at_5 = subset[subset['Delta_t'] == 5]['Statistical_Power'].mean()
        drop = power_at_1 - power_at_5
        
        # 在图中写字
        mid_x = 3
        mid_y = (power_at_1 + power_at_5) / 2
        plt.text(mid_x, mid_y + 5, f"Drop: -{drop:.1f}%", 
                 color='red', fontsize=14, fontweight='bold', ha='center')
        
        # 画一条红色的虚线连接首尾
        plt.plot([1, 5], [power_at_1, power_at_5], 'r--', alpha=0.5)
        
        print(f"验证成功：Sampling Interval 从 1 变到 5 时，Power 下降了 {drop:.1f}%")
        
    except Exception as e:
        print("无法计算具体下降数值，仅绘图。")

    # 6. 保存图片
    output_file = 'Figure_Replication.png'
    plt.savefig(output_file, dpi=300)
    print(f"图表已保存为: {output_file}")
    print("请打开这张图片，确认曲线是【从左上角往右下角】下降的。")

if __name__ == "__main__":
    plot_power_curve()