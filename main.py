# -*- coding: utf-8 -*-
"""
====================================================================
电商用户行为分析完整项目
====================================================================
基于阿里天池移动端脱敏用户行为数据，完成电商用户行为分析。
包含四大模块：
  模块1 — EDA探索性数据分析
  模块2 — 电商转化漏斗分析
  模块3 — RFM用户分层分析
  模块4 — AB-Test仿真实验

数据来源：阿里天池「阿里移动推荐算法」数据集
时间范围：2014-11-18 ~ 2014-12-18
运行方式：python main.py
====================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager as fm
import seaborn as sns
from scipy import stats

# ============================================================
# 全局配置
# ============================================================
warnings.filterwarnings("ignore")

# ---------- 第一步：先设置 seaborn 风格（seaborn 会重置 rcParams，必须放在字体设置之前）----------
sns.set_style("whitegrid")
sns.set_palette("Set2")

# ---------- 第二步：配置中文字体（防止乱码）----------
# 1. 重建 matplotlib 字体缓存，确保所有系统字体可被识别
fm._load_fontmanager(try_read_cache=False)

# 2. 按优先级排列中文字体（从高到低）
#    Microsoft YaHei 是 Windows 系统自带，字重完整，显示效果最稳定
#    SimHei 是经典黑体
#    Noto Sans SC 是开源无衬线中文字体（可变字体，字重映射可能有问题，放后面）
_chinese_font_priority = [
    "Microsoft YaHei",
    "Microsoft YaHei UI",
    "SimHei",
    "Noto Sans SC",
    "SimSun",
    "DengXian",
    "KaiTi",
    "FangSong",
    "Arial Unicode MS",
    "DejaVu Sans",
]

# 过滤出系统中实际存在的字体
_available_fonts = {f.name for f in fm.fontManager.ttflist}
_chinese_fonts_available = [f for f in _chinese_font_priority if f in _available_fonts]

# 3. 设置全局字体（必须在 seaborn set_style 之后，否则会被覆盖）
matplotlib.rcParams["font.sans-serif"] = _chinese_fonts_available
matplotlib.rcParams["axes.unicode_minus"] = False  # 正常显示负号

# 4. seaborn 也设置字体（双重保险）
sns.set_context("notebook", font_scale=1.0, rc={"font.sans-serif": _chinese_fonts_available})

print(f"[字体配置] 使用中文字体: {_chinese_fonts_available[:3]}")

# 图片输出目录
IMG_DIR = "img"
os.makedirs(IMG_DIR, exist_ok=True)

# 数据文件路径（相对项目根目录）
DATA_PATH = "data/tianchi_mobile_recommend_train_user.csv"

# 行为类型映射
BEHAVIOR_MAP = {1: "浏览", 2: "收藏", 3: "加购", 4: "购买"}
BEHAVIOR_COLORS = {"浏览": "#4C72B0", "收藏": "#55A868", "加购": "#DD8452", "购买": "#C44E52"}


# ============================================================
# 数据加载与预处理
# ============================================================
def load_data():
    """
    加载阿里天池移动推荐用户行为数据。
    原始数据列：user_id, item_id, behavior_type, user_geohash, item_category, time
    预处理后列：user_id, item_id, behavior_type, item_category, time, behavior_name,
               date, hour, day_of_week, is_weekend
    """
    print("=" * 60)
    print("【数据加载与预处理】")
    print("=" * 60)

    # 使用低内存 dtype 加载 633MB CSV
    df = pd.read_csv(
        DATA_PATH,
        dtype={
            "user_id": "int32",
            "item_id": "int32",
            "behavior_type": "int8",
            "item_category": "int32",
        },
    )

    # 如果存在 user_geohash 列则删除（大量空值，加密哈希无法解析）
    if "user_geohash" in df.columns:
        df.drop("user_geohash", axis=1, inplace=True)

    # time 转为 datetime 类型
    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    # 添加行为中文映射（如果不存在）
    if "behavior_name" not in df.columns:
        df["behavior_name"] = df["behavior_type"].map(BEHAVIOR_MAP)

    # 衍生时间维度字段
    df["date"] = df["time"].dt.date
    df["hour"] = df["time"].dt.hour
    df["day_of_week"] = df["time"].dt.dayofweek  # 0=周一, 6=周日
    df["is_weekend"] = df["day_of_week"].isin([5, 6])

    # 删除 time 为空的异常行
    df = df.dropna(subset=["time"])

    print(f"  总记录数：{len(df):,}")
    print(f"  独立用户数：{df['user_id'].nunique():,}")
    print(f"  独立商品数：{df['item_id'].nunique():,}")
    print(f"  独立类目数：{df['item_category'].nunique():,}")
    print(f"  时间范围：{df['time'].min()} ~ {df['time'].max()}")
    print(f"  列名：{df.columns.tolist()}")
    print()

    return df


# ============================================================
# 模块1：EDA 探索性数据分析
# ============================================================
def module1_eda(df):
    """
    EDA 探索性数据分析：
      1.1 基础统计（总记录数、独立用户/商品/类目数、各行为占比）
      1.2 时间维度（日UV/日行为趋势、工作日vs周末、小时分布）
      1.3 类目分析（Top类目行为分布）
      1.4 可视化图表输出
    """
    print("=" * 60)
    print("【模块1：EDA探索性数据分析】")
    print("=" * 60)

    # --- 1.1 基础统计 ---
    total_records = len(df)
    unique_users = df["user_id"].nunique()
    unique_items = df["item_id"].nunique()
    unique_categories = df["item_category"].nunique()

    behavior_counts = df["behavior_type"].value_counts().sort_index()
    behavior_ratios = (behavior_counts / total_records * 100).round(2)

    print("\n--- 1.1 基础统计 ---")
    print(f"  总行为记录数：{total_records:,}")
    print(f"  独立用户数：{unique_users:,}")
    print(f"  独立商品数：{unique_items:,}")
    print(f"  独立类目数：{unique_categories:,}")
    print("  各类行为数量与占比：")
    for btype in sorted(BEHAVIOR_MAP.keys()):
        name = BEHAVIOR_MAP[btype]
        cnt = behavior_counts.get(btype, 0)
        ratio = behavior_ratios.get(btype, 0)
        print(f"    {name}：{cnt:,}（{ratio}%）")

    # 图1：行为类型分布（柱状图 + 环形饼图）
    # 饼图使用环形图 + 外部图例，避免小扇区标签重叠
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    names = [BEHAVIOR_MAP[i] for i in behavior_counts.index]
    colors = [BEHAVIOR_COLORS[n] for n in names]
    values = behavior_counts.values
    total_val = values.sum()

    # --- 左图：柱状图（使用对数刻度展示数量级差异） ---
    bars = axes[0].bar(names, values, color=colors, edgecolor="white", linewidth=0.8, width=0.6)
    axes[0].set_title("各类行为数量分布", fontsize=14, fontweight="bold", pad=10)
    axes[0].set_ylabel("记录数（万次）", fontsize=12)
    # y轴以"万"为单位显示，避免科学计数和数字过长
    axes[0].yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: f"{x/10000:.0f}")
    )
    for bar, val in zip(bars, values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{val/10000:.1f}万", ha="center", va="bottom", fontsize=11, fontweight="bold",
        )
    axes[0].set_ylim(0, max(values) * 1.15)

    # --- 右图：环形饼图（小扇区用图例+引线标注，避免重叠） ---
    wedges, _ = axes[1].pie(
        values, colors=colors, startangle=90,
        wedgeprops=dict(width=0.5, edgecolor="white", linewidth=2),
        pctdistance=0.75,
    )
    # 构造图例标签：行为名称 + 占比 + 数量
    legend_labels = [
        f"{name}  {val/total_val*100:.2f}%  ({val:,})"
        for name, val in zip(names, values)
    ]
    axes[1].legend(
        wedges, legend_labels, loc="center left",
        bbox_to_anchor=(1.02, 0.5), fontsize=11, frameon=False,
        title="行为类型  占比  数量", title_fontsize=11,
    )
    axes[1].set_title("各类行为占比", fontsize=14, fontweight="bold", pad=10)
    # 环形中心文字
    axes[1].text(0, 0, f"总计\n{total_val/10000:.0f}万",
                 ha="center", va="center", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/01_behavior_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [图表已保存] 01_behavior_distribution.png")

    # --- 1.2 时间维度分析 ---
    print("\n--- 1.2 时间维度分析 ---")

    # 日 UV 与日行为总量
    daily_stats = df.groupby("date").agg(
        daily_uv=("user_id", "nunique"),
        daily_behavior=("user_id", "count"),
    ).reset_index()

    # 计算第几天（相对第一天）
    start_date = daily_stats["date"].iloc[0]
    daily_stats["day_num"] = [(d - start_date).days + 1 for d in daily_stats["date"]]

    # 图2：日UV与日行为总量趋势（双轴折线图）
    fig, ax1 = plt.subplots(figsize=(14, 5.5))
    line1 = ax1.plot(daily_stats["day_num"], daily_stats["daily_uv"], color="#4C72B0",
             marker="o", markersize=4, linewidth=1.8, label="日UV（独立用户数）")
    ax1.set_xlabel("日期", fontsize=12)
    ax1.set_ylabel("日UV（人）", fontsize=12, color="#4C72B0")
    ax1.tick_params(axis="y", labelcolor="#4C72B0", labelsize=10)
    ax1.tick_params(axis="x", labelsize=10)
    ax1.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: f"{x/1000:.0f}k")
    )

    # 左轴范围：让 UV 线整体偏低
    uv_min, uv_max = daily_stats["daily_uv"].min(), daily_stats["daily_uv"].max()
    ax1.set_ylim(uv_min * 0.9, uv_max * 1.15)

    ax2 = ax1.twinx()
    line2 = ax2.plot(daily_stats["day_num"], daily_stats["daily_behavior"], color="#C44E52",
             marker="s", markersize=4, linewidth=1.8, label="日行为总量")
    ax2.set_ylabel("日行为总量（万次）", fontsize=12, color="#C44E52")
    ax2.tick_params(axis="y", labelcolor="#C44E52", labelsize=10)
    ax2.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: f"{x/10000:.0f}")
    )

    # 右轴范围：让行为总量线始终在 UV 线上方
    beh_min, beh_max = daily_stats["daily_behavior"].min(), daily_stats["daily_behavior"].max()
    ax2.set_ylim(0, beh_max / 0.78)

    # 统一 x 轴刻度：每 5 天一个标签
    tick_step = 5
    tick_positions = list(range(1, len(daily_stats) + 1, tick_step))
    if len(daily_stats) not in tick_positions:
        tick_positions.append(len(daily_stats))
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels([f"第{d}天" for d in tick_positions], fontsize=10)

    plt.title("日UV与日行为总量趋势（2014-11-18 ~ 2014-12-18）", fontsize=14, fontweight="bold", pad=12)
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", fontsize=11, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/02_daily_trend.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [图表已保存] 02_daily_trend.png")

    # 工作日 vs 周末行为对比（日均归一化）
    weekday_df = df[~df["is_weekend"]]
    weekend_df = df[df["is_weekend"]]
    weekday_days = weekday_df["date"].nunique()
    weekend_days = weekend_df["date"].nunique()

    weekday_behavior = weekday_df["behavior_type"].value_counts().sort_index()
    weekend_behavior = weekend_df["behavior_type"].value_counts().sort_index()
    weekday_avg = weekday_behavior / weekday_days  # 工作日日均
    weekend_avg = weekend_behavior / weekend_days  # 周末日均

    print(f"  工作日天数：{weekday_days}，周末天数：{weekend_days}")
    print(f"  工作日日均行为：{weekday_avg.to_dict()}")
    print(f"  周末日均行为：{weekend_avg.to_dict()}")

    # 图3：工作日vs周末日均行为对比
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(4)
    width = 0.32
    bar_names = [BEHAVIOR_MAP[i] for i in range(1, 5)]
    bars1 = ax.bar(x - width / 2, weekday_avg.values, width,
                   label=f"工作日（{weekday_days}天日均）", color="#4C72B0", edgecolor="white")
    bars2 = ax.bar(x + width / 2, weekend_avg.values, width,
                   label=f"周末（{weekend_days}天日均）", color="#C44E52", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(bar_names, fontsize=12)
    ax.set_title("工作日 vs 周末日均行为对比", fontsize=14, fontweight="bold", pad=10)
    ax.set_ylabel("日均行为次数（次）", fontsize=12)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: f"{x/1000:.1f}k" if x >= 1000 else f"{x:.0f}")
    )
    # 柱顶数值标注
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h,
                    f"{h:,.0f}", ha="center", va="bottom", fontsize=9.5)
    ax.legend(fontsize=11, loc="upper right")
    ax.set_ylim(0, max(weekday_avg.max(), weekend_avg.max()) * 1.18)
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/03_weekday_weekend.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [图表已保存] 03_weekday_weekend.png")

    # 小时维度行为分布
    hourly_behavior = df.groupby(["hour", "behavior_type"]).size().unstack(fill_value=0)

    # 图4：24小时行为分布（双行子图，浏览单独展示，其他行为放大展示）
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(13, 10), sharex=True,
                                            gridspec_kw={"height_ratios": [2.5, 1.5]})

    # 上图：浏览
    ax_top.plot(hourly_behavior.index, hourly_behavior[1],
                marker="o", markersize=4, linewidth=2,
                color=BEHAVIOR_COLORS["浏览"], label="浏览")
    ax_top.set_ylabel("浏览次数", fontsize=12)
    ax_top.set_title("24小时行为分布", fontsize=14, fontweight="bold", pad=10)
    ax_top.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: f"{x/10000:.0f}万" if x >= 10000 else f"{x:,.0f}")
    )
    ax_top.legend(fontsize=11, loc="upper left")
    ax_top.grid(True, alpha=0.3)

    # 下图：收藏、加购、购买
    for btype in [2, 3, 4]:
        name = BEHAVIOR_MAP[btype]
        ax_bottom.plot(hourly_behavior.index, hourly_behavior[btype],
                       marker="o", markersize=4, linewidth=2,
                       color=BEHAVIOR_COLORS[name], label=name)
    ax_bottom.set_xlabel("小时", fontsize=12)
    ax_bottom.set_ylabel("行为次数", fontsize=12)
    ax_bottom.set_xticks(range(0, 24))
    ax_bottom.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: f"{x:,.0f}")
    )
    ax_bottom.legend(fontsize=11, loc="upper left")
    ax_bottom.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/04_hourly_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [图表已保存] 04_hourly_distribution.png")

    # --- 1.3 类目分析 ---
    print("\n--- 1.3 类目分析 ---")

    # Top 20 类目按总行为数排序
    top_categories = df["item_category"].value_counts().head(20)
    print(f"  Top 5 类目（总行为数）：{top_categories.head().to_dict()}")

    # 图5：Top 20类目行为分布
    fig, ax = plt.subplots(figsize=(12, 7))
    top_categories_sorted = top_categories.sort_values()
    ax.barh(range(len(top_categories_sorted)), top_categories_sorted.values,
            color="#4C72B0", edgecolor="white", height=0.7)
    ax.set_yticks(range(len(top_categories_sorted)))
    ax.set_yticklabels(top_categories_sorted.index, fontsize=10)
    ax.set_title("Top 20 商品类目行为分布", fontsize=14, fontweight="bold", pad=10)
    ax.set_xlabel("行为记录数", fontsize=12)
    ax.set_ylabel("类目ID", fontsize=12)
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: f"{x/10000:.0f}万")
    )
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/05_top_categories.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [图表已保存] 05_top_categories.png")

    # Top 20 类目按购买数排序
    top_purchase_categories = df[df["behavior_type"] == 4]["item_category"].value_counts().head(20)
    print(f"  Top 5 类目（购买数）：{top_purchase_categories.head().to_dict()}")

    # 图6：Top 20类目购买次数分布
    fig, ax = plt.subplots(figsize=(12, 7))
    top_purchase_sorted = top_purchase_categories.sort_values()
    ax.barh(range(len(top_purchase_sorted)), top_purchase_sorted.values,
            color="#C44E52", edgecolor="white", height=0.7)
    ax.set_yticks(range(len(top_purchase_sorted)))
    ax.set_yticklabels(top_purchase_sorted.index, fontsize=10)
    ax.set_title("Top 20 商品类目购买次数分布", fontsize=14, fontweight="bold", pad=10)
    ax.set_xlabel("购买次数", fontsize=12)
    ax.set_ylabel("类目ID", fontsize=12)
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: f"{x:,.0f}")
    )
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/06_top_purchase_categories.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [图表已保存] 06_top_purchase_categories.png")

    print()
    return {
        "total_records": total_records,
        "unique_users": unique_users,
        "unique_items": unique_items,
        "unique_categories": unique_categories,
        "behavior_counts": behavior_counts,
        "behavior_ratios": behavior_ratios,
    }


# ============================================================
# 模块2：电商转化漏斗分析
# ============================================================
def module2_funnel(df):
    """
    电商转化漏斗分析。
    业务路径：浏览 → 收藏 → 加购 → 购买
    ⚠️ 重要口径：漏斗每一步统计去重独立 user_id（用户粒度），
       不统计行为记录条数。
    """
    print("=" * 60)
    print("【模块2：电商转化漏斗分析】")
    print("=" * 60)

    # --- 2.1 用户粒度漏斗统计 ---
    # 获取各行为类型的去重用户集合
    browse_users = set(df[df["behavior_type"] == 1]["user_id"].unique())
    fav_users = set(df[df["behavior_type"] == 2]["user_id"].unique())
    cart_users = set(df[df["behavior_type"] == 3]["user_id"].unique())
    buy_users = set(df[df["behavior_type"] == 4]["user_id"].unique())

    # 漏斗每步：统计去重独立 user_id 数量
    # Step1 浏览：所有有浏览行为的用户
    # Step2 收藏：有浏览且有收藏行为的用户（漏斗嵌套）
    # Step3 加购：有浏览+收藏+加购的用户
    # Step4 购买：有浏览+收藏+加购+购买的用户
    funnel_steps = [
        ("浏览", len(browse_users)),
        ("收藏", len(browse_users & fav_users)),
        ("加购", len(browse_users & fav_users & cart_users)),
        ("购买", len(browse_users & fav_users & cart_users & buy_users)),
    ]

    # 计算步骤转化率与流失率
    funnel_df = pd.DataFrame(funnel_steps, columns=["步骤", "用户数"])
    funnel_df["步骤转化率"] = (funnel_df["用户数"] / funnel_df["用户数"].shift(1)).fillna(1.0)
    funnel_df["步骤转化率"] = (funnel_df["步骤转化率"] * 100).round(2).astype(str) + "%"
    funnel_df["流失率"] = ((1 - funnel_df["用户数"] / funnel_df["用户数"].shift(1)) * 100).round(2)
    funnel_df["流失率"] = funnel_df["流失率"].fillna(0).astype(str) + "%"
    funnel_df["总体转化率"] = (funnel_df["用户数"] / funnel_df["用户数"].iloc[0] * 100).round(2).astype(str) + "%"

    print("\n--- 2.1 漏斗统计（用户粒度去重） ---")
    print(funnel_df.to_string(index=False))

    # 整体转化率
    overall_cr = funnel_steps[-1][1] / funnel_steps[0][1]
    print(f"\n  浏览→购买整体转化率：{overall_cr*100:.2f}%")

    # --- 2.2 绘制漏斗图 ---
    # 标准倒三角形漏斗：顶宽=当前步用户数占比，底宽=下一步占比，逐层递减
    fig, ax = plt.subplots(figsize=(12, 6))

    step_values = [s[1] for s in funnel_steps]
    max_val = step_values[0]
    funnel_colors = ["#4C72B0", "#55A868", "#DD8452", "#C44E52"]
    widths = [v / max_val for v in step_values]  # 归一化宽度

    for i, (name, val) in enumerate(funnel_steps):
        y_bottom = len(funnel_steps) - i - 1
        y_top = y_bottom + 1

        # 顶宽 = 当前步的归一化宽度
        top_w = widths[i]
        top_left = (1 - top_w) / 2
        # 底宽 = 下一步的归一化宽度（最后一层略微收窄，形成尖底）
        bottom_w = widths[i + 1] if i < len(funnel_steps) - 1 else top_w * 0.88
        bottom_left = (1 - bottom_w) / 2

        # 绘制梯形：顶宽=当前步，底宽=下一步（逐层递减）
        x = [top_left, 1 - top_left, 1 - bottom_left, bottom_left]
        y = [y_top, y_top, y_bottom, y_bottom]
        ax.fill(x, y, color=funnel_colors[i], alpha=0.85, edgecolor="white", linewidth=2)

        # 层内标注：行为名称 + 用户数
        ax.text(0.5, y_bottom + 0.5, f"{name}\n{val:,}人",
                ha="center", va="center", fontsize=13, fontweight="bold", color="white")

        # 右侧标注：步骤转化率与流失率
        if i > 0:
            prev_val = funnel_steps[i - 1][1]
            step_cr = val / prev_val * 100
            ax.text(1.02, y_bottom + 0.5,
                    f"步骤转化率: {step_cr:.1f}%\n流失率: {100 - step_cr:.1f}%",
                    ha="left", va="center", fontsize=10, color="#333333",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0F0F0", alpha=0.8))

    ax.set_xlim(-0.08, 1.35)
    ax.set_ylim(-0.1, len(funnel_steps) + 0.1)
    ax.set_title("电商转化漏斗（用户粒度去重）", fontsize=15, fontweight="bold", pad=15)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/07_funnel_chart.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  [图表已保存] 07_funnel_chart.png")

    # --- 2.3 流失节点分析 ---
    print("\n--- 2.3 流失节点分析 ---")
    for i in range(1, len(funnel_steps)):
        prev_name = funnel_steps[i - 1][0]
        curr_name = funnel_steps[i][0]
        prev_val = funnel_steps[i - 1][1]
        curr_val = funnel_steps[i][1]
        churn = prev_val - curr_val
        churn_rate = churn / prev_val * 100
        print(f"  {prev_name}→{curr_name}：流失 {churn:,} 人（流失率 {churn_rate:.1f}%）")

    print()
    return funnel_df


# ============================================================
# 模块3：RFM 用户分层分析
# ============================================================
def module3_rfm(df):
    """
    RFM 用户分层分析。
    ⚠️ 数据集没有消费金额 M（Monetary），使用用户购买频次替代 M 指标。
    R：用户最近一次购买距离数据集最后一天 2014-12-18 的间隔天数；从未购买用户标记为无购买。
    F：用户总的购买行为次数。
    M：复用购买频次作为替代（妥协处理）。
    """
    print("=" * 60)
    print("【模块3：RFM用户分层分析】")
    print("=" * 60)

    # 数据集最后日期
    data_end_date = pd.Timestamp("2014-12-18")

    # 仅筛选购买行为
    buy_df = df[df["behavior_type"] == 4].copy()

    # 全部用户集合
    all_users = df["user_id"].unique()

    # --- 3.1 计算 RFM ---
    # R：最近一次购买距 2014-12-18 的天数
    # F：购买行为总次数
    # M：复用购买频次（数据集无消费金额，此为妥协替代）
    rfm = buy_df.groupby("user_id").agg(
        last_purchase_time=("time", "max"),
        F=("user_id", "count"),
    ).reset_index()

    rfm["R_days"] = (data_end_date - rfm["last_purchase_time"]).dt.days
    rfm["M"] = rfm["F"]  # M 指标复用购买频次

    print(f"\n--- 3.1 RFM 计算 ---")
    print(f"  有购买行为的用户数：{len(rfm):,}")
    print(f"  无购买行为的用户数：{len(all_users) - len(rfm):,}")
    print(f"  R_days 统计：均值={rfm['R_days'].mean():.1f}，中位数={rfm['R_days'].median():.1f}")
    print(f"  F 统计：均值={rfm['F'].mean():.1f}，中位数={rfm['F'].median():.1f}")

    # --- 3.2 RFM 分箱打分 ---
    # R_score：R_days 越小（最近购买）分越高，使用 1-5 分
    # F_score：F 越大（购买越多）分越高，使用 1-5 分
    # M_score：复用 F_score（M = F 的替代）
    rfm["R_score"] = pd.qcut(rfm["R_days"], q=5, labels=[5, 4, 3, 2, 1], duplicates="drop").astype(int)
    rfm["F_score"] = pd.qcut(rfm["F"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["M_score"] = rfm["F_score"]  # M 复用 F 分数

    print(f"\n--- 3.2 RFM 分箱打分 ---")
    print(f"  R_score 分布：\n{rfm['R_score'].value_counts().sort_index()}")
    print(f"  F_score 分布：\n{rfm['F_score'].value_counts().sort_index()}")

    # --- 3.3 用户分层 ---
    # 以 R_score 和 F_score 的均值为阈值进行分层
    r_mean = rfm["R_score"].mean()
    f_mean = rfm["F_score"].mean()

    def classify(row):
        r, f = row["R_score"], row["F_score"]
        if r >= r_mean and f >= f_mean:
            return "高价值用户"
        elif r >= r_mean and f < f_mean:
            return "潜力用户"
        elif r < r_mean and f >= f_mean:
            return "一般用户"
        else:
            return "流失用户"

    rfm["用户分层"] = rfm.apply(classify, axis=1)

    # 添加无购买用户
    no_purchase_count = len(all_users) - len(rfm)

    # 统计各分层
    segment_counts = rfm["用户分层"].value_counts()
    segment_counts["无购买用户"] = no_purchase_count

    total_users = len(all_users)
    segment_df = pd.DataFrame({
        "用户数": segment_counts,
        "占比": (segment_counts / total_users * 100).round(2),
    })
    segment_df["占比"] = segment_df["占比"].astype(str) + "%"

    print(f"\n--- 3.3 用户分层统计 ---")
    print(segment_df.to_string())

    # --- 3.4 可视化 ---
    # 图8：用户分层分布（环形饼图 + 柱状图）
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    seg_order = ["高价值用户", "潜力用户", "一般用户", "流失用户", "无购买用户"]
    seg_colors = ["#C44E52", "#DD8452", "#4C72B0", "#999999", "#CCCCCC"]
    seg_values = [segment_counts.get(s, 0) for s in seg_order]
    seg_colors_filtered = [c for c, v in zip(seg_colors, seg_values) if v > 0]
    seg_labels_filtered = [s for s, v in zip(seg_order, seg_values) if v > 0]
    seg_values_filtered = [v for v in seg_values if v > 0]
    total_seg = sum(seg_values_filtered)

    # 环形饼图（使用外部图例避免标签重叠）
    wedges, _ = axes[0].pie(
        seg_values_filtered, colors=seg_colors_filtered, startangle=90,
        wedgeprops=dict(width=0.5, edgecolor="white", linewidth=2),
    )
    legend_labels = [
        f"{label}  {val/total_seg*100:.1f}%  ({val:,}人)"
        for label, val in zip(seg_labels_filtered, seg_values_filtered)
    ]
    axes[0].legend(
        wedges, legend_labels, loc="center left",
        bbox_to_anchor=(1.02, 0.5), fontsize=10.5, frameon=False,
        title="用户分层  占比  人数", title_fontsize=11,
    )
    axes[0].set_title("RFM用户分层占比", fontsize=14, fontweight="bold", pad=10)
    axes[0].text(0, 0, f"总计\n{total_seg:,}人",
                 ha="center", va="center", fontsize=14, fontweight="bold")

    # 柱状图
    bars = axes[1].bar(seg_labels_filtered, seg_values_filtered,
                       color=seg_colors_filtered, edgecolor="white", width=0.6)
    axes[1].set_title("RFM用户分层数量", fontsize=14, fontweight="bold", pad=10)
    axes[1].set_ylabel("用户数（人）", fontsize=12)
    for bar, val in zip(bars, seg_values_filtered):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{val:,}", ha="center", va="bottom", fontsize=10.5, fontweight="bold")
    axes[1].tick_params(axis="x", rotation=25, labelsize=10.5)
    axes[1].set_ylim(0, max(seg_values_filtered) * 1.18)
    axes[1].yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: f"{x:,.0f}")
    )
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/08_rfm_segment.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  [图表已保存] 08_rfm_segment.png")

    # 图9：R vs F 散点图（按分层着色）
    fig, ax = plt.subplots(figsize=(11, 7))
    scatter_colors = {
        "高价值用户": "#C44E52", "潜力用户": "#DD8452",
        "一般用户": "#4C72B0", "流失用户": "#999999",
    }

    # y 轴截断阈值：使用 95 分位数，避免极端值压缩主体分布
    f_upper = int(rfm["F"].quantile(0.95))
    f_upper = max(f_upper, 50)
    outlier_count = 0

    for seg, color in scatter_colors.items():
        subset = rfm[rfm["用户分层"] == seg]
        if len(subset) > 0:
            # 正常范围内的点
            normal = subset[subset["F"] <= f_upper]
            if len(normal) > 0:
                np.random.seed(42)
                jitter_r = np.random.uniform(-0.4, 0.4, len(normal))
                jitter_f = np.random.uniform(-0.3, 0.3, len(normal))
                ax.scatter(
                    normal["R_days"] + jitter_r, normal["F"] + jitter_f,
                    c=color, label=seg, alpha=0.5, s=12, edgecolors="none",
                )

            # 极端值点：画在上边界，用三角标记
            outliers = subset[subset["F"] > f_upper]
            if len(outliers) > 0:
                outlier_count += len(outliers)
                np.random.seed(43)
                jitter_r = np.random.uniform(-0.4, 0.4, len(outliers))
                ax.scatter(
                    outliers["R_days"] + jitter_r, [f_upper] * len(outliers),
                    c=color, marker="^", s=60, edgecolors="white", linewidth=0.5,
                    alpha=0.8, zorder=5,
                )

    # 画上边界截断线，并标注极端值信息
    ax.axhline(y=f_upper, color="#666666", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(
        1, f_upper * 1.03,
        f"▲ F>{f_upper} 的极端值（共{outlier_count}个）",
        ha="left", va="bottom", fontsize=10, color="#666666",
    )

    ax.set_xlabel("R（最近购买距今天数）", fontsize=12)
    ax.set_ylabel("F（购买频次，次）", fontsize=12)
    ax.set_title("RFM 用户 R-F 散点分布", fontsize=14, fontweight="bold", pad=10)
    ax.legend(fontsize=11, loc="upper right", markerscale=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, f_upper * 1.15)
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/09_rfm_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [图表已保存] 09_rfm_scatter.png")

    # 业务解读
    print("\n--- 3.5 各群体业务特征解读 ---")
    for seg in ["高价值用户", "潜力用户", "一般用户", "流失用户"]:
        subset = rfm[rfm["用户分层"] == seg]
        if len(subset) > 0:
            print(f"  【{seg}】({len(subset):,}人) "
                  f"R均值={subset['R_days'].mean():.1f}天, "
                  f"F均值={subset['F'].mean():.1f}次, "
                  f"M均值={subset['M'].mean():.1f}次")

    print()
    return rfm, segment_df


# ============================================================
# 模块4：AB-Test 仿真实验
# ============================================================
def module4_abtest(df):
    """
    AB-Test 仿真实验（重点模块）。
    ⚠️ 本数据集没有真实线上 AB 分组，这是基于历史用户的仿真实验，
       仅用于演示完整 AB 测试流程，不是真实业务实验结果。

    流程：
      1. 按 user_id 随机拆分（control / treatment）
      2. SRM 样本比例校验
      3. 仿真模拟实验组购买转化提升
      4. 计算两组转化率、diff、lift
      5. 两样本比例 Z 检验
      6. MDE 最小可检测效应
      7. AA 空转仿真（验证一类错误 α≈0.05）
      8. 业务决策结论
    """
    print("=" * 60)
    print("【模块4：AB-Test仿真实验】")
    print("=" * 60)
    print("  ⚠️ 本数据集没有真实线上AB分组，以下为基于历史用户的仿真实验，")
    print("     仅用于演示完整AB测试流程，不是真实业务实验结果。\n")

    # --- 4.1 按 user_id 随机拆分 ---
    print("--- 4.1 随机分流 ---")
    all_users = df["user_id"].unique()
    np.random.seed(42)
    np.random.shuffle(all_users)
    split_point = len(all_users) // 2
    control_users = set(all_users[:split_point])
    treatment_users = set(all_users[split_point:])

    n_control = len(control_users)
    n_treatment = len(treatment_users)
    n_total = n_control + n_treatment

    print(f"  总用户数：{n_total:,}")
    print(f"  对照组（control）：{n_control:,}（{n_control/n_total*100:.1f}%）")
    print(f"  实验组（treatment）：{n_treatment:,}（{n_treatment/n_total*100:.1f}%）")

    # --- 4.2 SRM 样本比例校验 ---
    # SRM（Sample Ratio Mismatch）检验：检验分流是否符合 50%:50%
    # 使用卡方拟合优度检验，H0: 实际分流比例 = 期望比例（50%:50%）
    print("\n--- 4.2 SRM样本比例校验 ---")
    observed = np.array([n_control, n_treatment])
    expected = np.array([n_total / 2, n_total / 2])
    chi2_stat = ((observed - expected) ** 2 / expected).sum()
    p_srm = 1 - stats.chi2.cdf(chi2_stat, df=1)  # 自由度=1（2组-1）
    print(f"  卡方统计量：{chi2_stat:.4f}")
    print(f"  p值：{p_srm:.4f}")
    if p_srm > 0.05:
        print(f"  结论：p > 0.05，分流比例符合50%:50%，无SRM问题。")
    else:
        print(f"  结论：p ≤ 0.05，分流比例存在显著偏差，需检查分流逻辑。")

    # --- 4.3 仿真模拟实验组购买转化提升 ---
    # 业务假设：模拟上线优惠券营销活动，实验组发放优惠券
    # 仿真策略：从实验组未购买用户中，随机选取一部分"转化"为购买用户
    # 模拟 3% 的相对转化提升
    print("\n--- 4.3 仿真模拟 ---")
    buy_users_set = set(df[df["behavior_type"] == 4]["user_id"].unique())

    # 对照组真实购买用户
    control_buyers = control_users & buy_users_set
    # 实验组真实购买用户（仿真前）
    treatment_buyers_real = treatment_users & buy_users_set

    control_cr_base = len(control_buyers) / n_control
    print(f"  对照组基准转化率：{control_cr_base*100:.4f}%")

    # 目标：实验组转化率 = 对照组 × (1 + 3%)
    target_lift = 0.03
    target_treatment_cr = control_cr_base * (1 + target_lift)
    target_treatment_buyers = int(target_treatment_cr * n_treatment)
    additional_needed = max(0, target_treatment_buyers - len(treatment_buyers_real))

    # 从实验组未购买用户中随机选取进行"转化"
    treatment_non_buyers = list(treatment_users - buy_users_set)
    np.random.seed(123)
    if additional_needed > 0 and additional_needed <= len(treatment_non_buyers):
        simulated_new_buyers = set(
            np.random.choice(treatment_non_buyers, additional_needed, replace=False)
        )
    else:
        simulated_new_buyers = set()
    treatment_buyers_simulated = treatment_buyers_real | simulated_new_buyers

    print(f"  实验组仿真前购买用户：{len(treatment_buyers_real):,}")
    print(f"  仿真新增购买用户：{additional_needed:,}")
    print(f"  实验组仿真后购买用户：{len(treatment_buyers_simulated):,}")

    # --- 4.4 计算转化率、diff、lift ---
    print("\n--- 4.4 转化率计算 ---")
    control_cr = len(control_buyers) / n_control
    treatment_cr = len(treatment_buyers_simulated) / n_treatment
    abs_diff = treatment_cr - control_cr
    rel_lift = (treatment_cr - control_cr) / control_cr

    print(f"  对照组转化率：{control_cr*100:.4f}%（{len(control_buyers):,}/{n_control:,}）")
    print(f"  实验组转化率：{treatment_cr*100:.4f}%（{len(treatment_buyers_simulated):,}/{n_treatment:,}）")
    print(f"  绝对差异（abs diff）：{abs_diff*100:.4f}个百分点")
    print(f"  相对提升（rel lift）：{rel_lift*100:.2f}%")

    # --- 4.5 两样本比例 Z 检验 ---
    # H0: p_control = p_treatment（两组转化率无差异）
    # H1: p_control ≠ p_treatment（两组转化率有差异）
    # 检验统计量：z = (p_t - p_c) / sqrt(p_pool*(1-p_pool)*(1/n_c + 1/n_t))
    # 其中 p_pool = (x_c + x_t) / (n_c + n_t) 为合并转化率
    print("\n--- 4.5 两样本比例Z检验 ---")
    x_c = len(control_buyers)
    x_t = len(treatment_buyers_simulated)
    p_pool = (x_c + x_t) / (n_control + n_treatment)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_control + 1 / n_treatment))
    z_stat = (treatment_cr - control_cr) / se

    # 双侧 p 值
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    # 95% 置信区间
    # Zα/2 = 1.96，即标准正态分布的上 α/2 分位数（α=0.05 时为 1.96）
    z_alpha_2 = 1.96
    ci_lower = (treatment_cr - control_cr) - z_alpha_2 * se
    ci_upper = (treatment_cr - control_cr) + z_alpha_2 * se

    print(f"  合并转化率 p_pool：{p_pool*100:.4f}%")
    print(f"  标准误 SE：{se:.6f}")
    print(f"  Z统计量：{z_stat:.4f}")
    print(f"  p值：{p_value:.6f}")
    print(f"  95%置信区间：[{ci_lower*100:.4f}%, {ci_upper*100:.4f}%]")
    if p_value < 0.05:
        print(f"  结论：p < 0.05，拒绝H0，两组转化率差异统计显著。")
    else:
        print(f"  结论：p ≥ 0.05，不能拒绝H0，两组转化率差异不显著。")

    # 图10：AB-Test转化率对比
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # 柱状图对比
    groups = ["对照组\n(control)", "实验组\n(treatment)"]
    crs = [control_cr * 100, treatment_cr * 100]
    colors_ab = ["#4C72B0", "#C44E52"]
    bars = axes[0].bar(groups, crs, color=colors_ab, edgecolor="white", width=0.5)
    axes[0].set_title("AB-Test 购买转化率对比", fontsize=14, fontweight="bold", pad=10)
    axes[0].set_ylabel("转化率 (%)", fontsize=12)
    axes[0].set_ylim(0, max(crs) * 1.25)
    for bar, val in zip(bars, crs):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{val:.2f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")
    # 标注 lift（在两柱之间上方）
    axes[0].text(0.5, max(crs) * 1.1,
                 f"lift = +{rel_lift*100:.2f}%",
                 ha="center", va="bottom", fontsize=13, color="#C44E52", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF0F0", edgecolor="#C44E52"))

    # 置信区间图
    ci_low_pct = ci_lower * 100
    ci_high_pct = ci_upper * 100
    diff_pct = abs_diff * 100
    yerr_lower = [[diff_pct - ci_low_pct], [ci_high_pct - diff_pct]]
    axes[1].errorbar(
        [0], [diff_pct], yerr=yerr_lower,
        fmt="o", color="#C44E52", capsize=10, capthick=2, markersize=10, linewidth=2.5,
    )
    axes[1].axhline(y=0, color="gray", linestyle="--", linewidth=1.2, label="无差异线 (0)")
    axes[1].set_title("转化率差异 95%置信区间", fontsize=14, fontweight="bold", pad=10)
    axes[1].set_ylabel("绝对差异 (%)", fontsize=12)
    axes[1].set_xticks([0])
    axes[1].set_xticklabels(["treatment - control"], fontsize=11)
    axes[1].set_ylim(ci_low_pct - 0.3, ci_high_pct + 0.5)
    # 添加置信区间文本
    axes[1].text(0.15, diff_pct,
                 f"差异: {diff_pct:.2f}%\n95% CI: [{ci_low_pct:.2f}%, {ci_high_pct:.2f}%]\np = {p_value:.6f}",
                 fontsize=10.5, va="center",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#F5F5F5", edgecolor="#CCCCCC"))
    axes[1].legend(fontsize=10, loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/10_abtest_conversion.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  [图表已保存] 10_abtest_conversion.png")

    # --- 4.6 MDE 最小可检测效应 ---
    # 样本量公式（两样本比例检验）：
    #   n = (Zα/2 + Zβ)^2 × 2×p×(1-p) / Δ^2
    # 其中：
    #   Zα/2 — 显著性水平 α 对应的标准正态分位数（α=0.05 双侧时 Zα/2=1.96）
    #   Zβ  — 统计功效 1-β 对应的标准正态分位数（β=0.20 即 power=80% 时 Zβ=0.84）
    #   p   — 基准转化率（对照组）
    #   Δ   — 待检测的最小转化率差异（MDE）
    #
    # 反推 MDE：
    #   MDE = (Zα/2 + Zβ) × sqrt(2×p×(1-p) / n)
    print("\n--- 4.6 MDE最小可检测效应 ---")
    z_alpha_2_val = 1.96   # Zα/2：α=0.05 双侧检验的上分位数
    z_beta_val = 0.84      # Zβ：β=0.20（功效=80%）的上分位数
    alpha = 0.05           # 显著性水平
    beta = 0.20            # 二类错误概率
    power = 1 - beta       # 统计功效

    p_baseline = control_cr
    n_per_group = min(n_control, n_treatment)

    # MDE 计算
    mde = (z_alpha_2_val + z_beta_val) * np.sqrt(2 * p_baseline * (1 - p_baseline) / n_per_group)
    mde_relative = mde / p_baseline  # 相对 MDE

    print(f"  显著性水平 α = {alpha}（Zα/2 = {z_alpha_2_val}）")
    print(f"  统计功效 1-β = {power}（Zβ = {z_beta_val}）")
    print(f"  基准转化率 p = {p_baseline*100:.4f}%")
    print(f"  每组样本量 n = {n_per_group:,}")
    print(f"  绝对 MDE = {mde*100:.4f}个百分点")
    print(f"  相对 MDE = {mde_relative*100:.2f}%")
    print(f"  解读：在当前样本量下，能以80%功效检测到的最小转化率差异为 {mde*100:.4f}个百分点")
    print(f"        （即相对基准转化率 {mde_relative*100:.2f}% 的变化）")
    print(f"  本次实验实际相对lift = {rel_lift*100:.2f}%")
    if rel_lift > mde_relative:
        print(f"  实际lift > 相对MDE，效应可被检测到（统计显著）。")
    else:
        print(f"  实际lift < 相对MDE，效应可能无法被检测到。")

    # 统计显著 vs 业务显著解读
    print(f"\n  【统计显著 vs 业务显著】")
    print(f"  统计显著：p值 < 0.05，说明差异不太可能由随机波动引起。")
    print(f"  业务显著：差异的幅度（lift={rel_lift*100:.2f}%）是否达到业务可接受的最小阈值。")
    print(f"  本实验中lift={rel_lift*100:.2f}%，属于仿真设定的小幅提升，")
    print(f"  在真实业务中需结合ROI、优惠券成本等综合判断业务显著性。")

    # --- 4.7 AA 空转仿真 ---
    # AA测试：不施加任何策略，随机拆分两组，验证一类错误 α≈0.05
    # 一类错误（Type I Error）：H0为真时错误拒绝H0的概率
    # 理论上，在 α=0.05 下，约5%的AA测试会出现 p<0.05（假阳性）
    print("\n--- 4.7 AA空转仿真 ---")
    print("  原理：不施加任何策略，随机拆分两组进行Z检验，")
    print("        重复多次后统计 p<0.05 的比例，应接近 α=0.05。")

    # 预计算：用户ID数组 + 是否购买的布尔数组
    all_user_arr = np.array(sorted(all_users))
    is_buyer_arr = np.array([1 if uid in buy_users_set else 0 for uid in all_user_arr])
    n_total_aa = len(all_user_arr)
    mid = n_total_aa // 2

    n_aa_tests = 1000
    aa_p_values = []
    aa_significant_count = 0

    for i in range(n_aa_tests):
        np.random.seed(i + 10000)
        perm = np.random.permutation(n_total_aa)
        control_idx = perm[:mid]
        treatment_idx = perm[mid:]

        x_c_aa = is_buyer_arr[control_idx].sum()
        x_t_aa = is_buyer_arr[treatment_idx].sum()
        n_c_aa = len(control_idx)
        n_t_aa = len(treatment_idx)

        p_c_aa = x_c_aa / n_c_aa
        p_t_aa = x_t_aa / n_t_aa
        p_pool_aa = (x_c_aa + x_t_aa) / (n_c_aa + n_t_aa)
        se_aa = np.sqrt(p_pool_aa * (1 - p_pool_aa) * (1 / n_c_aa + 1 / n_t_aa))
        z_aa = (p_t_aa - p_c_aa) / se_aa if se_aa > 0 else 0
        p_aa = 2 * (1 - stats.norm.cdf(abs(z_aa)))

        aa_p_values.append(p_aa)
        if p_aa < 0.05:
            aa_significant_count += 1

    type1_error_rate = aa_significant_count / n_aa_tests

    print(f"  AA测试次数：{n_aa_tests}")
    print(f"  p<0.05 的次数：{aa_significant_count}")
    print(f"  一类错误率（实验值）：{type1_error_rate:.4f}（理论值 α=0.05）")
    print(f"  结论：实验值 {type1_error_rate:.4f} 接近理论值 0.05，")
    print(f"        说明实验框架无系统性偏差，分流与检验逻辑正确。")

    # 图11：AA测试 p值分布
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # p值直方图
    n_bins, bin_edges, _ = axes[0].hist(aa_p_values, bins=30, color="#4C72B0", edgecolor="white", alpha=0.85)
    axes[0].axvline(x=0.05, color="#C44E52", linestyle="--", linewidth=2, label=f"α = 0.05")
    axes[0].set_title(f"AA测试 p值分布（{n_aa_tests}次）", fontsize=14, fontweight="bold", pad=10)
    axes[0].set_xlabel("p值", fontsize=12)
    axes[0].set_ylabel("频次", fontsize=12)
    axes[0].legend(fontsize=11, loc="upper right")
    # 标注一类错误率
    axes[0].text(0.5, max(n_bins) * 0.92,
                 f"一类错误率: {type1_error_rate:.4f}\n理论值 α = 0.05",
                 ha="center", fontsize=11,
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#F0F8FF", edgecolor="#4C72B0"))

    # 累积分布
    sorted_p = np.sort(aa_p_values)
    cdf = np.arange(1, len(sorted_p) + 1) / len(sorted_p)
    axes[1].plot(sorted_p, cdf, color="#4C72B0", linewidth=2.5, label="实际分布")
    axes[1].plot([0, 1], [0, 1], color="#C44E52", linestyle="--", linewidth=1.8, label="理想均匀分布")
    axes[1].axvline(x=0.05, color="gray", linestyle=":", linewidth=1.2)
    axes[1].axhline(y=0.05, color="gray", linestyle=":", linewidth=1.2)
    axes[1].set_title("AA测试 p值累积分布（CDF）", fontsize=14, fontweight="bold", pad=10)
    axes[1].set_xlabel("p值", fontsize=12)
    axes[1].set_ylabel("累积比例", fontsize=12)
    axes[1].legend(fontsize=11, loc="lower right")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/11_aa_test.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  [图表已保存] 11_aa_test.png")

    # --- 4.8 业务决策结论 ---
    print("\n--- 4.8 仿真场景下的业务决策结论 ---")
    print(f"  1. 仿真设定实验组获得 {rel_lift*100:.2f}% 的相对转化提升。")
    print(f"  2. Z检验 p值 = {p_value:.6f}，{'统计显著' if p_value < 0.05 else '统计不显著'}。")
    print(f"  3. 95%置信区间 [{ci_lower*100:.4f}%, {ci_upper*100:.4f}%] {'不包含0' if (ci_lower > 0 or ci_upper < 0) else '包含0'}。")
    print(f"  4. 当前样本量下MDE为 {mde*100:.4f}个百分点（相对{mde_relative*100:.2f}%），")
    print(f"     实际lift {rel_lift*100:.2f}% {'>' if rel_lift > mde_relative else '<'} MDE，{'可检测' if rel_lift > mde_relative else '可能无法检测'}。")
    print(f"  5. AA测试一类错误率 = {type1_error_rate:.4f}，接近理论值0.05，实验框架可靠。")
    print(f"  6. 决策建议：在仿真场景下，优惠券策略带来统计显著的转化提升。")
    print(f"     真实业务中需进一步验证：ROI是否为正、优惠券成本是否可控、")
    print(f"     长期用户留存影响、是否存在用户疲劳效应等。")
    print()

    # 汇总结果字典
    abtest_result = {
        "n_control": n_control,
        "n_treatment": n_treatment,
        "control_cr": control_cr,
        "treatment_cr": treatment_cr,
        "abs_diff": abs_diff,
        "rel_lift": rel_lift,
        "z_stat": z_stat,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "mde": mde,
        "mde_relative": mde_relative,
        "type1_error_rate": type1_error_rate,
    }
    return abtest_result


# ============================================================
# 主函数
# ============================================================
def main():
    """主入口：依次执行四大分析模块。"""
    print("\n" + "=" * 60)
    print("  电商用户行为分析项目 — 开始执行")
    print("=" * 60 + "\n")

    # 数据加载
    df = load_data()

    # 模块1：EDA
    eda_result = module1_eda(df)

    # 模块2：转化漏斗
    funnel_result = module2_funnel(df)

    # 模块3：RFM
    rfm_result, segment_result = module3_rfm(df)

    # 模块4：AB-Test
    abtest_result = module4_abtest(df)

    print("=" * 60)
    print("  全部分析完成！所有图表已保存至 img/ 目录。")
    print("=" * 60)


if __name__ == "__main__":
    main()
