import os

os.environ['OMP_NUM_THREADS'] = '2'

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO, StringIO
from sklearn.datasets import make_classification, make_blobs, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans, DBSCAN
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, silhouette_score, confusion_matrix
import base64
from datetime import datetime
import time
from functools import wraps
from scipy import stats

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer
from reportlab.lib.units import inch
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False


st.set_page_config(
    page_title="数据挖掘算法可视化平台 | 智能分析",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
        color: white;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #fff 0%, #a0aec0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .insight-card {
        background: rgba(30, 40, 60, 0.6);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    hr {
        margin: 1rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #667eea, #764ba2, transparent);
    }
</style>
""", unsafe_allow_html=True)

# ==================== Hero区域 ====================
st.markdown("""
<div style="
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 24px;
    padding: 2rem;
    margin-bottom: 2rem;
    text-align: center;
">
    <h1 style="color: white; margin-bottom: 0.5rem;">🤖 数据挖掘算法可视化平台</h1>
    <p style="color: rgba(255,255,255,0.95);">智能分析 | 实时可视化 | 一键报告 | 实验辅助理解</p>
</div>
""", unsafe_allow_html=True)


# ==================== 智能报告生成器 ====================
class IntelligentReportGenerator:
    """智能报告生成器 - 自动生成数据分析、模型原理、性能分析等内容"""

    @staticmethod
    def generate_data_insights(df, X, y, task):
        """生成数据智能洞察"""
        insights = []

        # 1. 数据规模分析
        insights.append(f"数据规模：数据集包含 {len(df):,} 个样本，{X.shape[1]} 个特征。")

        # 2. 数据质量分析
        missing_rate = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
        if missing_rate == 0:
            insights.append("数据质量：数据完整度100%，无缺失值，数据质量良好。")
        else:
            insights.append(f"数据质量：数据缺失率为 {missing_rate:.1f}%，建议进行数据清洗。")

        # 3. 数据分布分析
        if X.shape[1] >= 1:
            skewness = stats.skew(X[:, 0]) if len(X) > 0 else 0
            if abs(skewness) < 0.5:
                insights.append("数据分布：特征分布接近正态分布，适合大多数算法。")
            elif abs(skewness) < 1:
                insights.append("数据分布：特征存在轻微偏斜，可考虑数据变换优化。")
            else:
                insights.append("数据分布：特征分布严重偏斜，建议进行对数变换或Box-Cox变换。")

        # 4. 任务特定分析
        if task == "分类任务" and y is not None:
            unique, counts = np.unique(y, return_counts=True)
            class_balance = min(counts) / max(counts) if len(counts) > 1 else 1
            insights.append(f"类别分析：共 {len(unique)} 个类别")
            if class_balance > 0.8:
                insights.append("类别分布均衡，适合各类分类算法。")
            elif class_balance > 0.5:
                insights.append("类别分布基本均衡，注意可能存在轻微不平衡。")
            else:
                insights.append("⚠️ 类别分布严重不平衡，建议使用SMOTE等过采样技术或调整类别权重。")

        elif task == "聚类任务":
            from scipy.spatial.distance import pdist
            if len(X) > 1:
                avg_distance = np.mean(pdist(X))
                insights.append(f"聚类分析：样本间平均距离为 {avg_distance:.3f}")
                if avg_distance > 1:
                    insights.append("样本分布较为分散，适合基于密度的聚类算法如DBSCAN。")
                else:
                    insights.append("样本分布较为密集，适合K-Means等基于中心的聚类算法。")

        elif task == "回归任务" and y is not None:
            y_range = y.max() - y.min()
            y_std = y.std()
            insights.append(f"目标变量分析：目标变量范围 {y_range:.2f}，标准差 {y_std:.2f}")
            if y_std / y_range > 0.3:
                insights.append("目标变量变异较大，适合使用多项式回归等灵活模型。")
            else:
                insights.append("目标变量变异较小，线性回归可能已足够。")

        return insights

    @staticmethod
    def generate_model_performance_analysis(model_info, task, y_test=None, y_pred=None):
        """生成模型性能智能分析"""
        analysis = []

        if task == "分类任务":
            acc_str = model_info.get('准确率', '0%')
            acc = float(acc_str.strip('%')) / 100 if acc_str else 0
            analysis.append(f"🎯 **准确率分析**：模型准确率达到 {acc:.2%}")

            if acc >= 0.9:
                analysis.append("✅ 表现优秀，模型具有很强的分类能力，可以很好地识别不同类别的样本。")
            elif acc >= 0.8:
                analysis.append("✅ 表现良好，模型能够较好地完成分类任务，适合实际应用。")
            elif acc >= 0.7:
                analysis.append("📊 表现中等，可以考虑进一步优化参数或尝试其他算法。")
            else:
                analysis.append("⚠️ 表现一般，建议检查数据质量、特征选择或尝试其他算法。")

            # 过拟合/欠拟合分析
            if '训练准确率' in model_info and '测试准确率' in model_info:
                train_acc_str = model_info['训练准确率']
                test_acc_str = model_info['测试准确率']
                train_acc = float(train_acc_str.strip('%')) / 100 if train_acc_str else 0
                test_acc = float(test_acc_str.strip('%')) / 100 if test_acc_str else 0
                gap = train_acc - test_acc
                if gap > 0.1:
                    analysis.append(
                        f"⚠️ **过拟合风险**：训练准确率({train_acc:.2%})明显高于测试准确率({test_acc:.2%})，存在过拟合，建议降低模型复杂度或增加正则化。")
                elif gap < -0.05:
                    analysis.append(
                        f"📉 **欠拟合可能**：测试准确率({test_acc:.2%})高于训练准确率({train_acc:.2%})，可能需要增加样本量或模型复杂度。")
                else:
                    analysis.append(
                        f"✅ **泛化能力**：训练集({train_acc:.2%})与测试集({test_acc:.2%})表现一致，模型泛化能力良好。")

        elif task == "聚类任务":
            sil_str = model_info.get('轮廓系数', '0')
            sil = float(sil_str) if sil_str else 0
            analysis.append(f"🔍 **轮廓系数分析**：轮廓系数为 {sil:.3f}")

            if sil >= 0.5:
                analysis.append("✅ 聚类效果良好，簇内样本紧密、簇间分离明显，聚类结构清晰。")
            elif sil >= 0.3:
                analysis.append("📊 聚类效果中等，簇结构较为合理，但仍有优化空间。")
            elif sil >= 0:
                analysis.append("⚠️ 聚类效果一般，建议调整参数或尝试其他算法。")
            else:
                analysis.append("❌ 轮廓系数为负，表明聚类效果不理想，可能是参数设置不当或数据不适合聚类。")

            n_clusters = model_info.get('簇数量', model_info.get('K值', 0))
            if n_clusters:
                analysis.append(f"📊 **簇结构分析**：共生成 {n_clusters} 个簇")
                if n_clusters <= 3:
                    analysis.append("簇数量较少，数据划分较为宏观，适合初步探索。")
                elif n_clusters <= 6:
                    analysis.append("簇数量适中，能够较好地反映数据的自然结构。")
                else:
                    analysis.append("簇数量较多，可能存在过细划分，建议检查簇的合理性。")

        elif task == "回归任务":
            r2_str = model_info.get('R2', '0')
            mse_str = model_info.get('MSE', '0')
            r2 = float(r2_str) if r2_str else 0
            mse = float(mse_str) if mse_str else 0
            analysis.append(f"📈 **拟合优度分析**：R²分数为 {r2:.3f}，MSE为 {mse:.3f}")

            if r2 >= 0.8:
                analysis.append("✅ 模型拟合效果优秀，能够很好地解释目标变量的变化。")
            elif r2 >= 0.6:
                analysis.append("✅ 模型拟合效果良好，能够解释大部分目标变量变化。")
            elif r2 >= 0.4:
                analysis.append("📊 模型拟合效果中等，可以考虑增加特征或使用更复杂的模型。")
            else:
                analysis.append("⚠️ 模型拟合效果较差，建议检查线性假设或尝试非线性模型。")

            if mse > 100:
                analysis.append("⚠️ **预测误差**：MSE较大，预测误差较高，可能需要更多特征或数据预处理。")

        return analysis

    @staticmethod
    def generate_algorithm_explanation(alg_name, task, params=None):
        """生成算法原理详细说明"""
        # 确保params是字典
        if params is None:
            params = {}

        # 默认参数值
        default_k = params.get('K值', 5)
        default_depth = params.get('最大深度', 3)
        default_c = params.get('C值', 1.0)
        default_eps = params.get('eps', 0.5)
        default_min_samples = params.get('min_samples', 5)
        default_degree = params.get('次数', 2)

        explanations = {
            "KNN": {
                "原理": "🔍 **K近邻算法原理**：KNN是一种基于实例的懒惰学习算法，不进行显式的训练过程。对于新样本，算法计算其与训练集中所有样本的距离（通常使用欧氏距离），选择距离最近的K个邻居，通过多数投票（分类）或平均值（回归）进行预测。",
                "核心思想": "💡 **核心思想**：'近朱者赤，近墨者黑' — 一个样本的类别由其最近的K个邻居决定。",
                "适用场景": "🎯 **适用场景**：适合小样本、低维度的数据集，对异常值敏感，需要特征归一化。常用于推荐系统、模式识别等领域。",
                "优缺点": "✅ **优点**：简单、无训练过程、适合多分类\n❌ **缺点**：计算量大、对高维数据效果差、易受噪声影响",
                "调参建议": f"⚙️ **参数建议**：K={default_k}，K值越小模型越复杂易过拟合，K值越大模型越平滑。建议K取奇数避免平局，常用范围为3-15，可通过交叉验证选择最优K值。"
            },
            "决策树": {
                "原理": "🌳 **决策树算法原理**：决策树通过递归地选择最优特征进行数据划分，形成树状结构。内部节点表示特征测试，分支表示测试结果，叶节点表示决策结果。常用的划分准则包括信息增益、基尼系数等。",
                "核心思想": "💡 **核心思想**：通过一系列if-then规则将特征空间划分为不同的区域，每个区域对应一个预测值。",
                "适用场景": "🎯 **适用场景**：适合需要解释性的场景（如医疗诊断、金融风控），能处理数值型和分类型特征，对数据预处理要求低。",
                "优缺点": "✅ **优点**：可解释性强、能处理非线性关系、不需要特征缩放\n❌ **缺点**：易过拟合、对噪声敏感、不稳定",
                "调参建议": f"⚙️ **参数建议**：最大深度={default_depth}，控制树的复杂度，深度过大会导致过拟合，过小会导致欠拟合。建议从3-5开始尝试，配合剪枝策略。"
            },
            "SVM": {
                "原理": "📐 **支持向量机原理**：SVM通过寻找能够最大化分类间隔的超平面来实现分类。对于线性不可分数据，通过核函数将数据映射到高维空间，在高维空间中寻找最优超平面。支持向量是位于分类间隔边界上的样本点。",
                "核心思想": "💡 **核心思想**：找到最优分类超平面，使得两类样本之间的间隔最大化。",
                "适用场景": "🎯 **适用场景**：适合高维小样本数据，对非线性问题有良好表现，但对大规模数据计算量大。广泛应用于图像分类、文本分类等。",
                "优缺点": "✅ **优点**：高维数据表现好、泛化能力强、理论基础扎实\n❌ **缺点**：计算量大、对大规模数据不友好、多分类需扩展",
                "调参建议": f"⚙️ **参数建议**：C={default_c}，C越大对误分类惩罚越大，模型越复杂；核函数选择：linear适合线性可分，rbf适合非线性问题。"
            },
            "逻辑回归": {
                "原理": "📊 **逻辑回归原理**：逻辑回归通过Sigmoid函数将线性回归的输出映射到(0,1)区间，输出样本属于正类的概率。通过最大似然估计求解模型参数，是一种广义线性模型。",
                "核心思想": "💡 **核心思想**：将线性回归的连续输出转换为概率输出，用于二分类问题。",
                "适用场景": "🎯 **适用场景**：适合二分类问题，输出具有概率意义，计算快速，可解释性强。常用于信用评分、医学诊断等。",
                "优缺点": "✅ **优点**：可解释性强、输出概率、计算快、不易过拟合\n❌ **缺点**：仅适用于线性可分数据、对异常值敏感",
                "调参建议": f"⚙️ **参数建议**：可调整正则化参数防止过拟合，L1正则化产生稀疏解，L2正则化防止权重过大。"
            },
            "朴素贝叶斯": {
                "原理": "📈 **朴素贝叶斯原理**：朴素贝叶斯基于贝叶斯定理，假设特征之间相互独立（条件独立性假设）。计算样本属于每个类别的后验概率，选择概率最大的类别作为预测结果。",
                "核心思想": "💡 **核心思想**：'朴素'指的是特征条件独立假设，简化了概率计算。",
                "适用场景": "🎯 **适用场景**：适合高维数据（如文本分类），对缺失数据不敏感，计算快速，但特征独立性假设在现实中往往不成立。",
                "优缺点": "✅ **优点**：计算快、适合高维数据、抗噪声、对小样本友好\n❌ **缺点**：特征独立性假设过强，实际数据中常不成立",
                "调参建议": f"⚙️ **参数建议**：可指定各类别的先验概率，默认从数据中计算；对于连续特征，高斯朴素贝叶斯假设特征服从正态分布。"
            },
            "K-Means": {
                "原理": "🔴 **K-Means算法原理**：K-Means是一种基于中心的聚类算法。随机初始化K个簇中心，迭代地进行样本分配（将每个样本分配到最近的簇中心）和簇中心更新（计算簇内样本均值），直到收敛。目标是最小化簇内样本到簇中心的距离平方和。",
                "核心思想": "💡 **核心思想**：'物以类聚' — 相似的样本会被聚在一起，簇内距离最小化。",
                "适用场景": "🎯 **适用场景**：适合发现球形簇，算法快速简单，但需要预先指定K值，对初始中心敏感。常用于客户细分、图像压缩等。",
                "优缺点": "✅ **优点**：实现简单、计算快、可解释性好、适合大规模数据\n❌ **缺点**：需手动指定K值、对初始中心敏感、易陷入局部最优、只能发现球形簇",
                "调参建议": f"⚙️ **参数建议**：K={params.get('K值', 3)}，可通过肘部法则或轮廓系数确定最优K值；使用k-means++初始化可改善结果稳定性。"
            },
            "DBSCAN": {
                "原理": "🔵 **DBSCAN算法原理**：DBSCAN是一种基于密度的聚类算法。将密度相连的样本划分为同一簇，能够发现任意形状的簇，自动识别噪声点。核心思想是用一个点的邻域半径(eps)和邻域内最少样本数(min_samples)来定义密度。",
                "核心思想": "💡 **核心思想**：'物以类聚，人以群分' — 高密度区域形成簇，低密度区域为噪声。",
                "适用场景": "🎯 **适用场景**：适合发现任意形状的簇，对噪声鲁棒，不需要指定簇数，但对参数敏感。常用于空间数据分析、异常检测等。",
                "优缺点": "✅ **优点**：无需指定K值、可识别任意形状簇、抗噪声、可发现异常点\n❌ **缺点**：对高维数据效果差、对参数敏感、密度不均匀时效果差",
                "调参建议": f"⚙️ **参数建议**：eps={default_eps}，决定密度范围；min_samples={default_min_samples}，决定核心点判定。eps过小会产生过多噪声，eps过大会合并不同簇。"
            },
            "线性回归": {
                "原理": "📏 **线性回归原理**：线性回归假设目标变量与特征之间存在线性关系，通过最小化平方误差损失函数（最小二乘法）求解最优参数。模型形式为 y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b。",
                "核心思想": "💡 **核心思想**：找到一条最佳拟合直线，使得所有样本点到直线的距离平方和最小。",
                "适用场景": "🎯 **适用场景**：适合线性关系的数据，对异常值敏感，可解释性强。常用于销售预测、趋势分析等。",
                "优缺点": "✅ **优点**：实现简单、可解释性强、计算快、理论基础扎实\n❌ **缺点**：仅适用于线性关系、对异常值敏感、容易欠拟合",
                "调参建议": f"⚙️ **参数建议**：可使用正则化（Ridge/Lasso）防止过拟合和进行特征选择；检查残差是否满足独立同分布假设。"
            },
            "多项式回归": {
                "原理": "📈 **多项式回归原理**：多项式回归是线性回归的扩展，通过添加特征的高次项（x², x³等）来拟合非线性关系。本质上是将非线性问题转化为线性问题求解，模型形式为 y = w₀ + w₁x + w₂x² + ... + wₙxⁿ。",
                "核心思想": "💡 **核心思想**：'曲线救国' — 通过增加多项式特征，用线性模型拟合非线性关系。",
                "适用场景": "🎯 **适用场景**：适合拟合非线性关系，但次数过高容易过拟合。常用于物理实验数据拟合、经济趋势预测等。",
                "优缺点": "✅ **优点**：可拟合非线性关系、实现简单、基于线性回归\n❌ **缺点**：易过拟合、对高次项敏感、解释性随次数降低",
                "调参建议": f"⚙️ **参数建议**：次数={default_degree}，控制模型的复杂度。次数越高拟合能力越强，但过拟合风险越大。建议从2-3次开始尝试，使用验证集防止过拟合。"
            }
        }

        # 匹配算法名称
        for key in explanations:
            if key in alg_name:
                return explanations[key]

        # 默认返回
        return {
            "原理": f"📚 **{alg_name}算法原理**：{alg_name}是一种常用的{task}算法，通过分析数据特征之间的内在规律进行学习。",
            "核心思想": "💡 **核心思想**：基于数据特征的统计规律进行预测或聚类。",
            "适用场景": "🎯 **适用场景**：适用于大多数数据分析场景，建议根据数据特点选择合适的参数。",
            "优缺点": "✅ **优点**：实现简单、计算快速\n❌ **缺点**：需要根据具体任务调整参数",
            "调参建议": f"⚙️ **参数建议**：请参考算法参数说明进行调整，通过交叉验证选择最佳参数组合。"
        }

    @staticmethod
    def generate_optimization_suggestions(model_info, task):
        """生成优化建议"""
        suggestions = []

        if task == "分类任务":
            acc_str = model_info.get('准确率', '0%')
            acc = float(acc_str.strip('%')) / 100 if acc_str else 0
            if acc < 0.8:
                suggestions.append("💡 **特征工程**：尝试增加特征数量或进行特征组合，提取更有信息量的特征。")
                suggestions.append("💡 **集成学习**：尝试集成学习方法如随机森林、XGBoost等，通常能获得更好效果。")
            if acc < 0.7:
                suggestions.append("💡 **数据质量**：检查数据质量，处理异常值和缺失值。")
                suggestions.append("💡 **交叉验证**：使用交叉验证选择最佳参数，避免偶然性。")
            else:
                suggestions.append("💡 **模型调优**：当前模型表现良好，可通过网格搜索微调参数获得更好效果。")

        elif task == "聚类任务":
            sil_str = model_info.get('轮廓系数', '0')
            sil = float(sil_str) if sil_str else 0
            if sil < 0.3:
                suggestions.append("💡 **参数优化**：尝试不同的K值范围，使用肘部法则或轮廓系数选择最优K值。")
                suggestions.append("💡 **数据降维**：尝试数据降维(PCA/t-SNE)后再进行聚类，减少维度灾难影响。")
            if "DBSCAN" in model_info.get('算法', ''):
                suggestions.append(
                    "💡 **DBSCAN调参**：调整eps和min_samples参数，eps过小会产生过多噪声，eps过大会合并不同簇。")
            else:
                suggestions.append("💡 **初始化优化**：使用k-means++初始化方法改善聚类结果稳定性。")

        elif task == "回归任务":
            r2_str = model_info.get('R2', '0')
            r2 = float(r2_str) if r2_str else 0
            if r2 < 0.6:
                suggestions.append("💡 **特征交互**：尝试添加特征交互项或使用非线性模型。")
                suggestions.append("💡 **异常处理**：检查是否存在异常值，考虑使用稳健回归方法。")
            if "多项式" in model_info.get('算法', ''):
                degree = model_info.get('次数', 2)
                suggestions.append(f"💡 **次数调整**：当前使用{degree}次多项式，尝试调整次数或使用正则化防止过拟合。")

        suggestions.append("💡 **数据可视化**：绘制学习曲线和验证曲线，辅助参数调优和诊断模型问题。")
        suggestions.append("💡 **模型对比**：尝试多个算法并进行对比，选择最适合当前数据的模型。")

        return suggestions

    @staticmethod
    def generate_experiment_conclusion(model_info, task):
        """生成实验结论"""
        conclusions = []

        if task == "分类任务":
            acc = model_info.get('准确率', 'N/A')
            algo = model_info.get('算法', '所选算法')
            conclusions.append(f"📊 **实验总结**：本次实验使用{algo}算法对数据集进行分类，最终准确率达到{acc}。")

            if acc != 'N/A' and float(acc.strip('%')) / 100 >= 0.8:
                conclusions.append("✅ **成功经验**：模型表现良好，说明所选算法适用于当前数据集的特征分布。")
            else:
                conclusions.append(
                    "⚠️ **改进方向**：模型表现有提升空间，建议从数据预处理、特征工程、参数调优三个方向进行优化。")

        elif task == "聚类任务":
            sil = model_info.get('轮廓系数', 'N/A')
            algo = model_info.get('算法', '所选算法')
            conclusions.append(f"📊 **实验总结**：本次实验使用{algo}算法对数据进行聚类分析，轮廓系数为{sil}。")

            if sil != 'N/A' and float(sil) >= 0.5:
                conclusions.append("✅ **成功经验**：聚类效果良好，数据具有明显的簇结构。")
            else:
                conclusions.append("⚠️ **改进方向**：聚类效果一般，建议调整参数或尝试其他聚类算法。")

        elif task == "回归任务":
            r2 = model_info.get('R2', 'N/A')
            algo = model_info.get('算法', '所选算法')
            conclusions.append(f"📊 **实验总结**：本次实验使用{algo}算法进行回归分析，R²分数为{r2}。")

            if r2 != 'N/A' and float(r2) >= 0.7:
                conclusions.append("✅ **成功经验**：拟合效果良好，模型能够有效预测目标变量。")
            else:
                conclusions.append("⚠️ **改进方向**：拟合效果一般，建议考虑非线性模型或增加特征。")

        conclusions.append("✍️ **学习收获**：通过本次实验，深入理解了算法原理、参数影响和模型评估方法。")

        return conclusions


# ==================== 进度提示 ====================
def with_progress(message="处理中"):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with st.spinner(f'⏳ {message}...'):
                time.sleep(0.1)
                result = func(*args, **kwargs)
            return result

        return wrapper

    return decorator


# ==================== 侧边栏 ====================
with st.sidebar:
    st.markdown("### 🎛️ 控制面板")
    task = st.selectbox("📌 学习任务", ["分类任务", "聚类任务", "回归任务"])
    st.divider()
    enable_comparison = st.checkbox("🔍 模型对比模式", help="同时训练多个模型进行对比")
    st.divider()
    st.markdown("### 📝 实验信息")
    # 修复：使用session_state确保实时读取输入的信息
    if 'student_name' not in st.session_state:
        st.session_state.student_name = ""
    if 'student_id' not in st.session_state:
        st.session_state.student_id = ""
    if 'student_class' not in st.session_state:
        st.session_state.student_class = ""

    # 实时更新session_state中的值
    st.session_state.student_name = st.text_input("👤 姓名", value=st.session_state.student_name,
                                                  placeholder="请输入姓名")
    st.session_state.student_id = st.text_input("🔢 学号", value=st.session_state.student_id, placeholder="请输入学号")
    st.session_state.student_class = st.text_input("🏫 班级", value=st.session_state.student_class,
                                                   placeholder="请输入班级")

    # 展示实时输入的信息（可选，用于验证）
    if st.session_state.student_name or st.session_state.student_id or st.session_state.student_class:
        st.divider()
        st.markdown("#### 📋 当前信息")
        if st.session_state.student_name:
            st.write(f"姓名：{st.session_state.student_name}")
        if st.session_state.student_id:
            st.write(f"学号：{st.session_state.student_id}")
        if st.session_state.student_class:
            st.write(f"班级：{st.session_state.student_class}")

# ==================== 数据集模块 ====================
st.markdown("## 📊 数据预览")

data_source = st.radio("数据来源", ["✨ 使用内置数据集", "📁 上传CSV文件"], horizontal=True)
df = None
X, y = None, None

if data_source == "✨ 使用内置数据集":
    n_samples = st.slider("样本数量", 100, 1000, 300)
    if task == "分类任务":
        X, y = make_classification(n_samples=n_samples, n_features=2, n_informative=2, n_redundant=0, n_classes=2,
                                   random_state=42)
    elif task == "聚类任务":
        X, y = make_blobs(n_samples=n_samples, centers=3, n_features=2, random_state=42)
        y = None
    elif task == "回归任务":
        X, y = make_regression(n_samples=n_samples, n_features=1, noise=20, random_state=42)
    df = pd.DataFrame(X, columns=["特征1", "特征2"] if X.shape[1] == 2 else ["特征1"])
    if task != "回归任务" and y is not None:
        df["标签"] = y
else:
    uploaded_file = st.file_uploader(label="上传CSV文件", type="csv")

    # 【新增】CSV文件格式提示
    st.info("""
        📋 **CSV文件格式要求（上传前请确认）**
        - 文件编码为 **UTF-8**
        - 文件不包含空行
        - 第一行是列名（特征列+标签列）
        - 所有特征列是数字类型（int/float）
        - 数据量足够（样本数量 > 30 行）
        - 分类/回归任务需选择标签列：
          - 分类任务：标签列需有 **2个以上不同值**
          - 回归任务：标签列需为数字类型（int/float）
        """)

    if uploaded_file is not None:
        try:
            # 读取CSV文件（指定编码为UTF-8）
            df = pd.read_csv(uploaded_file, encoding="utf-8")
            st.success("✅ 文件上传成功！")

            # 【新增】数据格式校验
            valid = True
            # 1. 样本量校验
            if len(df) < 30:
                st.error("❌ 数据量不足！样本数量需大于30行，请检查文件。")
                valid = False
            # 2. 空行校验
            if df.isnull().any().any():
                st.error("❌ 文件包含空值/空行，请清理数据后重新上传。")
                valid = False
            # 3. 特征列类型校验（先让用户选择特征列，再校验）
            feature_cols = st.multiselect("选择特征列", df.columns,
                                          default=df.columns[:-1] if len(df.columns) > 1 else df.columns)
            if feature_cols:
                # 检查特征列是否为数字类型
                for col in feature_cols:
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        st.error(f"❌ 特征列「{col}」不是数字类型，请检查数据。")
                        valid = False
            # 4. 标签列校验（仅非聚类任务）
            if task != "聚类任务":
                target_col = st.selectbox("选择标签列", df.columns, index=len(df.columns) - 1)
                if target_col:
                    if task == "分类任务":
                        # 检查标签列不同值数量
                        if df[target_col].nunique() < 2:
                            st.error("❌ 分类任务标签列需包含2个以上不同值，请检查数据。")
                            valid = False
                    elif task == "回归任务":
                        # 检查标签列是否为数字类型
                        if not pd.api.types.is_numeric_dtype(df[target_col]):
                            st.error("❌ 回归任务标签列需为数字类型，请检查数据。")
                            valid = False
            else:
                # 聚类任务不需要标签列
                target_col = None

            # 校验通过后再执行后续操作
            if valid and feature_cols:
                if task != "聚类任务" and target_col:
                    X = df[feature_cols].values
                    y = df[target_col].values
                else:
                    X = df[feature_cols].values
                    y = None
                    if task == "聚类任务":
                        y = None
            else:
                st.stop()

        except UnicodeDecodeError:
            st.error("❌ 文件编码错误！请将CSV文件保存为UTF-8编码后重新上传。")
        except Exception as e:
            st.error(f"❌ 文件读取失败：{str(e)}")

if df is not None:
    st.dataframe(df.head(8), use_container_width=True)
    # 【新增】下载全部数据按钮
    st.download_button(
        label="📥 下载全部数据（CSV）",
        data=df.to_csv(index=False, encoding="utf-8"),
        file_name=f"数据集_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总样本数", f"{len(df):,}")
    with col2:
        st.metric("特征维度", X.shape[1])
    with col3:
        if task != "聚类任务" and y is not None:
            st.metric("类别数量", len(np.unique(y)))
        else:
            missing_rate = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
            st.metric("数据完整度", f"{100 - missing_rate:.1f}%")
    with col4:
        st.metric("任务类型", task.replace("任务", ""))

    # 显示数据智能洞察
    with st.expander("📈 数据智能洞察", expanded=False):
        insights = IntelligentReportGenerator.generate_data_insights(df, X, y, task)
        for insight in insights:
            st.markdown(f'<div class="insight-card">{insight}</div>', unsafe_allow_html=True)

    st.divider()

    # 展示实验信息（实时读取）
    st.markdown("### 📝 实验信息展示")
    exp_col1, exp_col2, exp_col3 = st.columns(3)
    with exp_col1:
        st.text_input("姓名（只读）", value=st.session_state.student_name, disabled=True)
    with exp_col2:
        st.text_input("学号（只读）", value=st.session_state.student_id, disabled=True)
    with exp_col3:
        st.text_input("班级（只读）", value=st.session_state.student_class, disabled=True)

if X is None:
    st.info("👆 请选择数据集或上传CSV文件开始实验")
    st.stop()

# 标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

if task != "聚类任务" and y is not None:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def make_grid(X, h=0.02):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    return np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))


# ==================== 分类任务 ====================
fig = None
model_info = {}
comparison_results = []
model = None
labels = None
clf_name = None

if task == "分类任务":
    st.markdown("## 📈 分类算法实验")

    if enable_comparison:
        models_to_compare = st.multiselect(
            "选择要对比的分类算法",
            ["KNN", "决策树", "SVM", "逻辑回归", "朴素贝叶斯"],
            default=["KNN", "决策树"]
        )

        if models_to_compare:
            with st.spinner("⏳ 正在训练多个模型..."):
                for model_name in models_to_compare:
                    if model_name == "KNN":
                        temp_model = KNeighborsClassifier(n_neighbors=5)
                    elif model_name == "决策树":
                        temp_model = DecisionTreeClassifier(max_depth=3)
                    elif model_name == "SVM":
                        temp_model = SVC(C=1.0, kernel="linear")
                    elif model_name == "逻辑回归":
                        temp_model = LogisticRegression()
                    else:
                        temp_model = GaussianNB()

                    temp_model.fit(X_train, y_train)
                    train_acc = accuracy_score(y_train, temp_model.predict(X_train))
                    test_acc = accuracy_score(y_test, temp_model.predict(X_test))
                    comparison_results.append({
                        "算法": model_name,
                        "训练准确率": f"{train_acc:.2%}",
                        "测试准确率": f"{test_acc:.2%}",
                        "准确率数值": test_acc,
                        "模型": temp_model
                    })

            st.dataframe(pd.DataFrame(comparison_results)[["算法", "训练准确率", "测试准确率"]],
                         use_container_width=True)

            best_model = max(comparison_results, key=lambda x: x["准确率数值"])
            st.success(f"🏆 推荐算法：**{best_model['算法']}**，测试准确率 {best_model['测试准确率']}")

            # 让用户选择要详细分析的模型
            selected_model_name = st.selectbox(
                "选择要详细可视化和分析的模型",
                [f"{r['算法']} (测试准确率={r['测试准确率']})" for r in comparison_results],
                index=0
            )
            selected = next(
                r for r in comparison_results if f"{r['算法']} (测试准确率={r['测试准确率']})" == selected_model_name)
            clf_name = selected['算法']
            model = selected['模型']
            model_info = {
                "算法": clf_name,
                "训练准确率": selected['训练准确率'],
                "测试准确率": selected['测试准确率'],
                "准确率": selected['测试准确率'],
                # 新增：加入实验人员信息
                "实验人员": st.session_state.student_name,
                "学号": st.session_state.student_id,
                "班级": st.session_state.student_class
            }

            st.metric("测试准确率", selected['测试准确率'])
        else:
            clf_name = st.selectbox("选择分类算法", ["KNN", "决策树", "SVM", "逻辑回归", "朴素贝叶斯"])
            # 训练模型
            if clf_name == "KNN":
                k = st.slider("K近邻数", 1, 30, 5)
                model = KNeighborsClassifier(n_neighbors=k)
                model_info = {
                    "算法": "KNN",
                    "K值": k,
                    "实验人员": st.session_state.student_name,
                    "学号": st.session_state.student_id,
                    "班级": st.session_state.student_class
                }
            elif clf_name == "决策树":
                depth = st.slider("最大深度", 1, 10, 3)
                model = DecisionTreeClassifier(max_depth=depth)
                model_info = {
                    "算法": "决策树",
                    "最大深度": depth,
                    "实验人员": st.session_state.student_name,
                    "学号": st.session_state.student_id,
                    "班级": st.session_state.student_class
                }
            elif clf_name == "SVM":
                c = st.slider("正则化C", 0.01, 10.0, 1.0)
                model = SVC(C=c, kernel="linear")
                model_info = {
                    "算法": "SVM",
                    "C值": c,
                    "实验人员": st.session_state.student_name,
                    "学号": st.session_state.student_id,
                    "班级": st.session_state.student_class
                }
            elif clf_name == "逻辑回归":
                model = LogisticRegression()
                model_info = {
                    "算法": "逻辑回归",
                    "实验人员": st.session_state.student_name,
                    "学号": st.session_state.student_id,
                    "班级": st.session_state.student_class
                }
            else:
                model = GaussianNB()
                model_info = {
                    "算法": "朴素贝叶斯",
                    "实验人员": st.session_state.student_name,
                    "学号": st.session_state.student_id,
                    "班级": st.session_state.student_class
                }

            model.fit(X_train, y_train)
            train_acc = accuracy_score(y_train, model.predict(X_train))
            test_acc = accuracy_score(y_test, model.predict(X_test))
            model_info["训练准确率"] = f"{train_acc:.2%}"
            model_info["测试准确率"] = f"{test_acc:.2%}"
            model_info["准确率"] = f"{test_acc:.2%}"
            st.metric("测试准确率", f"{test_acc:.2%}")
    else:
        clf_name = st.selectbox("选择分类算法", ["KNN", "决策树", "SVM", "逻辑回归", "朴素贝叶斯"])
        if clf_name == "KNN":
            k = st.slider("K近邻数", 1, 30, 5)
            model = KNeighborsClassifier(n_neighbors=k)
            model_info = {
                "算法": "KNN",
                "K值": k,
                "实验人员": st.session_state.student_name,
                "学号": st.session_state.student_id,
                "班级": st.session_state.student_class
            }
        elif clf_name == "决策树":
            depth = st.slider("最大深度", 1, 10, 3)
            model = DecisionTreeClassifier(max_depth=depth)
            model_info = {
                "算法": "决策树",
                "最大深度": depth,
                "实验人员": st.session_state.student_name,
                "学号": st.session_state.student_id,
                "班级": st.session_state.student_class
            }
        elif clf_name == "SVM":
            c = st.slider("正则化C", 0.01, 10.0, 1.0)
            model = SVC(C=c, kernel="linear")
            model_info = {
                "算法": "SVM",
                "C值": c,
                "实验人员": st.session_state.student_name,
                "学号": st.session_state.student_id,
                "班级": st.session_state.student_class
            }
        elif clf_name == "逻辑回归":
            model = LogisticRegression()
            model_info = {
                "算法": "逻辑回归",
                "实验人员": st.session_state.student_name,
                "学号": st.session_state.student_id,
                "班级": st.session_state.student_class
            }
        elif clf_name == "朴素贝叶斯":
            model = GaussianNB()
            model_info = {
                "算法": "朴素贝叶斯",
                "实验人员": st.session_state.student_name,
                "学号": st.session_state.student_id,
                "班级": st.session_state.student_class
            }

        # 训练模型并更新信息
        model.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        model_info["训练准确率"] = f"{train_acc:.2%}"
        model_info["测试准确率"] = f"{test_acc:.2%}"
        model_info["准确率"] = f"{test_acc:.2%}"
        st.metric("测试准确率", f"{test_acc:.2%}")

    # 可视化分类结果（补充原代码缺失的部分，保证功能完整）
    if X.shape[1] == 2 and model is not None:
        st.markdown("### 📈 分类结果可视化")
        fig, ax = plt.subplots(figsize=(10, 6))
        X0, X1 = make_grid(X)
        Z = model.predict(np.c_[X0.ravel(), X1.ravel()])
        Z = Z.reshape(X0.shape)
        ax.contourf(X0, X1, Z, alpha=0.3, cmap=plt.cm.coolwarm)
        ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title(f'{clf_name}  Classification Result (Accuracy:{model_info["测试准确率"]})')
        st.pyplot(fig)

    # 展示完整的模型信息（包含实验人员信息）
    with st.expander("🔍 完整模型信息", expanded=True):
        st.json(model_info)

# ==================== 聚类任务 ====================
if task == "聚类任务":
    st.markdown("## 🔍 聚类算法实验")

    cluster_algo = st.selectbox("选择聚类算法", ["K-Means", "DBSCAN"])
    model_info = {
        "算法": cluster_algo,
        "实验人员": st.session_state.student_name,
        "学号": st.session_state.student_id,
        "班级": st.session_state.student_class
    }

    if cluster_algo == "K-Means":
        k = st.slider("K值（簇数量）", 2, 10, 3)
        model = KMeans(n_clusters=k, random_state=42)
        model_info["K值"] = k
        y_pred = model.fit_predict(X)
        sil_score = silhouette_score(X, y_pred)
        model_info["轮廓系数"] = f"{sil_score:.3f}"
        model_info["簇数量"] = k
    else:
        eps = st.slider("邻域半径(eps)", 0.1, 2.0, 0.5)
        min_samples = st.slider("最小样本数(min_samples)", 2, 20, 5)
        model = DBSCAN(eps=eps, min_samples=min_samples)
        model_info["eps"] = eps
        model_info["min_samples"] = min_samples
        y_pred = model.fit_predict(X)
        n_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)
        model_info["簇数量"] = n_clusters
        if n_clusters > 1:
            sil_score = silhouette_score(X[y_pred != -1], y_pred[y_pred != -1])
            model_info["轮廓系数"] = f"{sil_score:.3f}"
        else:
            model_info["轮廓系数"] = "N/A"

    # 可视化聚类结果
    st.markdown("### 📈 聚类结果可视化")
    fig, ax = plt.subplots(figsize=(10, 6))
    unique_labels = set(y_pred)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]  # 噪声点为黑色

        class_member_mask = (y_pred == k)
        xy = X[class_member_mask]
        ax.scatter(xy[:, 0], xy[:, 1], c=[col], label=f'簇 {k}' if k != -1 else '噪声点', s=50, edgecolors='k')

    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(f'{cluster_algo} Clustering Result (Clusters: {model_info["簇数量"]})')
    ax.legend()
    st.pyplot(fig)

    # 展示聚类指标
    col1, col2 = st.columns(2)
    with col1:
        st.metric("簇数量", model_info["簇数量"])
    with col2:
        st.metric("轮廓系数", model_info["轮廓系数"])

    # 展示完整的模型信息
    with st.expander("🔍 完整模型信息", expanded=True):
        st.json(model_info)

# ==================== 回归任务 ====================
if task == "回归任务":
    st.markdown("## 📈 回归算法实验")

    reg_algo = st.selectbox("选择回归算法", ["线性回归", "多项式回归"])
    model_info = {
        "算法": reg_algo,
        "实验人员": st.session_state.student_name,
        "学号": st.session_state.student_id,
        "班级": st.session_state.student_class
    }

    if reg_algo == "多项式回归":
        degree = st.slider("多项式次数", 2, 5, 2)
        poly_features = PolynomialFeatures(degree=degree)
        X_poly = poly_features.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y)
        model_info["次数"] = degree

        # 预测
        X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        X_range_poly = poly_features.transform(X_range)
        y_pred = model.predict(X_range_poly)

        # 评估
        y_train_pred = model.predict(poly_features.transform(X_train))
        y_test_pred = model.predict(poly_features.transform(X_test))
    else:
        model = LinearRegression()
        model.fit(X, y)

        # 预测
        X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_pred = model.predict(X_range)

        # 评估
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

    # 计算评估指标
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    model_info["训练MSE"] = f"{train_mse:.3f}"
    model_info["测试MSE"] = f"{test_mse:.3f}"
    model_info["训练R2"] = f"{train_r2:.3f}"
    model_info["测试R2"] = f"{test_r2:.3f}"
    model_info["MSE"] = f"{test_mse:.3f}"
    model_info["R2"] = f"{test_r2:.3f}"

    # 可视化回归结果
    st.markdown("### 📈 回归结果可视化")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, y, color='blue', alpha=0.5, label='原始数据')
    ax.plot(X_range, y_pred, color='red', linewidth=2, label=f'{reg_algo} 拟合曲线')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Target Variable')
    ax.set_title(f'{reg_algo} Regression Result (R²：{test_r2:.3f})')
    ax.legend()
    st.pyplot(fig)

    # 展示回归指标
    col1, col2 = st.columns(2)
    with col1:
        st.metric("测试MSE", f"{test_mse:.3f}")
    with col2:
        st.metric("测试R²", f"{test_r2:.3f}")

    # 展示完整的模型信息
    with st.expander("🔍 完整模型信息", expanded=True):
        st.json(model_info)

# ==================== 智能报告生成 ====================
st.markdown("## 📑 智能分析报告")
if st.button("📝 生成完整实验报告", type="primary"):
    with st.spinner("正在生成报告..."):
        # 1. 基础信息
        st.markdown("### 📋 实验基础信息")
        report_info = f"""
        - **实验人员**：{st.session_state.student_name or '未填写'}
        - **学号**：{st.session_state.student_id or '未填写'}
        - **班级**：{st.session_state.student_class or '未填写'}
        - **实验时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        - **任务类型**：{task}
        - **算法名称**：{model_info.get('算法', '未选择')}
        """
        st.markdown(report_info)

        # 2. 数据洞察
        st.markdown("### 📊 数据智能洞察")
        if df is not None:
            insights = IntelligentReportGenerator.generate_data_insights(df, X, y, task)
            for insight in insights:
                st.markdown(f"- {insight}")

        # 3. 算法原理
        st.markdown("### 📚 算法原理说明")
        algo_explain = IntelligentReportGenerator.generate_algorithm_explanation(model_info.get('算法', ''), task,
                                                                                 model_info)
        for key, value in algo_explain.items():
            st.markdown(f"{value}")

        # 4. 性能分析
        st.markdown("### 📈 模型性能分析")
        performance_analysis = IntelligentReportGenerator.generate_model_performance_analysis(model_info, task)
        for analysis in performance_analysis:
            st.markdown(f"- {analysis}")

        # 5. 优化建议
        st.markdown("### 💡 模型优化建议")
        suggestions = IntelligentReportGenerator.generate_optimization_suggestions(model_info, task)
        for suggestion in suggestions:
            st.markdown(f"- {suggestion}")

        # 6. 实验结论
        st.markdown("### 📄 实验结论")
        conclusions = IntelligentReportGenerator.generate_experiment_conclusion(model_info, task)
        for conclusion in conclusions:
            st.markdown(f"- {conclusion}")

        st.success("✅ 实验报告生成完成！")

# ==================== 【新增】一键导出 Markdown 报告 ====================
st.divider()
st.markdown("## 📥 导出实验报告（Markdown）")

if st.button("💾 下载完整实验报告（.md）", use_container_width=True):
    name = st.session_state.get("student_name", "未命名")
    stu_id = st.session_state.get("student_id", "")
    cls = st.session_state.get("student_class", "")

    if not name or not stu_id:
        st.error("❌ 请先填写姓名和学号！")
    else:
        md_content = f"""# 数据挖掘实验报告
**班级**：{cls}
**姓名**：{name}
**学号**：{stu_id}
**实验时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 一、实验信息
- 任务类型：{task}
- 使用算法：{model_info.get('算法', '未知')}

## 二、数据洞察
"""
        if df is not None:
            for i in IntelligentReportGenerator.generate_data_insights(df, X, y, task):
                md_content += f"- {i}\n"

        md_content += "\n## 三、算法原理\n"
        exp = IntelligentReportGenerator.generate_algorithm_explanation(model_info.get('算法', ''), task, model_info)
        for v in exp.values():
            md_content += f"{v}\n\n"

        md_content += "## 四、模型性能\n"
        for a in IntelligentReportGenerator.generate_model_performance_analysis(model_info, task):
            md_content += f"- {a}\n"

        md_content += "\n## 五、优化建议\n"
        for s in IntelligentReportGenerator.generate_optimization_suggestions(model_info, task):
            md_content += f"- {s}\n"

        md_content += "\n## 六、实验结论\n"
        for c in IntelligentReportGenerator.generate_experiment_conclusion(model_info, task):
            md_content += f"- {c}\n"

        b64 = base64.b64encode(md_content.encode("utf-8")).decode()
        href = f'<a href="data:text/markdown;base64,{b64}" download="实验报告_{name}.md">📥 点击下载 Markdown 报告</a>'
        st.markdown(href, unsafe_allow_html=True)
        st.success("✅ 报告已生成，点击上方链接下载！")
