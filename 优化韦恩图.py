# 可视化LASSO和多元Logistic回归筛选出的VIF小于5的特征
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# 定义LASSO筛选出的VIF小于5的特征
lasso_features = ['CRRT', 'Renal infection', 'Diabetes', 'Others', 'Urine output', 'SpO2(min)', 'GCS', 'Ventilation(NI)', 'APS III', 'WBC(min)', 'Lactate(min)', 'DBP mean(NI)', 'SpO2(max)', 'Charlson', 'SOFA', 'Sodium(max)', 'Hosp day', 'Resperate rate(max)', 'Ventilation hour(NI)', 'SIRS', 'Platelet(min)', 'Malignancy', 'ICU day', 'Ventilation(AI)', 'Heart rate(mean)', 'Liver infection', 'OASIS', 'SAPS II', 'Aniongap(min)', 'DBP max(NI)', 'Resperate rate(min)', 'CRRT day', 'Temperature(max)', 'Creatinine(max)', 'SBP min(NI)', 'BUN(min)']

# 定义多元Logistic回归筛选出的VIF小于5的特征
logistic_features = ['CRRT', 'Coagulation infection', 'Renal infection', 'Invase ventilation', 'Diabetes', 'Others', 'Urine output', 'SpO2(min)', 'GCS', 'Ventilation(NI)', 'APS III', 'Hemoglobin(max)', 'Lactate(min)', 'SpO2(max)', 'Charlson', 'SOFA', 'Hosp day', 'SIRS', 'Malignancy', 'ICU day', 'Ventilation(AI)', 'Heart rate(mean)', 'Liver infection', 'Temperature(mean)', 'OASIS', 'DBP max(NI)', 'Resperate rate(min)', 'CRRT day', 'CNS infection', 'Temperature(max)', 'DBP mean(NI)', 'SBP min(NI)', 'Cardiovascular infection']

# 创建集合用于求交集
logistic_set = set(logistic_features)  # 多元Logistic特征集合
lasso_set = set(lasso_features)    # Lasso特征集合
intersection = logistic_set.intersection(lasso_set)  # 两个集合的交集
logistic_only = logistic_set - intersection  # 仅Logistic选择的特征
lasso_only = lasso_set - intersection    # 仅Lasso选择的特征

# 创建图
G = nx.Graph()

# 添加节点和边
for feature in logistic_only:    
    G.add_edge('Logistic', feature, color='#DB432C')  # 淡红色表示仅被Logistic选择的特征
for feature in lasso_only:    
    G.add_edge('Lasso', feature, color='#DCCD5B')  # 淡蓝色表示仅被Lasso选择的特征
for feature in intersection:    
    G.add_edge('Logistic', feature, color='#DB432C')  # 淡红色边连接交集特征到Logistic
    G.add_edge('Lasso', feature, color='#DCCD5B')  # 淡蓝色边连接交集特征到Lasso

# 获取边的颜色
edge_colors = [data['color'] for _, _, data in G.edges(data=True)]

# 设置节点的颜色
node_colors = []
for node in G.nodes():    
    if node == 'Logistic':        
        node_colors.append('#DB432C')  # Logistic节点淡红色
    elif node == 'Lasso':        
        node_colors.append('#DCCD5B')   # Lasso节点淡蓝色
    elif node in logistic_only:        
        node_colors.append('#DB432C')  # 仅被Logistic选择的特征淡红色
    elif node in lasso_only:        
        node_colors.append('#DCCD5B')   # 仅被Lasso选择的特征淡蓝色
    elif node in intersection:        
        node_colors.append('#C2ABC8')        # 交集特征节点用淡紫色表示

# 布局
pos = nx.spring_layout(G, seed=42, k=0.5)  # k 参数调整节点之间的距离

# 绘制图形
plt.figure(figsize=(18, 18))  # 增大图的尺寸
nx.draw_networkx(
    G,
    pos=pos,
    edge_color=edge_colors,
    node_color=node_colors,
    with_labels=True,
    node_size=2500,  # 调整节点大小
    font_size=12,  # 调整字体大小
    font_color='black',  # 字体颜色
    edgecolors='none'  # 移除节点边框
)
plt.title('Feature Selection by Multinomial Logistic Regression and Lasso', fontsize=20)  # 调整标题字体大小
plt.show()