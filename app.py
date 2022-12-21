import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import warnings

warnings.filterwarnings(action="ignore")

from sklearn.cluster import KMeans, AgglomerativeClustering, Birch
from sklearn.mixture import GaussianMixture
from matplotlib import cm
from sklearn.metrics import silhouette_samples, silhouette_score

st.set_page_config(page_title='Dasboard', page_icon=':bar_chart:', layout='wide')

data_frame = pd.read_csv("CC GENERAL.csv")

st.title(":bar_chart:  Credit Card Dataset for Clustering")
st.dataframe(data_frame)

st.markdown("""---""")

data_frame.loc[(data_frame['MINIMUM_PAYMENTS'].isnull()==True),'MINIMUM_PAYMENTS'] = data_frame['MINIMUM_PAYMENTS'].median()
data_frame.loc[(data_frame['CREDIT_LIMIT'].isnull()==True),'CREDIT_LIMIT'] = data_frame['CREDIT_LIMIT'].median()

our_data = data_frame.drop('CUST_ID', 1)
scaller = StandardScaler()
scaled_data = scaller.fit_transform(our_data)

imputed_data = pd.DataFrame(scaled_data, columns=our_data.columns)

plt.figure(figsize=(12, 12))
sns.heatmap(imputed_data.corr(), annot=True, cmap='coolwarm', xticklabels=imputed_data.columns,
            yticklabels=imputed_data.columns)


st.markdown("""---""")

st.title("Кластеризация : Проверка корреляции")

st.pyplot(plt)
plt.clf()

st.markdown("""---""")
st.title("K - средние")

def inertia_plot(clust, X, start = 2, stop = 20):
    inertia = []
    for x in range(start,stop):
        km = clust(n_clusters = x)
        labels = km.fit_predict(X)
        inertia.append(km.inertia_)
    plt.figure(figsize = (12,6))
    plt.plot(range(start,stop), inertia, marker = 'o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Inertia plot with K')
    plt.xticks(list(range(start, stop)))
    plt.show()
k = inertia_plot(KMeans, imputed_data)

st.pyplot(plt)
plt.clf()

st.markdown("""---""")
st.title("Очки за силуэт")



for x in range(2, 7):
    alg = KMeans(n_clusters = x, )
    label = alg.fit_predict(imputed_data)
    st.write('Silhouette-Score for', x,  'Clusters: ', silhouette_score(imputed_data, label))

st.markdown("""---""")
st.title("Силуэтные участки:")

def silh_samp_cluster(clust,  X, start=2, stop=7, metric = 'euclidean'):
    for x in range(start, stop):
        km = clust(n_clusters = x)
        y_km = km.fit_predict(X)
        cluster_labels = np.unique(y_km)
        n_clusters = cluster_labels.shape[0]
        silhouette_vals = silhouette_samples(X, y_km, metric = metric)
        y_ax_lower, y_ax_upper =0,0
        yticks = []
        for i, c in enumerate(cluster_labels):
            c_silhouette_vals = silhouette_vals[y_km == c]
            c_silhouette_vals.sort()
            y_ax_upper += len(c_silhouette_vals)
            color = cm.jet(float(i)/n_clusters)
            plt.barh(range(y_ax_lower, y_ax_upper),
                    c_silhouette_vals,
                    height=1.0,
                    edgecolor='none',
                    color = color)
            yticks.append((y_ax_lower + y_ax_upper)/2.)
            y_ax_lower+= len(c_silhouette_vals)

        silhouette_avg = np.mean(silhouette_vals)
        plt.axvline(silhouette_avg,
                   color = 'red',
                   linestyle = "--")
        plt.yticks(yticks, cluster_labels+1)
        plt.ylabel("cluster")
        plt.xlabel('Silhouette Coefficient')
        plt.title('Silhouette for ' + str(x) + " Clusters")
        plt.show()
silh = silh_samp_cluster(KMeans, imputed_data)
st.pyplot(plt)
plt.clf()

st.markdown("""---""")
st.title("Извлечение функций с помощью PCA")

st.markdown("<p style='font-size: 30px'>Метрики кластеризации</p>", unsafe_allow_html=True)

for y in range(2, 5):
    st.write("PCA with # of components: ", y)
    pca = PCA(n_components=y)
    data_p = pca.fit_transform(imputed_data)
    for x in range(2, 7):
        alg = KMeans(n_clusters = x)
        label = alg.fit_predict(data_p)
        st.write('Silhouette-Score for', x,  'Clusters: ', silhouette_score(data_p, label) , '       Inertia: ',alg.inertia_)


st.markdown("""---""")
st.title("Визуализация")
st.markdown('<p style="font-size: 30px">Иерархической кластеризации с помощью PCA (Birch)</p>', unsafe_allow_html=True)

data_p = pd.DataFrame(PCA(n_components = 2).fit_transform(imputed_data))
preds = pd.Series(Birch(n_clusters = 5,).fit_predict(data_p))
data_p = pd.concat([data_p, preds], axis =1)
data_p.columns = [0,1,'target']

fig = plt.figure(figsize=(20, 13))
colors = ['red', 'green', 'blue', 'purple', 'orange', 'brown']
plt.subplot(121)
plt.scatter(data_p[data_p['target']==0].iloc[:,0], data_p[data_p.target==0].iloc[:,1], c = colors[0], label = 'cluster 1')
plt.scatter(data_p[data_p['target']==1].iloc[:,0], data_p[data_p.target==1].iloc[:,1], c = colors[1], label = 'cluster 2')
plt.scatter(data_p[data_p['target']==2].iloc[:,0], data_p[data_p.target==2].iloc[:,1], c = colors[2], label = 'cluster 3')
plt.scatter(data_p[data_p['target']==3].iloc[:,0], data_p[data_p.target==3].iloc[:,1], c = colors[3], label = 'cluster 4')
plt.scatter(data_p[data_p['target']==4].iloc[:,0], data_p[data_p.target==4].iloc[:,1], c = colors[4], label = 'cluster 5')
plt.legend()
plt.title('Birch Clustering with 5 Clusters')
plt.xlabel('PC1')
plt.ylabel('PC2')

st.pyplot()
plt.clf()

data_p = pd.DataFrame(PCA(n_components = 2).fit_transform(imputed_data))
preds = pd.Series(KMeans(n_clusters = 6,).fit_predict(data_p))
data_p = pd.concat([data_p, preds], axis =1)
data_p.columns = [0,1,'target']

plt.subplot(122)
plt.scatter(data_p[data_p['target']==0].iloc[:,0], data_p[data_p.target==0].iloc[:,1], c = colors[0], label = 'cluster 1')
plt.scatter(data_p[data_p['target']==1].iloc[:,0], data_p[data_p.target==1].iloc[:,1], c = colors[1], label = 'cluster 2')
plt.scatter(data_p[data_p['target']==2].iloc[:,0], data_p[data_p.target==2].iloc[:,1], c = colors[2], label = 'cluster 3')
plt.scatter(data_p[data_p['target']==3].iloc[:,0], data_p[data_p.target==3].iloc[:,1], c = colors[3], label = 'cluster 4')
plt.scatter(data_p[data_p['target']==4].iloc[:,0], data_p[data_p.target==4].iloc[:,1], c = colors[4], label = 'cluster 5')
plt.scatter(data_p[data_p['target']==5].iloc[:,0], data_p[data_p.target==5].iloc[:,1], c = colors[5], label = 'cluster 6')
plt.legend()
plt.title('Birch Clustering with 6 Clusters')
plt.xlabel('PC1')
plt.ylabel('PC2')

st.pyplot()
plt.clf()

st.markdown("""---""")
st.title("Кластеризация k-средних c помощью PCA")

data_p = pd.DataFrame(PCA(n_components = 2).fit_transform(imputed_data))
preds = pd.Series(KMeans(n_clusters = 5,).fit_predict(data_p))
data_p = pd.concat([data_p, preds], axis =1)
data_p.columns = [0,1,'target']

fig = plt.figure(figsize=(20, 13))
colors = ['red', 'green', 'blue', 'purple', 'orange', 'brown']
plt.subplot(121)
plt.scatter(data_p[data_p['target']==0].iloc[:,0], data_p[data_p.target==0].iloc[:,1], c = colors[0], label = 'cluster 1')
plt.scatter(data_p[data_p['target']==1].iloc[:,0], data_p[data_p.target==1].iloc[:,1], c = colors[1], label = 'cluster 2')
plt.scatter(data_p[data_p['target']==2].iloc[:,0], data_p[data_p.target==2].iloc[:,1], c = colors[2], label = 'cluster 3')
plt.scatter(data_p[data_p['target']==3].iloc[:,0], data_p[data_p.target==3].iloc[:,1], c = colors[3], label = 'cluster 4')
plt.scatter(data_p[data_p['target']==4].iloc[:,0], data_p[data_p.target==4].iloc[:,1], c = colors[4], label = 'cluster 5')
plt.legend()
plt.title('KMeans Clustering with 5 Clusters')
plt.xlabel('PC1')
plt.ylabel('PC2')

st.pyplot()
plt.clf()

data_p = pd.DataFrame(PCA(n_components = 2).fit_transform(imputed_data))
preds = pd.Series(KMeans(n_clusters = 6,).fit_predict(data_p))
data_p = pd.concat([data_p, preds], axis =1)
data_p.columns = [0,1,'target']

plt.subplot(122)
plt.scatter(data_p[data_p['target']==0].iloc[:,0], data_p[data_p.target==0].iloc[:,1], c = colors[0], label = 'cluster 1')
plt.scatter(data_p[data_p['target']==1].iloc[:,0], data_p[data_p.target==1].iloc[:,1], c = colors[1], label = 'cluster 2')
plt.scatter(data_p[data_p['target']==2].iloc[:,0], data_p[data_p.target==2].iloc[:,1], c = colors[2], label = 'cluster 3')
plt.scatter(data_p[data_p['target']==3].iloc[:,0], data_p[data_p.target==3].iloc[:,1], c = colors[3], label = 'cluster 4')
plt.scatter(data_p[data_p['target']==4].iloc[:,0], data_p[data_p.target==4].iloc[:,1], c = colors[4], label = 'cluster 5')
plt.scatter(data_p[data_p['target']==5].iloc[:,0], data_p[data_p.target==5].iloc[:,1], c = colors[5], label = 'cluster 6')
plt.legend()
plt.title('KMeans Clustering with 6 Clusters')
plt.xlabel('PC1')
plt.ylabel('PC2')

st.pyplot()
plt.clf()

st.markdown("""---""")
st.title("Агломеративная иерархическая кластеризация с помощью PCA")

data_p = pd.DataFrame(PCA(n_components = 2).fit_transform(imputed_data))
preds = pd.Series(AgglomerativeClustering(n_clusters = 5,).fit_predict(data_p))
data_p = pd.concat([data_p, preds], axis =1)
data_p.columns = [0,1,'target']

fig = plt.figure(figsize=(20, 13))
colors = ['red', 'green', 'blue', 'purple', 'orange', 'brown']
plt.subplot(121)
plt.scatter(data_p[data_p['target']==0].iloc[:,0], data_p[data_p.target==0].iloc[:,1], c = colors[0], label = 'cluster 1')
plt.scatter(data_p[data_p['target']==1].iloc[:,0], data_p[data_p.target==1].iloc[:,1], c = colors[1], label = 'cluster 2')
plt.scatter(data_p[data_p['target']==2].iloc[:,0], data_p[data_p.target==2].iloc[:,1], c = colors[2], label = 'cluster 3')
plt.scatter(data_p[data_p['target']==3].iloc[:,0], data_p[data_p.target==3].iloc[:,1], c = colors[3], label = 'cluster 4')
plt.scatter(data_p[data_p['target']==4].iloc[:,0], data_p[data_p.target==4].iloc[:,1], c = colors[4], label = 'cluster 5')
plt.legend()
plt.title('Agglomerative Hierarchical Clustering with 5 Clusters')
plt.xlabel('PC1')
plt.ylabel('PC2')

st.pyplot()
plt.clf()

data_p = pd.DataFrame(PCA(n_components = 2).fit_transform(imputed_data))
preds = pd.Series(AgglomerativeClustering(n_clusters = 6,).fit_predict(data_p))
data_p = pd.concat([data_p, preds], axis =1)
data_p.columns = [0,1,'target']

plt.subplot(122)
plt.scatter(data_p[data_p['target']==0].iloc[:,0], data_p[data_p.target==0].iloc[:,1], c = colors[0], label = 'cluster 1')
plt.scatter(data_p[data_p['target']==1].iloc[:,0], data_p[data_p.target==1].iloc[:,1], c = colors[1], label = 'cluster 2')
plt.scatter(data_p[data_p['target']==2].iloc[:,0], data_p[data_p.target==2].iloc[:,1], c = colors[2], label = 'cluster 3')
plt.scatter(data_p[data_p['target']==3].iloc[:,0], data_p[data_p.target==3].iloc[:,1], c = colors[3], label = 'cluster 4')
plt.scatter(data_p[data_p['target']==4].iloc[:,0], data_p[data_p.target==4].iloc[:,1], c = colors[4], label = 'cluster 5')
plt.scatter(data_p[data_p['target']==5].iloc[:,0], data_p[data_p.target==5].iloc[:,1], c = colors[5], label = 'cluster 6')
plt.legend()
plt.title('Agglomerative Hierarchical Clustering with 6 Clusters')
plt.xlabel('PC1')
plt.ylabel('PC2')

st.pyplot()
plt.clf()

st.markdown("""---""")
st.title("Кластеризация гауссовой смеси с помощью PCA")

data_p = pd.DataFrame(PCA(n_components = 2).fit_transform(imputed_data))
preds = pd.Series(GaussianMixture(n_components = 5,).fit_predict(data_p))
data_p = pd.concat([data_p, preds], axis =1)
data_p.columns = [0,1,'target']

fig = plt.figure(figsize=(20, 13))
colors = ['red', 'green', 'blue', 'purple', 'orange', 'brown']
plt.subplot(121)
plt.scatter(data_p[data_p['target']==0].iloc[:,0], data_p[data_p.target==0].iloc[:,1], c = colors[0], label = 'cluster 1')
plt.scatter(data_p[data_p['target']==1].iloc[:,0], data_p[data_p.target==1].iloc[:,1], c = colors[1], label = 'cluster 2')
plt.scatter(data_p[data_p['target']==2].iloc[:,0], data_p[data_p.target==2].iloc[:,1], c = colors[2], label = 'cluster 3')
plt.scatter(data_p[data_p['target']==3].iloc[:,0], data_p[data_p.target==3].iloc[:,1], c = colors[3], label = 'cluster 4')
plt.scatter(data_p[data_p['target']==4].iloc[:,0], data_p[data_p.target==4].iloc[:,1], c = colors[4], label = 'cluster 5')
plt.legend()
plt.title('Gaussian Mixture Clustering with 5 Clusters')
plt.xlabel('PC1')
plt.ylabel('PC2')

st.pyplot()
plt.clf()


data_p = pd.DataFrame(PCA(n_components = 2).fit_transform(imputed_data))
preds = pd.Series(GaussianMixture(n_components = 6,).fit_predict(data_p))
data_p = pd.concat([data_p, preds], axis =1)
data_p.columns = [0,1,'target']

plt.subplot(122)
plt.scatter(data_p[data_p['target']==0].iloc[:,0], data_p[data_p.target==0].iloc[:,1], c = colors[0], label = 'cluster 1')
plt.scatter(data_p[data_p['target']==1].iloc[:,0], data_p[data_p.target==1].iloc[:,1], c = colors[1], label = 'cluster 2')
plt.scatter(data_p[data_p['target']==2].iloc[:,0], data_p[data_p.target==2].iloc[:,1], c = colors[2], label = 'cluster 3')
plt.scatter(data_p[data_p['target']==3].iloc[:,0], data_p[data_p.target==3].iloc[:,1], c = colors[3], label = 'cluster 4')
plt.scatter(data_p[data_p['target']==4].iloc[:,0], data_p[data_p.target==4].iloc[:,1], c = colors[4], label = 'cluster 5')
plt.scatter(data_p[data_p['target']==5].iloc[:,0], data_p[data_p.target==5].iloc[:,1], c = colors[5], label = 'cluster 6')
plt.legend()
plt.title('Gaussian Mixture Clustering with 6 Clusters')
plt.xlabel('PC1')
plt.ylabel('PC2')

st.pyplot()
plt.clf()

st.markdown("""---""")
st.title("Исследовательский анализ данных")
st.markdown('<p style="font-size: 30px">Кластер 0 (синий): скромные пользователи</p>', unsafe_allow_html=True)


best_cols = ["BALANCE", "PURCHASES", "CASH_ADVANCE", "CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS"]


data_final = pd.DataFrame(imputed_data[best_cols])

st.write('New dataframe with best columns has just been created. Data shape: ' + str(data_final.shape))

alg = KMeans(n_clusters = 6)
label = alg.fit_predict(data_final)
data_final['cluster'] = label
best_cols.append('cluster')
cl_0 = sns.pairplot(data_final[best_cols], hue='cluster', x_vars=['PURCHASES', 'PAYMENTS', 'CREDIT_LIMIT'], y_vars=['cluster'], height=5, aspect=1)
st.pyplot(cl_0)
plt.clf()

st.markdown('<p style="font-size: 30px">Кластер 1 (оранжевый): активные пользователи</p>', unsafe_allow_html=True)
cl_1 = sns.pairplot(data_final[best_cols], hue='cluster', x_vars=['PURCHASES', 'PAYMENTS', 'CREDIT_LIMIT'], y_vars=['cluster'], height=5, aspect=1)

st.pyplot(cl_1)
plt.clf()

st.markdown('<p style="font-size: 30px">Кластер 2 (зеленый): богатые пользователи</p>', unsafe_allow_html=True)
cl_2 = sns.pairplot(data_final[best_cols], hue='cluster', x_vars=['PURCHASES', 'PAYMENTS', 'CREDIT_LIMIT'], y_vars=['cluster'], height=5, aspect=1)


st.pyplot(cl_2)
plt.clf()

st.markdown('<p style="font-size: 30px">На сколько сильно отличается кластер</p>', unsafe_allow_html=True)
check_1 = sns.pairplot(data_final[best_cols], hue='cluster', x_vars=['PURCHASES'], y_vars=['PAYMENTS'], height=5, aspect=1)
st.pyplot(check_1)
plt.clf()

st.markdown('<p style="font-size: 30px">Кластер 3 (красный): заемщики денег</p>', unsafe_allow_html=True)
cl_3 = sns.pairplot(data_final[best_cols], hue='cluster', x_vars=['BALANCE', 'CASH_ADVANCE', 'PAYMENTS'], y_vars=['cluster'], height=5, aspect=1)
st.pyplot(cl_3)
plt.clf()

st.markdown('<p style="font-size: 30px">Кластер 4 (фиолетовый): люди с высоким риском</p>', unsafe_allow_html=True)
cl_4 = sns.pairplot(data_final[best_cols], hue='cluster', x_vars=['MINIMUM_PAYMENTS'], y_vars=['CREDIT_LIMIT'], height=5, aspect=1)
st.pyplot(cl_4)
plt.clf()

st.markdown('<p style="font-size: 30px">Кластер 5 (коричневый) Сложные для анализа люди</p>', unsafe_allow_html=True)
cl_5 = sns.pairplot(data_final[best_cols], hue='cluster', x_vars=['BALANCE'], y_vars=['CREDIT_LIMIT'],
            height=5, aspect=1)
st.pyplot(cl_5)
plt.clf()

st.title("Вывод")
st.write("Мы многому научились из этого набора данных, сегментируя клиентов на шесть меньших групп: средний пользователь, активные пользователи, крупные спонсоры, заемщики денег, лица с высоким риском и подстановочные знаки. Чтобы завершить этот кластерный анализ, давайте подведем итог тому, что мы узнали, и некоторым возможным маркетинговым стратегиям::")
st.markdown("- Обычный Джо не часто пользуется кредитными картами в повседневной жизни. У них средние финансы и низкие долги. Хотя поощрение этих людей к использованию кредитных карт необходимо для получения прибыли компании, следует также учитывать этику ведения бизнеса и социальную ответственность.")
st.markdown("- Определите активных клиентов, чтобы применить к ним правильную маркетинговую стратегию. Эти люди - основная группа, на которой мы должны сосредоточиться.")
st.markdown("- Некоторые люди просто плохо умеют управлять финансами - например, Заемщики. К этому нельзя относиться легкомысленно.")
st.markdown("- Несмотря на то, что в настоящее время мы хорошо справляемся с рисками, предоставляя им низкие кредитные лимиты, следует рассмотреть дополнительные маркетинговые стратегии, ориентированные на эту группу клиентов.")
st.write(" В этом проекте мы выполнили предварительную обработку данных, извлечение признаков с помощью PCA, изучили различные метрики кластеризации (инерции, оценки силуэтов), экспериментировали с различными алгоритмами кластеризации (KMeans Clustering, Agglomerative Hierarchical Clustering, Gaussian Mixture Clustering), визуализацией данных и бизнесом, аналитикой.")
st.markdown('''
<style>
[data-testid="stMarkdownContainer"] ul{
    padding-left:40px;
}
</style>
''', unsafe_allow_html=True)