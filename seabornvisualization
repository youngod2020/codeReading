# 출처: http://growthj.link/python-seaborn-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%8B%9C%EA%B0%81%ED%99%94-%EC%B4%9D%EC%A0%95%EB%A6%AC/

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[17]:


import warnings
warnings.filterwarnings('ignore')


# In[18]:


titanic  = sns.load_dataset('titanic')
titanic


# In[19]:


tips = sns.load_dataset('tips')
tips


# # 1. count plot

# In[20]:


'''
항목별 갯수를 세어주는 countplot
해당 columns을 구성하고 있는 value들을 구분해서 보여줌
'''


# In[21]:


sns.set_style('whitegrid')

sns.countplot(x = "class", hue = "who", data=titanic)
plt.show


# In[22]:


sns.countplot(y="class", hue="who", data=titanic)
plt.show()


# In[23]:


sns.countplot(x = "class", hue = "who", palette = "Accent", data=titanic)
plt.show()


# In[ ]:





# # 2. distplot

# In[24]:


'''
matplotlib의 hist 그래프와 kdeplot을 통합한 그래프임
분포와 밀도를 확인할 수 있음
'''


# In[25]:


x = np.random.randn(100)
x


# In[26]:


sns.distplot(x)
plt.show()


# In[ ]:





# In[27]:


'''
rug는 rugplot이라고 불리며, 데이터 위치를 x축 위에 작은 선분(rug)로 나타냄
데이터의 위치 및 분포를 보여줌
'''


# In[28]:


sns.distplot(x, rug = True, hist = False, kde = True)
plt.show()


# In[ ]:





# In[ ]:


'''
kde(kernel density)plot
kde는 histogram보다 부드러운 형태의 분포 곡선을 보여줌
'''


# In[29]:


sns.distplot(x, rug = False, hist = False, kde = True)
plt.show()


# In[ ]:





# In[30]:


sns.distplot(x, vertical=True)


# In[ ]:





# In[31]:


sns.distplot(x, color = 'y')
plt.show()


# In[ ]:





# # 3. heatmap

# In[32]:


'''
색상으로 표현할 수 있는 다양한 정보를 일정한 이미지 위해 
열분포형태로 비쥬얼한 그래픽으로 출력 가능
'''


# In[33]:


heatmap_data = np.random.rand(10, 12)
sns.heatmap(heatmap_data, annot=True)
plt.show()


# In[ ]:





# In[37]:


tips


# In[38]:


pivot = tips.pivot_table(index = 'day', columns = 'size', values = 'tip')
pivot


# In[41]:


sns.heatmap(pivot, cmap = "Blues", annot=True)
plt.show()


# In[ ]:





# In[ ]:





# In[42]:


titanic.corr()


# In[43]:


sns.heatmap(titanic.corr(), annot=True, cmap = "YlGnBu")
plt.show()


# In[ ]:





# In[48]:


tips.head()


# In[ ]:





# # pairplot

# In[45]:


sns.pairplot(tips)
plt.show()


# In[46]:


sns.pairplot(tips, hue='size')
plt.show()


# In[ ]:





# In[ ]:





# # violinplot

# In[49]:


sns.violinplot(x = tips["total_bill"])
plt.show()


# In[ ]:





# In[50]:


sns.violinplot(x= "day", y="total_bill", data=tips)
plt.show()


# In[ ]:





# In[51]:


sns.violinplot(x = "day", y="total_bill", hue="smoker", data=tips, palette = "muted", split= True)


# In[ ]:





# # lmplot

# In[ ]:


'''
column 간의 선형관계를 확인하기 위한 차트
outlier 짐작가능
'''


# In[52]:


sns.lmplot(x = "total_bill", y="tip", height=8, data=tips)
plt.show()


# In[53]:


sns.lmplot(x = "total_bill", y = "tip", hue = "smoker", height=8, data=tips)
plt.show()


# In[ ]:





# In[54]:


sns.lmplot(x = "total_bill", y = "tip", hue="smoker", col = "day", col_wrap = 2, height = 6, data=tips)


# In[ ]:





# In[ ]:


# dataframe.groupby() : array로 여러개 가능
# sns.countplot() : hue
# dataframe.crosstab() : 표생성. 세로필드, 가로필드. (세로필드를 array로 어러개 가능)
# sns.factorplot(x,y,hue) : y평균값 그래프. hue로 지정된 필드의 종류만큼 라인이 그려짐.
# sns.violinplot(x,y,hue) : y값의 범위와 분포를 알 수 있는 그래프.


# In[ ]:




