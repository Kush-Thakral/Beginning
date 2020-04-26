#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#loading the dataset
df = pd.read_csv(r"C:\Users\kusht\Downloads\ecommerce_sample_dataset.csv")


# In[3]:


df.head()


# In[4]:


k=0
for i in range(0,20000) :
    if df.at[i,'product_rating']==df.at[i,'overall_rating'] : 
        k=k+1


# In[5]:


print(k)


# In[6]:


#Since k=20,000 we know for all data overall_rating=product_rating so we can drop one column
df.drop("product_rating",axis=1,inplace=True)


#  Since We dont have any information about the user data , We'll be using <b> Content Based Filtering </b> in which we'll find the importance of each word in dataset using <b><i> TF-IDF Vector </b></i> and then find five most similar product using <i> Cosine Similarity </i> and <i> Euclidean Distance </i> 

# In[7]:


#Using Tf-Idf vectorizer and then finding similarity scores using cosine similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel


# In[8]:


#initialising vectorizer
vectorizer = TfidfVectorizer(analyzer='word')


# We only need information about categorical data like product name and their category tree to find the importance of words so we'll be dropping unnecessary columns

# In[9]:


#dropping unnecessary columns
df_new=df.drop(['crawl_timestamp','product_url','retail_price','discounted_price','image','is_FK_Advantage_product','product_specifications'],axis=1)


# In[10]:


df_new.head()


# Seeing the data we understand theres more information in the product_category_tree that will be beneficial for the recommender system so we'll extract this information , firstly we'll make it in readable form

# In[11]:


df_new.product_category_tree=df_new.product_category_tree.apply(lambda x: x.replace(">>",","))
df_new.product_category_tree=df_new.product_category_tree.apply(lambda x: x.strip('['))
df_new.product_category_tree=df_new.product_category_tree.apply(lambda x: x.strip(']'))
df_new.product_category_tree=df_new.product_category_tree.apply(lambda x: x.replace(" , ",","))
df_new.product_category_tree=df_new.product_category_tree.apply(lambda x: x.replace(" & ",","))
df_new.product_category_tree=df_new.product_category_tree.apply(lambda x: x.replace("'s",""))
df_new.product_category_tree=df_new.product_category_tree.apply(lambda x: x.replace("and",","))


# In[12]:


df_new.product_category_tree


# 
# It's also visible that there is some information like brand and exact product type is not there in some of the elements of product_category_tree for example , some elements have category_tree as jewelery,rings but does not tell what type of ring it is like 18K gold, the information which is there in product_name . To handle that we'll make a new column NameTree where information of both product_name and product_category_tree would be added and we'll train our Tf-Idf vector on that feature
# 
# 

# In[13]:


for i in range(0,20000) : 
    if (df_new.at[i,'product_category_tree'].find(df_new.at[i,'product_name']))==-1 : df_new.at[i,'NameTree']=df_new.at[i,'product_category_tree'] + ',{}'.format(df_new.at[i,'product_name']) 
    else : df_new.at[i,'NameTree']=df_new.at[i,'product_category_tree']


# In[14]:


for i in df_new.NameTree : 
    print(i)


# So We've added product_name to those product_category_tree queries who already didnt have product_name in it. Now all elements of the new column NameTree has both the information of product_name and product_category_tree

# In[15]:


#More Data Cleaning 
df_new.NameTree=df_new.NameTree.apply(lambda x: x.replace('"',''))


# In[16]:


df_new.NameTree


# In[17]:


for i in df_new.NameTree :
    print(i)


# Now we'll fit the Tf-Idf vector in the column NameTree

# In[18]:


#build NameTree tfidf matrix
tfidf_matrix = vectorizer.fit_transform(df_new['NameTree'])
tfidf_feature_name= vectorizer.get_feature_names()

tfidf_matrix.shape


# In[28]:


# cosine similarity matrix using linear_kernal of sklearn
cosine_similarity_new = linear_kernel(tfidf_matrix, tfidf_matrix)
df_new=df_new.reset_index(drop=True)
indices=pd.Series(df_new['NameTree'].index)


#Function to get the most similar products
def recommend(index, method):
    id = indices[index]
    # Get the pairwise similarity scores of all products compared to that product,
    # sorting them and getting top 5
    similarity_scores = list(enumerate(method[id]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    #Removing duplicates having same similarity scores so recommender system doesnt print all outputs as same
    final_scores = []
    for i in range(1,20000) :
         if similarity_scores[i][1]!=similarity_scores[i-1][1] : final_scores.append(similarity_scores[i-1])
    final_scores = final_scores[1:6]
    
    #Get the products index
    products_index = [i[0] for i in final_scores]
    
    print("Top 5 recommendations of {} are:\n ".format(df_new.at[id,'product_name']))
    for i in range(0,5) : 
        print(str(i+1)+ " {}\n".format(df_new.at[products_index[i],'product_name']))
              
    #Return the top 5 most similar products using integar-location based indexing (iloc)
    #print('Top 5 recommendations are: \n1) {0} \n2) {1} \n3) {2} \n4) {3} \n5) {4}'.format(df_new.at[final_scores[0][0],'product_name'],df_new.at[final_scores[1][0],'product_name'],df_new.at[final_scores[2][0],'product_name'],df_new.at[final_scores[3][0],'product_name'],df_new.at[final_scores[4][0],'product_name']))
    #return df_new.product_name.iloc[products_index]


# In[29]:


recommend(3,cosine_similarity_new)


# The above function was used to handle duplicates. for example , Alisha Solid Womens Cycling shorts have many duplicates and all will have a similarity score of 1 so recommneder system is likely to print all results as same and equal to Alisha Solid Womens Cycling shorts so we first sorted the similarity scores and then removed the dame similarity scores

# Now we'll try the same thing with similarity as <b> <i> Euclidean </b></i>

# In[30]:


from sklearn.metrics.pairwise import euclidean_distances
D = euclidean_distances(tfidf_matrix)


# In[31]:


def recommend_euclidean_distance(j):
    ind = j
    distance = list(enumerate(D[ind]))
    distance = sorted(distance, key=lambda x: x[1])
    final_distance = []
    for i in range(1,20000) :
         if distance[i][1]!=distance[i-1][1] : final_distance.append(distance[i-1])
    final_distance = final_distance[1:6]
    
     #Get the products index
    products_index = [i[0] for i in final_distance]
    
    print("Top 5 recommendations of {} are:\n ".format(df_new.at[ind,'product_name']))
    for i in range(0,5) : 
        print(str(i+1) + " {}\n".format(df_new.at[products_index[i],'product_name']))
   
    #Get the books index
    #books_index = [i[0] for i in distance]

    #Return the top 5 most similar books using integar-location based indexing (iloc)
    #return df3.product_name.iloc[books_index]


# In[32]:


recommend_euclidean_distance(3)


# In[33]:


recommend(3,cosine_similarity_new)


# As is clear from the above example both the cosine similarity and euclidean similarity gives the same result and thus we can finalise any one of the two models. We'll be taking the cosine similarity model. Few predictions of the above model are: 

# In[34]:


recommend(100,cosine_similarity_new)


# In[35]:


recommend(1000,cosine_similarity_new)


# In[37]:


recommend(500,cosine_similarity_new)


# In[40]:


recommend(1750,cosine_similarity_new)


# In[41]:


recommend(50,cosine_similarity_new)


# <b><i> Thank you </b></i>
