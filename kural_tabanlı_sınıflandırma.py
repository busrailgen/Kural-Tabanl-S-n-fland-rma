###############################################################
# Kural Tabanlı Sınıflandırma ile Potansiyel Müşteri Getirisi Hesaplama
###############################################################

import numpy as np
import pandas as pd
import seaborn as sns

###############################################################
# 1. İş Problemi (Business Problem)
###############################################################

#Bir oyun şirketi müşterilerinin bazı özelliklerini kullanarak seviye tabanlı(levelbased) yeni müşteri tanımları(persona) oluşturmak
#ve bu yeni müşteri tanımlarına göre segmentler oluşturup
#bu segmentlere göre yeni gelebilecek müşterilerin şirkete ortalama ne kadar kazandırabileceğini tahmin etmek istemektedir.


#############################################
# GÖREV 1: Aşağıdaki soruları yanıtlayınız.
#############################################
# Soru 1: persona.csv dosyasını okutunuz ve veri seti ile ilgili genel bilgileri gösteriniz.
df = pd.read_csv("datasets/persona.csv")
df.head()
df.shape
df.info()

# Soru 2: Kaç unique SOURCE vardır? Frekansları nedir?
df["SOURCE"].nunique()
df["SOURCE"].value_counts()

# Soru 3: Kaç unique PRICE vardır?
df["PRICE"].nunique()
# Soru 4: Hangi PRICE'dan kaçar tane satış gerçekleşmiş?
df["PRICE"].value_counts()

# Soru 5: Hangi ülkeden kaçar tane satış olmuş?
df["COUNTRY"].value_counts()
df.groupby("COUNTRY")["PRICE"].count()

# Soru 6: Ülkelere göre satışlardan toplam ne kadar kazanılmış?
df.groupby("COUNTRY")["PRICE"].sum()
df.groupby("COUNTRY").agg({"PRICE": "sum"})

# Soru 7: SOURCE türlerine göre göre satış sayıları nedir?
df["SOURCE"].value_counts()

# Soru 8: Ülkelere göre PRICE ortalamaları nedir?
df.groupby("COUNTRY").agg({"PRICE": "mean"})

# Soru 9: SOURCE'lara göre PRICE ortalamaları nedir?
df.groupby("SOURCE").agg({"PRICE": "mean"})

# Soru 10: COUNTRY-SOURCE kırılımında PRICE ortalamaları nedir?
df.groupby(["COUNTRY", "SOURCE"]).agg({"PRICE": "mean"})



#############################################
# GÖREV 2: COUNTRY, SOURCE, SEX, AGE kırılımında ortalama kazançlar nedir?
#############################################
df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"}).head()


#############################################
# GÖREV 3: Çıktıyı PRICE'a göre sıralayınız.
#############################################

agg_df = df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"}).sort_values("PRICE", ascending=False)
agg_df.head()

#############################################
# GÖREV 4: Indekste yer alan isimleri değişken ismine çeviriniz.
#############################################
agg_df = agg_df.reset_index()
agg_df.head()

#############################################
# GÖREV 5: AGE değişkenini kategorik değişkene çeviriniz ve agg_df'e ekleyiniz.
#############################################

bins = [0, 18, 23, 30, 40, agg_df["AGE"].max()]
mylabels = ['0_18', '19_23', '24_30', '31_40', '41_' + str(agg_df["AGE"].max())]
agg_df["age_cat"] = pd.cut(agg_df["AGE"], bins, labels=mylabels)
agg_df.head()

#############################################
# GÖREV 6: Yeni level based müşterileri tanımlayınız ve veri setine değişken olarak ekleyiniz.
#############################################
agg_df['customers_level_based'] = agg_df[['COUNTRY', 'SOURCE', 'SEX', 'age_cat']].agg(lambda x: '_'.join(x).upper(), axis=1)
agg_df.head()
# YÖNTEM 2
agg_df["customers_level_based"] = ['_'.join(i).upper() for i in agg_df.drop(["AGE", "PRICE"], axis=1).values]


agg_df = agg_df[["customers_level_based", "PRICE"]]
agg_df.head()

agg_df["customers_level_based"].value_counts()

agg_df.groupby("customers_level_based").agg({"PRICE": "mean"})
agg_df = agg_df.groupby("customers_level_based").agg({"PRICE": "mean"})

agg_df.head()
agg_df = agg_df.reset_index()
agg_df.head()

agg_df["customers_level_based"].value_counts()


#############################################
# GÖREV 7: Yeni müşterileri (USA_ANDROID_MALE_0_18) segmentlere ayırınız.
#############################################
agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 4, labels= ["D", "C", "B", "A"])
agg_df.head()
agg_df.tail()
agg_df.groupby("SEGMENT").agg({"PRICE": ["mean", "max", "sum"]})

#############################################
# GÖREV 8: Yeni gelen müşterileri sınıflandırınız ne kadar gelir getirebileceğini tahmin ediniz.
#############################################
# 33 yaşında ANDROID kullanan bir Türk kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?
new_user = "TUR_ANDROID_FEMALE_31_40"
agg_df[agg_df["customers_level_based"] == new_user]

# 35 yaşında IOS kullanan bir Fransız kadını hangi segmente ve ortalama ne kadar gelir kazandırması beklenir?
new_user = "FRA_IOS_FEMALE_31_40"
agg_df[agg_df["customers_level_based"] == new_user]


