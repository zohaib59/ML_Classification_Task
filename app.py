
import numpy as np
import seaborn as sns
import os
import pylab as pl
import matplotlib.pyplot as plt

os.chdir("C:\\Users\\zohaib khan\\OneDrive\\Desktop\\USE ME\\dump\\zk")

data = pd.read_csv("email.csv")

data.head()

pd.set_option('display.max_columns', None)

data.head()

#Check for missing values
data.isnull().sum()

#Check for duplicate values
data[data.duplicated()]

#remove duplicates permanent
data.drop_duplicates(inplace=True)

# To show Outliers in the data set run the code 

num_vars = data.select_dtypes(include=['int','float']).columns.tolist()

num_cols = len(num_vars)
num_rows = (num_cols + 2 ) // 3
fig, axs = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5*num_rows))
axs = axs.flatten()

for i, var in enumerate (num_vars):
    sns.boxplot(x=data[var],ax=axs[i])
    axs[i].set_title(var)

if num_cols < len(axs):
  for i in range(num_cols , len(axs)):
    fig.delaxes(axs[i])

plt.tight_layout()
plt.show()


def pintu (data,age):
 Q1 = data[age].quantile(0.25)
 Q3 = data[age].quantile(0.75)
 IQR = Q3 - Q1
 data= data.loc[~((data[age] < (Q1 - 1.5 * IQR)) | (data[age] > (Q3 + 1.5 * IQR))),]
 return data

data.boxplot(column=["num_urgent_keywords"])

data = pintu(data,"num_urgent_keywords")

from autoviz.AutoViz_Class import AutoViz_Class 
AV = AutoViz_Class()
import matplotlib.pyplot as plt
%matplotlib INLINE
filename = 'email.csv'
sep =","
dft = AV.AutoViz(
    filename  
)




def class_distribution(data, column_name='label'):
    # Display total counts and percentage for each class
    distribution = data[column_name].value_counts()
    percentage = data[column_name].value_counts(normalize=True) * 100
    
    print(f"Class distribution for '{column_name}':")
    print(distribution)
    print("\nPercentage distribution:")
    print(percentage)

# Call the function to display the distribution for the 'Resigned' column
class_distribution(data, 'label')




#Segregrating dataset into X and y

X = data.drop("label", axis = 1)

y = data["label"]

X.head()

y.head()

#This will resample the dependent variable using the SMOTE

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
smote = SMOTE(random_state=42) 
X_resampled,y_resampled = smote.fit_resample(X,y)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size = 0.3, random_state = 20)


from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=0)
X_train_over, y_train_over = smote.fit_resample(X_train, y_train)

print("Before SMOTE: ", X_train.shape, y_train.shape)
print("After SMOTE: ", X_train_over.shape, y_train_over.shape)
print("After SMOTE Label Distribution: ", pd.Series(y_train_over).value_counts())

#Scale the dataset
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



# Import required libraries
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


# Initialize classifiers
log_model = LogisticRegression()
rf_model = RandomForestClassifier()
knn_model = KNeighborsClassifier()
gnb_model = GaussianNB()
gbc_model = GradientBoostingClassifier()
ada_model = AdaBoostClassifier()
xgb_model = XGBClassifier()
mlp_model = MLPClassifier()

# Fitting models
log_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)
gnb_model.fit(X_train, y_train)
gbc_model.fit(X_train, y_train)
ada_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)
mlp_model.fit(X_train, y_train)

# Importing model testing libraries
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define a function to evaluate models
def evaluate_model(model, model_name):
    y_pred = model.predict(X_test)
    print(model_name)
    print(f"Training data accuracy: {model.score(X_train, y_train)}")
    print(f"Testing data accuracy: {model.score(X_test, y_test)}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted')}")
    print(f"Recall: {recall_score(y_test, y_pred, average='weighted')}")
    print(f"F1 score: {f1_score(y_test, y_pred, average='weighted')}\n")

# Evaluate each model
evaluate_model(log_model, "Logistic Regression")
evaluate_model(rf_model, "Random Forest")
evaluate_model(knn_model, "K Nearest Neighbors")
evaluate_model(gnb_model, "Gaussian Naive Bayes")
evaluate_model(gbc_model, "Gradient Boosting")
evaluate_model(ada_model, "AdaBoost")
evaluate_model(xgb_model, "XGBoost")
evaluate_model(mlp_model, "Neural Network (MLP)")









