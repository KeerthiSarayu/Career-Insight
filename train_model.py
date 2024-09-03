import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import pickle

# Load the dataset
dataset = pd.read_csv("EPICS_1000_dataset.csv")
# print(dataset.columns)
# Assuming 'CGPA' to 'Build live project' are the features
X = dataset[['CGPA', 'Did you do webdev during college time ?',
       'Are you good at Data analysis ?', 'reading and writing skills',
       'Are you a tech person ?', 'Were you in a non tech society ?', 
       'Are you good at coding ?', 'Have you developed mobile apps ?',
       'Are you good at communication ?',
       'Do you have specialization in security',
       'Have you ever handled large databases ?',
       'Do you have knowlege of statistics and data science?',        
       'Are you proficient in English ?', 'Have you ever managed some event?',
       'Do you write technical blogs ?', 'Are you into marketing ?',  
       'Are you a ML expert ?', 'Do you have a lot of connections ?', 
       'Have you ever built live project ?', 'Role']]

X = pd.DataFrame(X, columns=['CGPA', 'Done WebDev', 'Data Analysis', 'reading & writing skills', 'Tech person', 'In a non-tech society', 'Good at Coding', 'Developed mobile app', 'Good at communication', 'Specialization in security', 'Handled large databases', 'Have knowledge of Statistics and Data Science', 'Proficient in English', 'Managed an event', 'Wrote tech blogs', 'Like marketing', 'ML expert', 'Lot of connections', 'Build live project'])

# Assuming 'Suggested Job Role' is the target variable
y = dataset['Role']

# Impute missing values if any (you might need to modify this based on your actual data)
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Encode categorical labels
labelencoder = LabelEncoder()
y_encoded = labelencoder.fit_transform(y)

# Create and train the Decision Tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_imputed, y_encoded)

with open('label_encoder.pkl', 'wb') as le_file:
    pickle.dump(labelencoder, le_file)
    
# Save the trained model if needed
pickle.dump(clf,open('trained_model.pkl','wb'))
