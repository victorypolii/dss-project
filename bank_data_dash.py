import pickle
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

#read model
data_bank_model = pickle.load(open('bank_data_model.sav', 'rb'))
df=pd.read_csv('bank-full.csv', sep=';')
data_bersih=pd.read_csv('data_baru.csv', sep=',')
x=data_bersih[['age', 'job', 'marital', 'education', 'balance', 'pdays', 'previous', 'loan']]
y=data_bersih['y']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, max_depth=10, max_features=5, random_state=45)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

from streamlit_option_menu import option_menu
#navigasi sidebar
with st.sidebar :
    selected = option_menu ('Predict if the client will subscribe a term deposit',
    ['Home','Visualisasi',
     'Prediksi','Model'],
    default_index=0)

if (selected=='Home') :
    st.title('Selamat Datang di Web Analisis Nasabah Bank')
    st.write("Develop by Kelompok 3 DSS")
    st.write("1. Stevi Stevanus")
    st.write("2. Victory Polii")
    st.write("3. Arwin Pangaila")
    st.write("4. Rycko Frans")
    st.write("5. Hendry Tangkuman")

if (selected=='Visualisasi') :
    st.title('Tampilan Grafik')
    
    
    #hitung jumlah usia
    count_age = df["age"].value_counts().sort_index()
    col1, col2 = st.columns(2)
    # tampilkan histogram usia
    fig_age, ax = plt.subplots()
    ax.bar(count_age.index, count_age.values)
    ax.set_xlabel("Age")
    ax.set_ylabel("Nasabah")
    ax.set_title("Histogram of Age")
    #st.pyplot(fig_age)
    
    # membuat tabel distribusi usia
    usia_count = df['age'].value_counts()
    usia_count = pd.DataFrame({'Usia':usia_count.index, 'Jumlah':usia_count.values})
    #st.write('Distribusi Usia Nasabah:')
    #st.write(usia_count)
        
    #hitung jumlah pekerjaan
    count_job=df["job"].value_counts().sort_index()
    fig_job, ax= plt.subplots()
    ax.bar(count_job.index, count_job.values)
    ax.set_xlabel("Job")
    ax.set_ylabel("Nasabah")
    ax.set_title("Bar Chart Job Distribution")
    ax.set_xticklabels(count_job.index, rotation=90)
    #st.pyplot(fig_job)

    job_count = df['job'].value_counts()
    job_count = pd.DataFrame({'Pekerjaan':job_count.index, 'Jumlah':job_count.values})
    #st.write('Distribusi Usia Nasabah:')
    #st.write(job_count)
    
    # Hitung jumlah nasabah berdasarkan status pernikahan
    count_marital = df['marital'].value_counts()
    # Buat Pie Chart
    fig_marital, ax = plt.subplots()
    ax.pie(count_marital.values, labels=count_marital.index, autopct='%1.1f%%')
    ax.set_title("Pie Chart Status Pernikahan")
    # st.pyplot(fig_marital)    
    marital_count = df['marital'].value_counts()
    marital_count = pd.DataFrame({'Status':marital_count.index, 'Jumlah':marital_count.values})
            
    sns.set_style("whitegrid")
    fig_balance, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x='y', y='balance', data=df, ax=ax)
    ax.set_xlabel('Berlangganan')
    ax.set_ylabel('Balance')
    ax.set_title('Box Plot Distribusi Balance by Target')
    
    # Membuat scatter plot balance vs age
    fig_agebal, ax = plt.subplots()
    ax.scatter(df["age"], df["balance"])
    ax.set_xlabel("Age")
    ax.set_ylabel("Balance")
    ax.set_title("Scatter Plot Balance vs Age")

    # Bar chart for education
    count_education = df['education'].value_counts()
    fig_edu, ax = plt.subplots()
    ax.bar(count_education.index, count_education.values)
    ax.set_xlabel('Education')
    ax.set_ylabel('Nasabah')
    ax.set_title('Bar Chart Education Distribution')
    education_count = df['marital'].value_counts()
    education_count = pd.DataFrame({'Status':education_count.index, 'Jumlah':education_count.values})
    
    
    # Hitung jumlah nasabah berdasarkan status pinjaman
    count_loan = df['loan'].value_counts()

    # Visualisasi dengan Bar chart
    fig_loan, ax = plt.subplots()
    ax.bar(count_loan.index, count_loan.values)
    ax.set_xlabel('Loan')
    ax.set_ylabel('Jumlah Nasabah')
    ax.set_title('Bar Chart Loan Distribution')
    loan_count = df['marital'].value_counts()
    loan_count = pd.DataFrame({'Status':loan_count.index, 'Jumlah':loan_count.values})
        
    # membuat pilihan grafik yang tersedia
    option = st.sidebar.selectbox(
        'Pilih Grafik yang Akan Ditampilkan',
        ('Histogram Usia', 'Bar Chart Job Distribution', 'Pie Chart Status Pernikahan',
         'Distribusi Balance berdasarkan Nasabah', 'Scatter hubungan antara saldo dan usia nasabah.', 'Jumlah nasabah berdasarkan tingkat pendidikan.',
         'Jumlah nasabah berdasarkan status pinjaman'))

    # menampilkan grafik yang dipilih
    if option == 'Histogram Usia':
        # code untuk menampilkan histogram usia
        st.write("Berikut merupakan Histogram yang menampilkan distribusi usia yang ada pada data mulai dari belasan tahun hingga 90-an tahun. Distribusi terbanyak ada pada usia 30an hingga usia 40 dan usia 32 merupakan yang terbanyak.")
        st.pyplot(fig_age)
        st.write(usia_count)
    if option == 'Bar Chart Job Distribution':
        # code untuk menampilkan bar chart job distribution
        st.pyplot(fig_job)
        st.write(job_count)
    if option == 'Pie Chart Status Pernikahan' :
        st.pyplot(fig_marital)
        st.write(marital_count)
    if option == 'Distribusi Balance berdasarkan Nasabah' :
        st.pyplot(fig_balance)
    if option == 'Scatter hubungan antara saldo dan usia nasabah.' :
        st.pyplot(fig_agebal)
    if option == 'Jumlah nasabah berdasarkan tingkat pendidikan.' :
        st.pyplot(fig_edu)
        st.write(education_count)
    if option == 'Jumlah nasabah berdasarkan status pinjaman':
        st.pyplot(fig_loan)
        st.write(loan_count)

#Prediksi
if (selected=='Prediksi') :
    
    st.title('Prediksi klien berlangganan deposit jangka panjang')

    #create dictionaries for categorical features
    job_dict = {'admin.': 0, 'blue-collar': 1, 'entrepreneur': 2, 'housemaid': 3, 'management': 4,
    'retired': 5, 'self-employed': 6, 'services': 7, 'student': 8, 'technician': 9,
    'unemployed': 10, 'unknown': 11}

    marital_dict = {'divorced': 0, 'married': 1, 'single': 2}

    education_dict = {'primary': 0, 'secondary': 1, 'tertiary': 2, 'unknown': 3}

    loan_dict = {'no': 0, 'yes': 1}

    #input fields
    age = st.text_input('Usia')
    job = st.selectbox('Pekerjaan', tuple(job_dict.keys()))
    job = job_dict[job]
    marital = st.selectbox('Status Pernikahan', tuple(marital_dict.keys()))
    marital = marital_dict[marital]
    education = st.selectbox('Pendidikan', tuple(education_dict.keys()))
    education = education_dict[education]
    balance = st.text_input('Saldo Rata-rata Tahunan (dalam ribu Euro)')
    pdays = st.text_input('Jumlah hari sejak kontak terakhir')
    previous = st.text_input('Jumlah kontak sebelumnya')
    loan = st.selectbox('Punya pinjaman?', tuple(loan_dict.keys()))
    loan = loan_dict[loan]

    #code untuk prediksi
    pred_subs = ''

    try:
        if st.button('Test Prediksi'):
            pred_subs_client = data_bank_model.predict([[age, job, marital, education, balance, pdays, previous, loan]])

            if(pred_subs_client[0]==1):
                pred_subs = 'Client Will Subscribe'
            else :
                pred_subs = 'Client Will not Subscribe'
            st.success(pred_subs)
            
    except ValueError:
            st.error('Field Tidak Boleh Kosong!')
            
if (selected=='Model') :
    
    st.title('Model Prediksi')
    option = st.sidebar.selectbox(
        'Pilih yang Akan Ditampilkan',
        ('Data', 'Feature Importances', 'Corelation Matrix', 'Confusion Matrix'))
    
    if option == 'Data':
        # Menampilkan data dalam bentuk tabel
        st.write(df)
    if option == 'Feature Importances':
        # Create a dataframe of feature importances
        feature_importances = pd.DataFrame(clf.feature_importances_, index=x.columns, columns=['Importance'])
        feature_importances.sort_values('Importance', ascending=False, inplace=True)

        # Plot the feature importances
        fig, ax = plt.subplots()
        ax.bar(feature_importances.index, feature_importances['Importance'])
        plt.xticks(rotation=90)
        ax.set_title("Feature Importances")
        ax.set_xlabel("Features")
        ax.set_ylabel("Importance")
        # Display the plot in Streamlit
        st.pyplot(fig)
        # Display the feature importances in a table
        st.write("Feature Importances:")
        st.write(feature_importances)
    
    if option == 'Corelation Matrix':
    
        # Compute correlation matrix
        corr = data_bersih.corr()
        # Set figure size
        fig, ax = plt.subplots(figsize=(12, 10))
        # Create heatmap
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        # Set font size of tick labels
        ax.tick_params(labelsize=10)
        # Show plot
        st.pyplot(fig)
    
    if option == 'Confusion Matrix':
               
        cm = confusion_matrix(y_test, y_pred)
        st.write("Confusion Matrix:")
        fig, ax = plt.subplots()
        ax.imshow(cm, cmap='Blues')
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_xticks(np.arange(len(np.unique(y_test))))
        ax.set_yticks(np.arange(len(np.unique(y_test))))
        ax.set_xticklabels(np.unique(y_test))
        ax.set_yticklabels(np.unique(y_test))
        ax.tick_params(axis='x')
        for i in range(len(np.unique(y_test))):
            for j in range(len(np.unique(y_test))):
                ax.text(j, i, cm[i, j], ha='center', va='center', color='black')
        st.pyplot(fig)
