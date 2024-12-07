# Article Popularity Prediction

## 00_Web_Scraping 🕸️
This section involves scraping data from the Scopus API. The provided credentials in `data.py` will be used to authenticate and retrieve data from Scopus.

### Steps:
1. Navigate to the project directory:
   ```bash
   cd 00_Web_Scraping
   code .
   ```

2. Create a file named `data.py` with the following content:
   ```python
   _id = "your email"
   _pass = "your pass"
   ```

## 01_Data_Preparation 🧹
This section involves extracting data provided by the professor and combining it with the data scraped from the web.

### Steps:
1. Run the `1_ingest_Ajarn.ipynb` file to extract data provided by the professor

2. Run the `2_concat_data.ipynb` file to combine the data from the professor with the data scraped from the web.

3. The combined data will be saved as `2_data_combined`.

## 02_0_Data_Storage 💾
This section covers how to store the combined dataset into a Cassandra database and interact with it using CQL (cassandra).

### Steps:
1. Install Cassandra locally or use a cloud-based instance
   Start cassandra using `sudo service cassandra start`
2. Set up the database by running `cqlsh -f 01.5_Data_Storage/structure.cql` 
3. Write the data to Cassandra using PySpark `python -u 01.5_Data_Storage/spark_storage.py` 
After completing these steps, you can query and use the stored data with PySpark.

## 02_Data_Science 🔬
This section involves using the `2_data_combined` dataset to train a model to predict the cited by count of research papers.

### Steps:
1. Run the `train_model.ipynb` file to train the model on the combined dataset and generate predictions for the cited by count of research papers.

## 03_Data_Visualization 📊
This section involves visualizing the data using Streamlit.

### Steps:
1. Navigate to the project directory and run the app:
   ```bash
   cd 03_Data_Visualization
   streamlit run app.py
   ```


<!-- ## Instructions

You can train model by yourself using Data3_en.ipynb or smthing. and adjust parameter by yourself.
but if you lazy you can just copy model that i push it on repository, 90 percent of data is from scopus.

## Try it your self

https://chulinuwu-articlepopularityprediction-app-dev-cpmott.streamlit.app/ -->

