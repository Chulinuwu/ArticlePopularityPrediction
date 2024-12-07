import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import folium
from streamlit_folium import st_folium
import json
import numpy as np
from collections import Counter
import seaborn as sns
import plotly.express as px
from autogluon.tabular import TabularPredictor
import os
import networkx as nx
from pyvis.network import Network

# Load data
data_path = '03_Data_Visualization/data_sample.csv'  # Replace with actual path in deployment
if os.path.exists(data_path):
    data = pd.read_csv(data_path)
else:
    st.error(f"File not found: {data_path}")

# Sidebar with enhanced design
st.sidebar.markdown("""
<style>
    .sidebar-title {
        font-size: 24px;
        font-weight: bold;
        color: #4CAF50;
        margin-bottom: 15px;
    }
    .sidebar-selectbox {
        font-size: 18px;
        color: #333;
        margin-top: 10px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-title">🔍 Navigation</div>', unsafe_allow_html=True)
page = st.sidebar.selectbox(
    "Choose a page:",
    ["Overview", "Relational Graphics", "Spatial Data Visualization", "Network Visualization", "Model Inference", "Model Comparison"],
    format_func=lambda x: f"📄 {x}" if x == "Overview" else f"📊 {x}" if x == "Relational Graphics" else f"🌍 {x}" if x == "Spatial Data Visualization" else f"🔗 {x}" if x == "Network Visualization" else f"🤖 {x}" if x == "Model Inference" else f"⚖️ {x}"
)

# Add some additional explanation or information
st.sidebar.markdown("""
---
### About This App
Use the navigation menu above to explore:
- **Overview**: A summary of the dataset and key statistics.
- **Relational Graphics**: Interactive visualizations of data relationships.
- **Spatial Data Visualization**: Geographic analysis of data.
""")


if page == "Overview":
# Title of the app
    st.title("📊 Article Popularity Prediction - Data Exploration")

    # Overview of data
    st.header("📋 Dataset Overview")
    st.markdown("### A quick glimpse of the first five rows of the dataset:")
    st.write(data.head())

    # Summary statistics
    st.header("📈 Summary Statistics")
    st.markdown("### Key statistics for selected columns:")
    columns_to_describe = ['CitedByCount', 'Asia', 'Europe', 'Oceania', 'North America', 'Africa', 'South America']
    st.write(data[columns_to_describe].describe())

elif page == "Relational Graphics":
    # Distribution of CitedByCount
    st.header("🧮 Distribution of Article Citations")
    st.markdown("### Histogram of article citation counts (truncated for clarity):")

    fig = px.histogram(data, x='CitedByCount', nbins=100, title="Distribution of Article Citations")
    fig.update_layout(
        xaxis_title="Citations",
        yaxis_title="Frequency",
        xaxis=dict(range=[0, 200]),
        yaxis=dict(range=[0, 500])
    )

    st.plotly_chart(fig)

    # Word Cloud for keywords
    st.header("🌟 Keyword Analysis")
    if 'keywords_list' in data.columns:
        all_keywords = data['keywords_list'].apply(eval).explode()
        word_counts = Counter(all_keywords)
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_counts)

        st.markdown("### Word Cloud of Article Keywords:")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

        st.markdown("### Top 20 Keywords:")
        top_words = word_counts.most_common(20)
        for word, count in top_words:
            st.markdown(f"- **{word}**: {count}")
    else:
        st.error("The uploaded file does not contain a 'keywords_list' column.")


    # Articles per journal
    st.header("📚 Top Journals by Article Count")
    st.markdown("### Number of articles published in the top 10 journals:")

    journal_counts = data['PublicationName'].value_counts().head(10).reset_index()
    journal_counts.columns = ['PublicationName', 'ArticleCount']

    fig = px.bar(journal_counts, x='ArticleCount', y='PublicationName', orientation='h', 
                title="Top 10 Journals by Article Count", 
                labels={'ArticleCount': 'Number of Articles', 'PublicationName': 'Journals'},
                color='ArticleCount', color_continuous_scale='Blues')

    fig.update_layout(yaxis={'categoryorder':'total ascending'})

    st.plotly_chart(fig)

    # Articles by region
    st.header("🌍 Articles by Region")
    st.markdown("### Distribution of articles across different regions:")

    regions = ['Asia', 'Europe', 'Oceania', 'North America', 'Africa', 'South America']
    region_counts = data[regions].sum().reset_index()
    region_counts.columns = ['Region', 'ArticleCount']

    fig = px.bar(region_counts, x='Region', y='ArticleCount', 
                title="Articles Published by Region", 
                labels={'ArticleCount': 'Number of Articles', 'Region': 'Regions'},
                color='ArticleCount', color_continuous_scale='Viridis')

    st.plotly_chart(fig)

    # CitedByCount vs. Journal
    st.header("📊 Average Citations by Journal")
    st.markdown("### Average number of citations for the top 10 journals:")

    avg_citations = data.groupby('PublicationName')['CitedByCount'].mean().sort_values(ascending=False).head(10).reset_index()
    avg_citations.columns = ['PublicationName', 'AverageCitations']

    fig = px.bar(avg_citations, x='AverageCitations', y='PublicationName', orientation='h', 
                title="Top 10 Journals by Average Citations", 
                labels={'AverageCitations': 'Average Citations', 'PublicationName': 'Journals'},
                color='AverageCitations', color_continuous_scale='viridis')

    fig.update_layout(yaxis={'categoryorder':'total ascending'})

    st.plotly_chart(fig)

    # Keyword Category Distribution
    st.header("📌 Keyword Category Distribution")
    st.markdown("### Distribution of articles across different keyword categories:")

    keyword_category_counts = data['keyword_category'].value_counts().reset_index()
    keyword_category_counts.columns = ['KeywordCategory', 'ArticleCount']

    fig = px.bar(keyword_category_counts, x='ArticleCount', y='KeywordCategory', orientation='h', 
                title="Keyword Category Distribution", 
                labels={'ArticleCount': 'Number of Articles', 'KeywordCategory': 'Keyword Categories'},
                color='ArticleCount', color_continuous_scale='magma')

    fig.update_layout(yaxis={'categoryorder':'total ascending'})

    st.plotly_chart(fig)


elif page == "Spatial Data Visualization":
    # สร้างพิกัดละติจูดและลองจิจูดสำหรับแต่ละทวีป
    continent_coords = {
        'Asia': [34.0479, 100.6197],
        'Europe': [54.5260, 15.2551],
        'Oceania': [-25.2744, 133.7751],
        'North America': [54.5260, -105.2551],
        'Africa': [-8.7832, 34.5085],
        'South America': [-14.2350, -51.9253],
    }

    # รวมข้อมูลจากทุกทวีป
    continent_columns = ['Asia', 'Europe', 'Oceania', 'North America', 'Africa', 'South America']
    continent_data = pd.DataFrame([
        {'Continent': continent, 'Value': int(data[continent].sum()), 'Lat': coord[0], 'Lon': coord[1]}
        for continent, coord in continent_coords.items()
    ])

    # กรองข้อมูลที่มีค่า Value มากกว่า 0
    continent_data = continent_data[continent_data['Value'] > 0]

    # Normalization ของค่า Value สำหรับการกำหนดสี
    max_value = continent_data['Value'].max()
    min_value = continent_data['Value'].min()

    def get_color(value):
        """ให้สีเข้มขึ้นตามค่าที่มากขึ้น โดยใช้สีแดงเป็นสีเข้มสุดและสีเหลืองเป็นสีอ่อนสุด"""
        # ใช้การปรับสเกลแบบลอการิทึมเพื่อเพิ่มความละเอียดของสีในช่วงค่าที่ต่ำกว่า
        log_value = np.log(value - min_value + 1)
        log_max = np.log(max_value - min_value + 1)
        normalized = log_value / log_max
        red = 255
        green = int(255 * (1 - normalized))
        blue = 0
        return f"rgba({red}, {green}, {blue}, 0.7)"

    # สร้าง Streamlit UI
    # สร้าง UI ที่สวยงามขึ้นใน Streamlit
    st.title("🗺️ Spatial Data Visualization")
    st.markdown("""
    ### Heatmap of citation by Continent
    This visualization shows the distribution of articles across different continents. 
    The intensity of the color represents the total number of citation.
    """)

    # สร้างแผนที่ด้วย Folium
    m = folium.Map(location=[20, 0], zoom_start=2)

    # เพิ่มจุดทวีปพร้อมสีเข้ม/อ่อนตามค่า Value
    # for _, row in continent_data.iterrows():
    #     folium.CircleMarker(
    #         location=[row['Lat'], row['Lon']],
    #         radius=15,  # ขนาดของ Circle
    #         popup=f"{row['Continent']}: {row['Value']}",
    #         color=get_color(row['Value']),
    #         fill=True,
    #         fill_opacity=0.8
    #     ).add_to(m)

    # โหลด GeoJSON สำหรับขอบเขตของแต่ละทวีป
    geojson_path = '03_Data_Visualization/continents.json'  # Replace with actual path to GeoJSON file
    with open(geojson_path) as f:
        geojson_data = json.load(f)

    # เพิ่มค่าใน properties ของ GeoJSON
    for feature in geojson_data['features']:
        continent_name = feature['properties']['CONTINENT']
        if continent_name in continent_data['Continent'].values:
            feature['properties']['Value'] = int(continent_data.set_index('Continent').loc[continent_name, 'Value'])

    # เพิ่ม Choropleth layer
    choropleth = folium.Choropleth(
        geo_data=geojson_data,
        name='choropleth',
        data=continent_data,
        columns=['Continent', 'Value'],
        key_on='feature.properties.CONTINENT',  # Adjust this key based on your GeoJSON structure
        fill_color='RdYlGn',  # ใช้พาเลตสีที่มีความแตกต่างกันมากขึ้น
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='Sum of Values by Continent'
    ).add_to(m)

    # เพิ่ม Tooltip เพื่อแสดงค่าเมื่อ hover
    folium.GeoJson(
        geojson_data,
        style_function=lambda feature: {
            'fillColor': get_color(continent_data.set_index('Continent').loc[feature['properties']['CONTINENT'], 'Value']) if feature['properties']['CONTINENT'] in continent_data['Continent'].values else 'transparent',
            'color': 'black' if feature['properties']['CONTINENT'] in continent_data['Continent'].values else 'transparent',
            'weight': 1,
            'dashArray': '5, 5'
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['CONTINENT', 'Value'],
            aliases=['Continent:', 'Value:'],
            localize=True
        )
    ).add_to(m)

    # แสดงแผนที่ใน Streamlit
    st_folium(m, width=700, height=500)
    
elif page == "Model Inference":
    # Load the model
    model_path = 'AutogluonModels/ag-20241206_093135'
    predictor = TabularPredictor.load(model_path, require_py_version_match=False)
    best_model_name = 'XGBoost'

    # Streamlit app
    st.title('CitedByCount Prediction')

    # Input form
    st.header('Input Data ( Model : Extreme Gradient Boosting)')
    year = st.number_input('Year (ปีที่ตีพิมพ์)', min_value=1900, max_value=2100, step=1)
    title = st.text_input('Title (หัวข้อบทความ)')
    publication_name = st.text_input('Publication Name (ชื่อวารสาร)')
    author_keywords = st.text_input('Author Keywords (คีย์เวิร์ดของผู้เขียน)')
    authors_from_asia = st.number_input('authors_from_Asia (จำนวนผู้เขียนจากทวีป Asia)', min_value=0, max_value=10, step=1)
    authors_from_oceania = st.number_input('authors_from_Oceania (จำนวนผู้เขียนจากทวีป Oceania)', min_value=0, max_value=10, step=1)
    authors_from_europe = st.number_input('authors_from_Europe (จำนวนผู้เขียนจากทวีป Europe)', min_value=0, max_value=10, step=1)
    authors_from_north_america = st.number_input('authors_from_North America (จำนวนผู้เขียนจากทวีป North America)', min_value=0, max_value=10, step=1)
    authors_from_africa = st.number_input('authors_from_Africa (จำนวนผู้เขียนจากทวีป Africa)', min_value=0, max_value=10, step=1)
    authors_from_south_america = st.number_input('authors_from_South America (จำนวนผู้เขียนจากทวีป America)', min_value=0, max_value=10, step=1)
    authors_from_unknown = st.number_input('authors_from_Unknown (จำนวนผู้เขียนจากทวีปอะไรก็ไม่รู้)', min_value=0, max_value=10, step=1)

    # Create a DataFrame from the input
    input_data = pd.DataFrame({
        'Year': [year],
        'Title': [title],
        'PublicationName': [publication_name],
        'AuthorKeywords': [author_keywords],
        'Asia': [authors_from_asia],
        'Oceania': [authors_from_oceania],
        'Europe': [authors_from_europe],
        'North America': [authors_from_north_america],
        'Africa': [authors_from_africa],
        'South America': [authors_from_south_america],
        'Unknown': [authors_from_unknown]
    })

    # Predict button
    if st.button('Predict'):
        # Predict the CitedByCount
        prediction = predictor.predict(input_data, model=best_model_name)
        st.write(f'Predicted CitedByCount: {prediction[0]}')

        # Display the input data and prediction
        st.subheader('Input Data')
        st.write(input_data)

        st.subheader('Prediction')
        st.write(prediction)
        
elif page == "Model Comparison":
    st.write("Model Comparison Page")
    def load_prediction_files(results_folder):
        """
        Load prediction CSV files from results folder
        
        Parameters:
        - results_folder: Path to folder containing prediction files
        
        Returns:
        - Dictionary of DataFrames with predictions
        """
        predictions = {}
        for filename in os.listdir(results_folder):
            if filename.endswith('.csv'):
                model_name = filename.replace('predicted_vs_actual_', '').replace('.csv', '')
                file_path = os.path.join(results_folder, filename)
                predictions[model_name] = pd.read_csv(file_path)
        return predictions

    def create_comparison_plot(df, model_name):
        """
        Create comparison plot using Seaborn
        
        Parameters:
        - df: DataFrame with actual and predicted values
        - model_name: Name of the model
        
        Returns:
        - Matplotlib figure
        """
        plt.figure(figsize=(12, 6))
        
        # Plot actual values
        sns.scatterplot(
            x=df.index, 
            y='Actual', 
            data=df, 
            color='blue', 
            label='Actual', 
            alpha=0.5
        )
        
        # Plot predicted values
        sns.scatterplot(
            x=df.index, 
            y='Predicted', 
            data=df, 
            color='red', 
            label='Predicted', 
            alpha=0.5
        )
        
        plt.xlabel('Index')
        plt.ylabel('CitedByCount')
        plt.title(f'Comparison of Actual vs Predicted CitedByCount - {model_name}')
        plt.legend()
        
        return plt
    
    # Title
    st.title('Model Predictions Comparison')
    
    # Results folder path (modify as needed)
    results_folder = 'result'
    
    # Load prediction files
    predictions = load_prediction_files(results_folder)
    
    # Sidebar for model selection
    selected_model = st.sidebar.selectbox(
        'Select Model', 
        list(predictions.keys())
    )
    
    # Main content area
    if selected_model:
        # Get selected model's predictions
        model_df = predictions[selected_model]
        
        # Create visualization
        fig = create_comparison_plot(model_df, selected_model)
        
        # Display plot
        st.pyplot(fig)
        
        # Display dataframe
        st.dataframe(model_df)
elif page == "Network Visualization":
    # Sample the data to reduce size
    sampled_data = data.sample(n=100, random_state=42)  # Adjust n to the desired sample size

    # Prepare the network data
    def create_network(data):
        G = nx.Graph()
        for idx, row in data.iterrows():
            research_title = row['Title']
            keywords = eval(row['keywords_list'])  # Convert string to list
            G.add_node(research_title, size=row['CitedByCount'], group=row['cluster'])
            for keyword in keywords:
                G.add_node(keyword, group='keyword')
                G.add_edge(research_title, keyword)
        return G

    # Visualize the network
    def visualize_network(graph, physics):
        net = Network(height="750px", width="100%", notebook=False)
        net.from_nx(graph)
        if not physics:
            net.toggle_physics(False)
        return net

    # Streamlit app setup
    def app():
        # st.set_page_config(page_title="Research Network Visualization", layout="wide")
        st.title("📚 Research Network Visualization")
        st.markdown(
            """
            This application provides an **interactive network visualization** of research papers 
            and their associated keywords. Use the sidebar to adjust the visualization settings.
            You can drag and drop the nodes as you prefer.
            """
        )
        
        # Sidebar options
        st.sidebar.title("🔧 Settings")
        st.sidebar.markdown("Adjust the settings below to customize the visualization:")
        n_samples = st.sidebar.slider("Number of Samples", min_value=10, max_value=50, value=20, step=5)
        physics_enabled = st.sidebar.checkbox("Enable Physics Simulation", value=True)

        # Sample the data
        sampled_data = data.sample(n=n_samples, random_state=42)

        # Show data insights
        st.sidebar.markdown("### 📊 Data Insights")
        # st.sidebar.write(f"**Total Papers:** {len(data)}")
        st.sidebar.write(f"**Selected Papers:** {n_samples}")
        st.sidebar.write(f"**Unique Keywords:** {len(set([kw for sublist in sampled_data['keywords_list'].apply(eval) for kw in sublist]))}")

        # Create the network graph
        graph = create_network(sampled_data)

        # Visualize the network
        net = visualize_network(graph, physics_enabled)
        path = "research_network.html"
        net.save_graph(path)
        
        # Display the graph in Streamlit
        with open(path, 'r') as f:
            html = f.read()
        st.components.v1.html(html, height=800, width=800)

        # Show data preview
        with st.expander("🔍 Data Preview"):
            st.write(sampled_data)

    if __name__ == "__main__":
        app()