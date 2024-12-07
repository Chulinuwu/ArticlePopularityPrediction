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

# Load data
data_path = 'data_sample.csv'  # Replace with actual path in deployment
data = pd.read_csv(data_path)

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
    ["Overview", "Relational Graphics", "Spatial Data Visualization"],
    format_func=lambda x: f"📄 {x}" if x == "Overview" else f"📊 {x}" if x == "Relational Graphics" else f"🌍 {x}"
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
    geojson_path = 'continents.json'  # Replace with actual path to GeoJSON file
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