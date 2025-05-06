import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
from bs4 import BeautifulSoup
import unicodedata
from wordcloud import WordCloud, STOPWORDS
import plotly.express as px
import geobr
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from datetime import datetime

# --- Data Collection Functions ---

# List of Brazilian states' abbreviations
estados_brasileiros = [
    'ac', 'al', 'ap', 'am', 'ba', 'ce', 'df', 'es', 'go', 'ma',
    'mt', 'ms', 'mg', 'pa', 'pb', 'pr', 'pe', 'pi', 'rj', 'rn',
    'rs', 'ro', 'rr', 'sc', 'sp', 'se', 'to'
]

def webcraping_leis_municipais(query, estado='sc', paginas=1):
    text_list, links_list, lista_cidades, tipo_da_lei, ano_da_lei = [], [], [], [], []
    scraper = cloudscraper.create_scraper()  # Create scraper instance
    
    for i in range(1, paginas + 1):
        url = f'https://leisestaduais.com.br/{estado}?q={query}&page={i}&types=&state={estado}&status=&date_start=&date_end=&lm=1'
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            result = scraper.get(url, headers=headers, timeout=30)
            result.raise_for_status()
            
            # Check if we got a Cloudflare challenge page
            if "Checking your browser before accessing" in result.text:
                st.warning("Cloudflare challenge detected. Retrying with different settings...")
                # Try with different settings
                scraper = cloudscraper.create_scraper(delay=10)
                result = scraper.get(url, timeout=30)
                
            soup = BeautifulSoup(result.text, 'html.parser')
            leis = soup.find_all(class_="listagem-leis")
            
            if not leis:
                st.warning(f"No laws found on page {i}")
                continue
                
            for lei in leis[0].find_all(class_="btn btn-lei-lista btn-lei-lista-leismunicipais"):
                text_list.append(lei.find('span', {'rel': 'text'}).get_text(strip=False))
                links_list.append(lei['href'])
                lista_cidades.append(lei['href'].split('/')[6])
                tipo_da_lei.append(lei['href'].split('/')[7])
                ano_da_lei.append(lei['href'].split('/')[8])
                
        except Exception as e:
            st.error(f"Error accessing page {i}: {str(e)}")
            continue
    
    # Rest of your function remains the same...
    
    if not text_list:
        st.error("No laws found with the current search criteria.")
        return pd.DataFrame()
    
    text_list = [re.split(r'\s+', string)[1] for string in text_list]
    leis_municipais = pd.DataFrame({
        'município': lista_cidades,
        'ano': ano_da_lei,
        'tipo': tipo_da_lei,
        'Link': links_list,
        'conteúdo': text_list
    })

    return leis_municipais.sort_values(by='ano', ascending=False).drop_duplicates()

# --- Data Preparation Functions ---

def normalize_municipality_name(name):
    name = name.lower().replace(' ', '-')
    return unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('utf-8')

def map_values_to_regions(treemap_df, state_code):
    try:
        municipalities = geobr.read_municipality(code_muni=state_code, year=2019, simplified=False)
        municipalities['name_muni'] = municipalities['name_muni'].apply(normalize_municipality_name)
        treemap_df['município'] = treemap_df['município'].apply(normalize_municipality_name)
        
        municipalities.index = municipalities['name_muni']
        final_data = municipalities.join(treemap_df.set_index('município'))[['Values', 'geometry']]
        
        plot_function(final_data, 'Values')
    except Exception as e:
        st.error(f"Error generating map: {str(e)}")

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    return colors.LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
        cmap(np.linspace(minval, maxval, n))
    )

# --- Plotting Functions ---

def plot_function(data, values_column):
    try:
        cmap = plt.get_cmap('RdYlGn')
        new_cmap = truncate_colormap(cmap, 0.3, 1)
        fig, ax = plt.subplots(1, figsize=(16, 9))
        ax.axis('off')
        ax.set_title('Heatmap of Laws Containing Search Word by Municipality', 
                     fontdict={'fontsize': '15', 'fontweight': '3'})
        data.plot(column=values_column,
                  cmap=new_cmap,
                  linewidth=0.9,
                  ax=ax,
                  edgecolor='1',
                  legend=True,
                  missing_kwds={"color": "lightgrey", "label": "Missing values"})
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.error(f"Error plotting heatmap: {str(e)}")

def plot_wordcloud(text, stopwords):
    try:
        wordcloud = WordCloud(
            stopwords=stopwords, 
            background_color="black", 
            width=2850, 
            height=1800,
            max_words=200
        ).generate(text)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_axis_off()
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.error(f"Error generating word cloud: {str(e)}")

def plot_time_series(final_time_series):
    try:
        final_time_series.index = pd.to_datetime(final_time_series.index, format='%Y')
        cumsum_df = final_time_series
        
        time_series_fig = px.line(
            cumsum_df,
            x=cumsum_df.index,
            y=cumsum_df.columns,
            labels={'value': 'Number of Laws', 'index': 'Year'},
            title='Evolution of Laws Over Time'
        )
        time_series_fig.update_layout(
            xaxis_title='Year',
            yaxis_title='Number of Laws',
            hovermode='x unified'
        )
        time_series_fig.update_xaxes(tickangle=45)
        
        st.plotly_chart(time_series_fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error plotting time series: {str(e)}")

def plot_treemap(cumsum_df):
    try:
        treemap_df = cumsum_df.iloc[-1:].stack().reset_index()
        treemap_df.rename(columns={0: 'Values'}, inplace=True)
        treemap_df.columns = ['ano', 'município', 'Values']
        
        treemap_fig = px.treemap(
            treemap_df,
            path=['ano', 'município'],
            values='Values',
            color='Values',
            color_continuous_scale='RdYlGn',
            title='Treemap of Laws Containing Search Word by Municipality and Year'
        )
        treemap_fig.update_traces(textinfo="label+value")
        
        st.plotly_chart(treemap_fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error plotting treemap: {str(e)}")

def plot_pizza_leis(df, ano=None):
    try:
        if ano:
            df = df.loc[df['ano'] == ano]
        
        if df.empty:
            st.warning("No data available for the selected year.")
            return
            
        fig = px.sunburst(
            data_frame=df, 
            path=['tipo', 'município'], 
            values='Counts', 
            height=700,
            title='Distribution of Law Types by Municipality'
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error plotting sunburst chart: {str(e)}")

# --- Streamlit App ---

def main():
    st.set_page_config(
        page_title="Brazilian Municipal Laws Dashboard",
        page_icon="📜",
        layout="wide"
    )
    
    st.title('📜 Brazilian Municipal Laws Dashboard')
    st.write("""
        This dashboard allows you to explore municipal laws across Brazilian states. 
        Select a state and enter a search term to analyze laws containing that term.
    """)
    
    with st.sidebar:
        st.header("Search Parameters")
        estado = st.selectbox('Select a state:', estados_brasileiros, index=23)  # Default to 'sc'
        query = st.text_input('Enter a search query:', value='startup')
        paginas = st.slider('Select the number of pages to scrape:', 1, 50, 5)
        
        st.markdown("---")
        st.markdown("### Filters")
        selected_year = st.selectbox(
            "Select year to filter (optional):",
            options=["All years"] + list(range(datetime.now().year, 1990, -1)),
            index=0
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
            This app scrapes data from [leisestaduais.com.br](https://leisestaduais.com.br) 
            and visualizes municipal laws across Brazilian states.
        """)
    
    if st.button('Generate Dashboards', type="primary"):
        with st.spinner('Scraping data... This may take a few minutes...'):
            test = webcraping_leis_municipais(query=query, estado=estado, paginas=paginas)
        
        if test.empty:
            st.warning("No data found with the current search criteria. Try different parameters.")
            return
            
        # Data preparation
        df_final = test.copy()[['ano', 'município', 'tipo']]
        df_final['Counts'] = df_final.groupby(['ano', 'município'])['tipo'].transform('count')
        df_final.drop_duplicates(inplace=True)

        # Generating cumulative time series
        series = df_final.copy()
        series_t = series.groupby(by=['ano', 'município']).sum()
        final_time_series = series_t.unstack().T.loc['Counts'].fillna(0).T.cumsum()

        # Preparing text for word cloud
        summary = test['conteúdo'].values
        all_summary = " ".join(s for s in summary).replace('"', '').lower().replace("-", "").replace("nº", "").replace('.', '')
        stop_words = STOPWORDS.update(["da", "meu", "em", "você", "de", "ao", "os", "e", "o", "a", 'para', 'à', 'dispõe', 'dá', 'outras'])

        # Mapping values to regions and plotting the heatmap
        state_code = estado.upper()
        treemap_df = df_final.groupby('município').sum()['Counts'].to_frame().reset_index().rename(columns={'Counts': 'Values'})

        # Display results
        st.success(f"Found {len(test)} laws containing '{query}' in {estado.upper()}!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Laws Found", len(test))
        with col2:
            st.metric("Municipalities Covered", len(treemap_df))
        
        st.subheader(f"Top Municipalities with Laws about '{query}'")
        st.dataframe(treemap_df.sort_values(by='Values', ascending=False).head(10).reset_index(drop=True))
        
        st.subheader("Word Cloud of Law Contents")
        plot_wordcloud(all_summary, stop_words)
        
        st.subheader("Law Type Distribution")
        plot_pizza_leis(df=df_final, ano=selected_year if selected_year != "All years" else None)
        
        st.subheader("Evolution of Laws Over Time")
        plot_time_series(final_time_series)
        
        st.subheader("Geographical Distribution")
        map_values_to_regions(treemap_df, state_code)
        
        st.subheader("Treemap of Laws by Municipality")
        plot_treemap(final_time_series)
        
        st.subheader("Raw Data Preview")
        st.dataframe(test.head(50))

if __name__ == "__main__":
    main()
