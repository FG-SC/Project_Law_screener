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

# --- Data Collection Functions ---

# List of Brazilian states' abbreviations
estados_brasileiros = [
    'ac', 'al', 'ap', 'am', 'ba', 'ce', 'df', 'es', 'go', 'ma',
    'mt', 'ms', 'mg', 'pa', 'pb', 'pr', 'pe', 'pi', 'rj', 'rn',
    'rs', 'ro', 'rr', 'sc', 'sp', 'se', 'to'
]

def webcraping_leis_municipais(query, estado='sc', paginas=1):
    text_list, links_list, lista_cidades, tipo_da_lei, ano_da_lei = [], [], [], [], []
    
    for i in range(1, paginas + 1):
        url = f'https://leisestaduais.com.br/{estado}?q={query}&page={i}&types=&state={estado}&status=&date_start=&date_end=&lm=1'
        result = requests.get(url)
        soup = BeautifulSoup(result.text, 'html.parser')
        leis = soup.find_all(class_="listagem-leis")
        
        for lei in leis[0].find_all(class_="btn btn-lei-lista btn-lei-lista-leismunicipais"):
            text_list.append(lei.find('span', {'rel': 'text'}).get_text(strip=False))
            links_list.append(lei['href'])
            lista_cidades.append(lei['href'].split('/')[6])
            tipo_da_lei.append(lei['href'].split('/')[7])
            ano_da_lei.append(lei['href'].split('/')[8])
    
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
    municipalities = geobr.read_municipality(code_muni=state_code, year=2019, simplified=False)
    municipalities['name_muni'] = municipalities['name_muni'].apply(normalize_municipality_name)
    treemap_df['município'] = treemap_df['município'].apply(normalize_municipality_name)
    
    municipalities.index = municipalities['name_muni']
    final_data = municipalities.join(treemap_df.set_index('município'))[['Values', 'geometry']]
    
    plot_function(final_data, 'Values')

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    return colors.LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
        cmap(np.linspace(minval, maxval, n))
    )

# --- Plotting Functions ---

def plot_function(data, values_column):
    cmap = plt.get_cmap('RdYlGn')
    new_cmap = truncate_colormap(cmap, 0.3, 1)
    fig, ax = plt.subplots(1, figsize=(16, 9))
    ax.axis('off')
    ax.set_title('Heatmap of Laws Containing Search Word by Municipality', fontdict={'fontsize': '15', 'fontweight': '3'})
    data.plot(column=values_column,
              cmap=new_cmap,
              linewidth=0.9,
              ax=ax,
              edgecolor='1',
              legend=True,
              missing_kwds={"color": "lightgrey", "label": "Missing values"})
    st.pyplot(fig)

def plot_wordcloud(text, stopwords):
    wordcloud = WordCloud(stopwords=stopwords, background_color="black", width=2850, height=1800).generate(text)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_axis_off()
    st.pyplot(fig)

def plot_time_series(final_time_series):
    final_time_series.index = pd.to_datetime(final_time_series.index, format='%Y')
    cumsum_df = final_time_series#.cumsum()
    
    time_series_fig = px.line(
        cumsum_df,
        x=cumsum_df.index,
        y=cumsum_df.columns,
        labels={'value': 'Values', 'index': 'Year'},
        title='Time Series Plot'
    )
    time_series_fig.update_layout(xaxis_title='Year', yaxis_title='Values')
    time_series_fig.update_xaxes(tickangle=45)
    
    st.plotly_chart(time_series_fig)

def plot_treemap(cumsum_df):
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
    
    st.plotly_chart(treemap_fig)

def plot_pizza_leis(df, ano=None):
    if ano:
        df = df.loc[df['ano'] == ano]
    
    fig = px.sunburst(data_frame=df, path=['tipo', 'município'], values='Counts', height=1000)
    st.plotly_chart(fig)

# --- Streamlit App ---

st.title('Brazilian Municipal Laws Dashboard')
st.write("Select a state and enter a search query to generate dashboards based on municipal laws.")

# State selection
estado = st.selectbox('Select a state:', estados_brasileiros, index=23)  # Default to 'sc'
query = st.text_input('Enter a search query:', value='startup')
paginas = st.slider('Select the number of pages to scrape:', 1, 50, 20)

if st.button('Generate Dashboards'):
    # Scraping data
    test = webcraping_leis_municipais(query=query, estado=estado, paginas=paginas)

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

    st.write(f'municípios com mais leis sobre {query}:\n\n', treemap_df.sort_values(by='Values', ascending=False).head().reset_index(drop=True))

    plot_wordcloud(all_summary, stop_words)
    plot_pizza_leis(df=df_final)
    plot_time_series(final_time_series)
    plot_treemap(final_time_series)
    map_values_to_regions(treemap_df, state_code)
