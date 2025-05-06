import streamlit as st
import pandas as pd
import numpy as np
import re
from wordcloud import WordCloud, STOPWORDS
import plotly.express as px
import geobr
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from datetime import datetime
import time
import random
import requests

# --- App Configuration ---
st.set_page_config(
    page_title="Brazilian Municipal Laws Dashboard",
    page_icon="📜",
    layout="wide"
)

# --- Data Collection Functions ---

# List of Brazilian states' abbreviations
estados_brasileiros = [
    'ac', 'al', 'ap', 'am', 'ba', 'ce', 'df', 'es', 'go', 'ma',
    'mt', 'ms', 'mg', 'pa', 'pb', 'pr', 'pe', 'pi', 'rj', 'rn',
    'rs', 'ro', 'rr', 'sc', 'sp', 'se', 'to'
]

# --- Complete list of municipalities by state ---
# This function will fetch municipalities from an API for the selected state
@st.cache_data(ttl=3600*24)  # Cache for 24 hours
def get_all_municipalities(state_code):
    """Get all municipalities for a given state using IBGE API"""
    try:
        # First attempt - try to get municipalities from geobr
        municipalities = geobr.read_municipality(code_muni=state_code.upper(), year=2019)
        muni_list = municipalities['name_muni'].apply(lambda x: x.lower().replace(' ', '-')).tolist()
        st.success(f"Successfully loaded {len(muni_list)} municipalities from geobr")
        return muni_list
    except Exception as e:
        st.warning(f"Could not load municipalities from geobr: {e}")
        
        try:
            # Second attempt - try IBGE API
            url = f"https://servicodados.ibge.gov.br/api/v1/localidades/estados/{state_code}/municipios"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                muni_list = [muni['nome'].lower().replace(' ', '-') for muni in data]
                st.success(f"Successfully loaded {len(muni_list)} municipalities from IBGE API")
                return muni_list
            else:
                st.error(f"Error accessing IBGE API: {response.status_code}")
        except Exception as e:
            st.error(f"Error fetching municipalities: {e}")
        
        # Fallback to a minimum predefined list
        fallback_municipalities = {
            'ac': ['rio-branco', 'cruzeiro-do-sul', 'sena-madureira', 'tarauaca', 'feijo'],
            'al': ['maceio', 'arapiraca', 'palmeira-dos-indios', 'rio-largo', 'penedo'],
            'ap': ['macapa', 'santana', 'laranjal-do-jari', 'oiapoque', 'porto-grande'],
            'am': ['manaus', 'parintins', 'itacoatiara', 'manacapuru', 'tefe'],
            'ba': ['salvador', 'feira-de-santana', 'vitoria-da-conquista', 'camaçari', 'juazeiro'],
            'ce': ['fortaleza', 'caucaia', 'juazeiro-do-norte', 'maracanau', 'sobral'],
            'df': ['brasilia', 'ceilandia', 'taguatinga', 'planaltina', 'samambaia'],
            'es': ['vitoria', 'serra', 'vila-velha', 'cariacica', 'linhares'],
            'go': ['goiania', 'aparecida-de-goiania', 'anapolis', 'rio-verde', 'luziania'],
            'ma': ['sao-luis', 'imperatriz', 'timon', 'caxias', 'codó'],
            'mt': ['cuiaba', 'varzea-grande', 'rondonopolis', 'sinop', 'tangara-da-serra'],
            'ms': ['campo-grande', 'dourados', 'tres-lagoas', 'corumba', 'ponta-pora'],
            'mg': ['belo-horizonte', 'uberlandia', 'contagem', 'juiz-de-fora', 'betim'],
            'pa': ['belem', 'ananindeua', 'santarem', 'maraba', 'castanhal'],
            'pb': ['joao-pessoa', 'campina-grande', 'santa-rita', 'patos', 'bayeux'],
            'pr': ['curitiba', 'londrina', 'maringa', 'ponta-grossa', 'cascavel'],
            'pe': ['recife', 'jaboatao-dos-guararapes', 'olinda', 'caruaru', 'petrolina'],
            'pi': ['teresina', 'parnaiba', 'picos', 'floriano', 'campo-maior'],
            'rj': ['rio-de-janeiro', 'sao-goncalo', 'duque-de-caxias', 'nova-iguaçu', 'niteroi'],
            'rn': ['natal', 'mossoro', 'parnamirim', 'sao-goncalo-do-amarante', 'macaiba'],
            'rs': ['porto-alegre', 'caxias-do-sul', 'pelotas', 'canoas', 'santa-maria'],
            'ro': ['porto-velho', 'ji-parana', 'ariquemes', 'vilhena', 'cacoal'],
            'rr': ['boa-vista', 'rorainopolis', 'caracarai', 'mucajai', 'canta'],
            'sc': ['florianopolis', 'joinville', 'blumenau', 'criciuma', 'lages'],
            'sp': ['sao-paulo', 'guarulhos', 'campinas', 'sao-bernardo-do-campo', 'santo-andre'],
            'se': ['aracaju', 'nossa-senhora-do-socorro', 'lagarto', 'itabaiana', 'estancia'],
            'to': ['palmas', 'araguaina', 'gurupi', 'porto-nacional', 'paraiso-do-tocantins']
        }
        
        fallback_list = fallback_municipalities.get(state_code.lower(), [f"unknown-city-{i}" for i in range(1, 11)])
        st.warning(f"Using fallback list with {len(fallback_list)} municipalities. Some municipalities may be missing.")
        return fallback_list

# --- Data Preparation Functions ---

def normalize_municipality_name(name):
    name = str(name).lower().replace(' ', '-')
    import unicodedata
    return unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('utf-8')

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    return colors.LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
        cmap(np.linspace(minval, maxval, n))
    )

# --- Plotting Functions ---

def map_values_to_regions(treemap_df, state_code):
    try:
        municipalities = geobr.read_municipality(code_muni=state_code, year=2019, simplified=False)
        municipalities['name_muni'] = municipalities['name_muni'].apply(normalize_municipality_name)
        treemap_df['município'] = treemap_df['município'].apply(normalize_municipality_name)
        
        # Try to merge data
        municipalities.index = municipalities['name_muni']
        final_data = municipalities.join(treemap_df.set_index('município'), how='left')
        
        # Check if we have any matching data
        if final_data['Values'].isnull().all():
            st.warning("Could not match any municipalities with the data. This may be due to differences in naming conventions.")
            return
            
        final_data = final_data[['Values', 'geometry']]
        final_data['Values'] = final_data['Values'].fillna(0)
        
        plot_function(final_data, 'Values')
    except Exception as e:
        st.error(f"Error generating map: {str(e)}")
        st.error("If this is related to geobr package, try reinstalling it with: pip install git+https://github.com/ipeaGIT/geobr.git")

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
        # Check if we have any data
        if final_time_series.empty:
            st.warning("No time series data available.")
            return
            
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
        # Check if we have any data
        if cumsum_df.empty:
            st.warning("No treemap data available.")
            return
            
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
        # Make a copy to avoid modifying the original
        plot_df = df.copy()
        
        if ano and ano != "All years":
            plot_df = plot_df.loc[plot_df['ano'] == str(ano)]
        
        if plot_df.empty:
            st.warning("No data available for the selected year.")
            return
            
        fig = px.sunburst(
            data_frame=plot_df, 
            path=['tipo', 'município'], 
            values='Counts', 
            height=700,
            title='Distribution of Law Types by Municipality'
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error plotting sunburst chart: {str(e)}")

# --- Function to generate comprehensive sample data ---
def get_complete_sample_data(estado, query, start_year, end_year):
    """Generate comprehensive sample data for all municipalities in a state within a time range"""
    # Get all municipalities from the selected state
    municipalities = get_all_municipalities(estado)
    
    # Ensure we have at least some municipalities
    if not municipalities:
        municipalities = [f"{estado}-city-{i}" for i in range(1, 21)]
        
    # Common law types across Brazil
    types = ["lei-ordinaria", "lei-complementar", "decreto", "resolucao", "portaria", "instrucao-normativa"]
    
    # Generate entries for each year in the range
    years = list(range(start_year, end_year + 1))
    
    # Scale the number of entries based on municipality count and year range
    base_entries_per_muni = 3  # Average laws per municipality per year
    # Calculate total entries with some randomness (70-130% of base estimate)
    scale_factor = random.uniform(0.7, 1.3)
    target_entries = int(len(municipalities) * len(years) * base_entries_per_muni * scale_factor)
    
    # Ensure we have at least some minimum number of entries
    min_entries = max(50, len(municipalities) * 2)
    target_entries = max(min_entries, target_entries)
    
    # Cap at a reasonable maximum to prevent performance issues
    max_entries = 2000
    n_entries = min(target_entries, max_entries)
    
    # Generate municipality distribution with some municipalities having more laws
    # Use a power law distribution to make some municipalities have many more laws than others
    weights = np.random.power(0.8, size=len(municipalities))
    weights = weights / np.sum(weights)
    
    # Year distribution - newer years tend to have more laws
    year_weights = np.linspace(0.5, 1.0, len(years))
    year_weights = year_weights / np.sum(year_weights)
    
    # Content templates with variables for more realistic content
    content_templates = [
        f"Lei sobre {query} no município de {{muni}}",
        f"Regulamentação de {query} para {{muni}}",
        f"Dispõe sobre {query} e dá outras providências em {{muni}}",
        f"Estabelece normas para {query} no âmbito municipal de {{muni}}",
        f"Altera a legislação sobre {query} em {{muni}}",
        f"Cria o programa municipal de {query} em {{muni}}",
        f"Institui política pública para {query} no município de {{muni}}",
        f"Estabelece diretrizes orçamentárias para {query} em {{muni}}",
        f"Autoriza o poder executivo a implementar ações de {query} em {{muni}}",
        f"Determina a obrigatoriedade de {query} nos órgãos públicos de {{muni}}"
    ]
    
    # Generate more realistic sample data
    sample_muni = random.choices(municipalities, weights=weights, k=n_entries)
    sample_years = random.choices([str(y) for y in years], weights=year_weights, k=n_entries)
    
    sample_data = {
        'município': sample_muni,
        'ano': sample_years,
        'tipo': random.choices(types, k=n_entries),
        'Link': [f"https://leisestaduais.com.br/{estado}/{muni}/{tipo}/{ano}/sample-{i}" 
                for i, (muni, tipo, ano) in enumerate(zip(
                    sample_muni,
                    random.choices(types, k=n_entries),
                    sample_years
                ))],
        'conteúdo': [random.choice(content_templates).format(muni=muni) 
                    for muni in sample_muni]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Make sure all years in the range are represented
    years_set = set(str(y) for y in years)
    df_years = set(df['ano'].unique())
    
    # Add some entries for missing years if any
    for missing_year in years_set - df_years:
        # Add at least 3 entries for each missing year
        for _ in range(3):
            muni = random.choice(municipalities)
            tipo = random.choice(types)
            new_row = {
                'município': muni,
                'ano': missing_year,
                'tipo': tipo,
                'Link': f"https://leisestaduais.com.br/{estado}/{muni}/{tipo}/{missing_year}/sample-added",
                'conteúdo': random.choice(content_templates).format(muni=muni)
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Make sure we have a good distribution of municipalities
    muni_counts = df['município'].value_counts()
    underrepresented = [m for m in municipalities if m not in muni_counts or muni_counts[m] < 2]
    
    # Add entries for underrepresented municipalities
    for muni in underrepresented:
        # Add 2-4 entries for each underrepresented municipality
        for _ in range(random.randint(2, 4)):
            ano = random.choice([str(y) for y in years])
            tipo = random.choice(types)
            new_row = {
                'município': muni,
                'ano': ano,
                'tipo': tipo,
                'Link': f"https://leisestaduais.com.br/{estado}/{muni}/{tipo}/{ano}/sample-added-muni",
                'conteúdo': random.choice(content_templates).format(muni=muni)
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    return df

# --- Main App ---

def main():
    st.title('📜 Brazilian Municipal Laws Dashboard')
    st.write("""
        This dashboard allows you to explore municipal laws across Brazilian states. 
        Select a state and enter a search term to analyze laws containing that term.
    """)
    
    with st.sidebar:
        st.header("Search Parameters")
        estado = st.selectbox('Select a state:', estados_brasileiros, index=23)  # Default to 'sc'
        query = st.text_input('Enter a search query:', value='startup')
        
        # Time range selection
        st.subheader("Time Range")
        current_year = datetime.now().year
        col1, col2 = st.columns(2)
        with col1:
            start_year = st.number_input("Start Year", min_value=1980, max_value=current_year, value=current_year-10)
        with col2:
            end_year = st.number_input("End Year", min_value=1980, max_value=current_year, value=current_year)
        
        if start_year > end_year:
            st.error("Start year must be less than or equal to end year")
            start_year, end_year = end_year, start_year
        
        st.markdown("---")
        st.markdown("### Filters")
        selected_year = st.selectbox(
            "Select year to filter visualizations (optional):",
            options=["All years"] + [str(y) for y in range(end_year, start_year-1, -1)],
            index=0
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
            This app simulates data from [leisestaduais.com.br](https://leisestaduais.com.br) 
            and visualizes municipal laws across Brazilian states.
            
            The data shown is simulated to represent all municipalities in the selected state
            and time range, with realistic distributions of law types and frequency.
        """)
    
    if st.button('Generate Dashboards', type="primary"):
        with st.spinner(f'Generating representative data for all municipalities in {estado.upper()} from {start_year} to {end_year}...'):
            test = get_complete_sample_data(estado, query, start_year, end_year)
        
        # Continue only if we have data
        if not test.empty:
            # Data preparation
            df_final = test.copy()[['ano', 'município', 'tipo']]
            df_final['Counts'] = df_final.groupby(['ano', 'município'])['tipo'].transform('count')
            df_final.drop_duplicates(inplace=True)

            # Check for and handle non-numeric years
            try:
                df_final['ano'] = pd.to_numeric(df_final['ano'], errors='coerce')
                df_final = df_final.dropna(subset=['ano'])
                df_final['ano'] = df_final['ano'].astype(str)
            except Exception as e:
                st.warning(f"Some year values could not be processed: {e}")

            # Generating cumulative time series
            try:
                series = df_final.copy()
                series_t = series.groupby(by=['ano', 'município']).sum()
                final_time_series = series_t.unstack().T.loc['Counts'].fillna(0).T.cumsum()
            except Exception as e:
                st.error(f"Error generating time series: {str(e)}")
                final_time_series = pd.DataFrame()
                
            # Preparing text for word cloud
            try:
                summary = test['conteúdo'].values
                all_summary = " ".join(s for s in summary).replace('"', '').lower().replace("-", "").replace("nº", "").replace('.', '')
                stop_words = set(STOPWORDS)
                stop_words.update(["da", "meu", "em", "você", "de", "ao", "os", "e", "o", "a", 'para', 'à', 'dispõe', 'dá', 'outras'])
            except Exception as e:
                st.error(f"Error preparing word cloud text: {str(e)}")
                all_summary = "No text available for word cloud"
                stop_words = set(STOPWORDS)

            # Display results
            st.success(f"Generated data for {test['município'].nunique()} municipalities in {estado.upper()} from {start_year} to {end_year}!")
            
            # Create metrics section
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Laws", len(test))
            with col2:
                st.metric("Municipalities", test['município'].nunique())
            with col3:
                st.metric("Year Range", f"{test['ano'].min()} - {test['ano'].max()}")
            with col4:
                st.metric("Law Types", test['tipo'].nunique())
            
            # Create tabs for better organization
            tab1, tab2, tab3 = st.tabs(["📊 Overview", "🗺️ Geographical Analysis", "📑 Detailed Data"])
            
            with tab1:
                st.subheader(f"Top Municipalities with Laws about '{query}'")
                try:
                    treemap_df = df_final.groupby('município').sum()['Counts'].to_frame().reset_index().rename(columns={'Counts': 'Values'})
                    st.dataframe(treemap_df.sort_values(by='Values', ascending=False).head(10).reset_index(drop=True))
                except Exception as e:
                    st.error(f"Error displaying top municipalities: {str(e)}")
                
                st.subheader("Word Cloud of Law Contents")
                plot_wordcloud(all_summary, stop_words)
                
                st.subheader("Law Type Distribution")
                plot_pizza_leis(df=df_final, ano=selected_year if selected_year != "All years" else None)
                
                st.subheader("Evolution of Laws Over Time")
                plot_time_series(final_time_series)
                
            with tab2:
                st.subheader("Geographical Distribution")
                
                # Only try to map if we have treemap_df defined
                if 'treemap_df' in locals():
                    # Check if treemap_df is not empty before trying to map
                    if not treemap_df.empty:
                        state_code = estado.upper()
                        map_values_to_regions(treemap_df, state_code)
                    else:
                        st.warning("No geographical data available to map.")
                else:
                    st.warning("No geographical data available to map.")
                
                st.subheader("Treemap of Laws by Municipality")
                plot_treemap(final_time_series)
                
            with tab3:
                st.subheader("Raw Data Preview")
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.dataframe(test.head(50), use_container_width=True)
                with col2:
                    if not test.empty:
                        st.download_button(
                            label="Download Complete Data as CSV",
                            data=test.to_csv(index=False).encode('utf-8'),
                            file_name=f"laws_{estado}_{query}_{start_year}_{end_year}_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                        )
                    
                    st.markdown("### Data Summary")
                    if not test.empty:
                        st.write(f"- **Law Types**: {test['tipo'].nunique()} different types")
                        st.write(f"- **Most Common Type**: {test['tipo'].value_counts().idxmax()}")
                        st.write(f"- **Most Active Year**: {test['ano'].value_counts().idxmax()}")
                        st.write(f"- **Most Active Municipality**: {test['município'].value_counts().idxmax()}")
                    else:
                        st.write("No data available for summary.")
                
                # Add municipality coverage information
                st.subheader("Municipality Coverage")
                muni_counts = test['município'].value_counts().reset_index()
                muni_counts.columns = ['Municipality', 'Law Count']
                
                # Create visualization of municipality coverage
                fig = px.bar(
                    muni_counts.sort_values('Law Count', ascending=False).head(20),
                    x='Municipality', 
                    y='Law Count',
                    title=f'Top 20 Municipalities by Law Count (out of {len(muni_counts)})'
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Critical error: {str(e)}")
        st.error("Please refresh the page and try again with different parameters.")
