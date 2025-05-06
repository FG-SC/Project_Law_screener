import streamlit as st
import pandas as pd
import numpy as np
import cloudscraper
import re
from bs4 import BeautifulSoup
import unicodedata
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

# Dictionary mapping states to some of their municipalities
estado_municipios = {
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

@st.cache_data(ttl=3600)  # Cache results for 1 hour
def webcraping_leis_municipais(query, estado='sc', paginas=1):
    text_list, links_list, lista_cidades, tipo_da_lei, ano_da_lei = [], [], [], [], []
    
    # Create a better-configured scraper with more browser-like settings
    scraper = cloudscraper.create_scraper(
        browser={
            'browser': 'chrome',
            'platform': 'windows',
            'desktop': True
        },
        delay=5  # Add some delay between requests
    )
    
    # More browser-like headers with randomization
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
    ]
    
    headers = {
        'User-Agent': random.choice(user_agents),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Cache-Control': 'max-age=0',
    }
    
    # First, try to get the main page to get cookies
    try:
        st.info("Initializing connection to the website...")
        main_url = f'https://leisestaduais.com.br/{estado}'
        init_resp = scraper.get(main_url, headers=headers, timeout=30)
        time.sleep(3)  # Wait for cookies to be set
        
        # Check if we need to display a manual intervention message
        if "Checking your browser" in init_resp.text or "Please check the box" in init_resp.text:
            st.warning("""
            ⚠️ **CAPTCHA Detected**
            
            It seems the website requires human verification. Try these steps:
            
            1. Open https://leisestaduais.com.br in your browser
            2. Solve the CAPTCHA or verification 
            3. Then return and run this app again
            
            Alternatively, try running the app with fewer pages or after waiting a while.
            """)
            return pd.DataFrame(), True  # Return empty DF and CAPTCHA flag
    except Exception as e:
        st.error(f"Error connecting to main site: {str(e)}")
        return pd.DataFrame(), True
    
    captcha_detected = False
    success_count = 0
    for i in range(1, paginas + 1):
        url = f'https://leisestaduais.com.br/{estado}?q={query}&page={i}&types=&state={estado}&status=&date_start=&date_end=&lm=1'
        
        try:
            st.info(f"Fetching page {i} of {paginas}...")
            
            # First attempt with normal settings
            result = scraper.get(url, headers=headers, timeout=30)
            
            # Check for Cloudflare or CAPTCHA challenges
            if any(phrase in result.text for phrase in ["Checking your browser", "Please check the box", "Please complete the security check"]):
                st.warning(f"Verification challenge detected on page {i}. Trying alternative approach...")
                captcha_detected = True
                
                # Create new scraper with different settings
                scraper = cloudscraper.create_scraper(
                    browser={'browser': 'firefox', 'platform': 'windows'},
                    delay=8
                )
                
                # Change user agent
                headers['User-Agent'] = random.choice(user_agents)
                
                time.sleep(5)  # Additional delay for verification
                result = scraper.get(url, headers=headers, timeout=40)
                
                # If still blocked, offer manual solution
                if any(phrase in result.text for phrase in ["Checking your browser", "Please check the box", "Please complete the security check"]):
                    st.warning(f"""
                    CAPTCHA still detected on page {i}. 
                    
                    Try these solutions:
                    1. Run this app with fewer pages (1-2)
                    2. Wait 15-30 minutes before trying again
                    3. Consider using a proxy service
                    """)
            
            soup = BeautifulSoup(result.text, 'html.parser')
            leis = soup.find_all(class_="listagem-leis")
            
            if not leis or len(leis) == 0:
                # Try to find if there's any law data at all
                any_law_content = soup.find_all(class_="btn btn-lei-lista")
                
                if any_law_content:
                    st.warning(f"Found some law content but not in expected format on page {i}. Structure may have changed.")
                else:
                    st.warning(f"No laws found on page {i}. Page might be blocked or empty.")
                
                # Try to detect if blocked vs. empty results
                if "Nenhum resultado encontrado" in result.text:
                    st.info("The search returned no results. Try different search terms.")
                    break  # No point continuing pagination with no results
                
                continue
                
            # Extract law data
            for lei in leis[0].find_all(class_="btn btn-lei-lista btn-lei-lista-leismunicipais"):
                try:
                    text_element = lei.find('span', {'rel': 'text'})
                    if text_element:
                        text_list.append(text_element.get_text(strip=False))
                        links_list.append(lei['href'])
                        
                        # Extract municipality, type, and year safely
                        url_parts = lei['href'].split('/')
                        if len(url_parts) >= 9:  # Ensure structure is as expected
                            lista_cidades.append(url_parts[6])
                            tipo_da_lei.append(url_parts[7])
                            ano_da_lei.append(url_parts[8])
                        else:
                            # Fallback for unexpected URL structure
                            lista_cidades.append("unknown")
                            tipo_da_lei.append("unknown")
                            ano_da_lei.append("unknown")
                except Exception as e:
                    st.warning(f"Error parsing a law item: {str(e)}")
                    continue
            
            success_count += 1
            # Variable delay between requests to avoid detection
            delay = 2 + np.random.rand() * 3  # Random delay between 2-5 seconds
            time.sleep(delay)
                
        except Exception as e:
            st.error(f"Error accessing page {i}: {str(e)}")
            time.sleep(5)  # Wait longer after an error
            continue
    
    if not text_list:
        if success_count > 0:
            st.warning("Connected to pages but found no law data. The website structure may have changed.")
        else:
            st.error("Could not retrieve any data. Website may be blocking automated access.")
        return pd.DataFrame(), captcha_detected
    
    # Process and clean the extracted text
    try:
        text_list = [re.split(r'\s+', string)[1] if len(re.split(r'\s+', string)) > 1 else string for string in text_list]
        leis_municipais = pd.DataFrame({
            'município': lista_cidades,
            'ano': ano_da_lei,
            'tipo': tipo_da_lei,
            'Link': links_list,
            'conteúdo': text_list
        })

        return leis_municipais.sort_values(by='ano', ascending=False).drop_duplicates(), captcha_detected
    except Exception as e:
        st.error(f"Error processing extracted data: {str(e)}")
        return pd.DataFrame(), captcha_detected

# --- Data Preparation Functions ---

def normalize_municipality_name(name):
    name = str(name).lower().replace(' ', '-')
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

# --- Function to generate sample data ---
def get_sample_data(estado, query):
    """Generate sample data specific to the selected state"""
    # Use municipalities from the selected state
    municipalities = estado_municipios.get(estado, ['unknown-city-1', 'unknown-city-2', 'unknown-city-3'])
    
    # Common law types across Brazil
    types = ["lei-ordinaria", "lei-complementar", "decreto", "resolucao", "portaria", "instrucao-normativa"]
    
    # Generate a realistic year range
    current_year = datetime.now().year
    years = list(range(current_year - 15, current_year + 1))
    
    # Generate 50-100 random entries
    n_entries = random.randint(50, 100)
    
    # More realistic content generation
    content_templates = [
        f"Lei sobre {query} no município de {{muni}}",
        f"Regulamentação de {query} para {{muni}}",
        f"Dispõe sobre {query} e dá outras providências em {{muni}}",
        f"Estabelece normas para {query} no âmbito municipal de {{muni}}",
        f"Altera a legislação sobre {query} em {{muni}}"
    ]
    
    # Generate sample data
    sample_data = {
        'município': random.choices(municipalities, k=n_entries),
        'ano': [str(random.choice(years)) for _ in range(n_entries)],
        'tipo': random.choices(types, k=n_entries),
        'Link': [f"https://leisestaduais.com.br/{estado}/{muni}/{tipo}/{ano}/sample-{i}" 
                for i, (muni, tipo, ano) in enumerate(zip(
                    random.choices(municipalities, k=n_entries),
                    random.choices(types, k=n_entries),
                    [str(random.choice(years)) for _ in range(n_entries)]
                ))],
        'conteúdo': [random.choice(content_templates).format(muni=muni) 
                    for muni in random.choices(municipalities, k=n_entries)]
    }
    
    return pd.DataFrame(sample_data)

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
        paginas = st.slider('Select the number of pages to scrape:', 1, 10, 3)
        
        st.markdown("---")
        st.markdown("### Filters")
        selected_year = st.selectbox(
            "Select year to filter (optional):",
            options=["All years"] + list(range(datetime.now().year, 1990, -1)),
            index=0
        )
        
        use_sample = st.checkbox("Use sample data (when CAPTCHA blocks access)", value=False)
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
            This app scrapes data from [leisestaduais.com.br](https://leisestaduais.com.br) 
            and visualizes municipal laws across Brazilian states.
            
            If you encounter CAPTCHA issues, try:
            1. Using fewer pages (1-2)
            2. Waiting 15-30 minutes before trying again
            3. Using the sample data option
        """)
    
    if st.button('Generate Dashboards', type="primary"):
        if use_sample:
            test = get_sample_data(estado, query)
            captcha_detected = False
            st.success(f"Using sample data for {estado.upper()} for demonstration purposes.")
        else:
            with st.spinner('Scraping data... This may take a few minutes due to Cloudflare protection...'):
                test, captcha_detected = webcraping_leis_municipais(query=query, estado=estado, paginas=paginas)
        
        if test.empty:
            if captcha_detected:
                st.error("CAPTCHA detected and could not be bypassed automatically.")
                st.info("Try the following options:")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Use Sample Data Instead"):
                        test = get_sample_data(estado, query)
                        st.success(f"Using sample data for {estado.upper()} for demonstration purposes.")
                with col2:
                    if st.button("Open Website in Browser"):
                        st.markdown(f"[Open leisestaduais.com.br/{estado}](https://leisestaduais.com.br/{estado})")
                        st.info("Solve the CAPTCHA in your browser, then return to this app and try again.")
            else:
                st.warning("No data found with the current search criteria. Try different parameters.")
                return
        
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
            st.success(f"Found {len(test)} laws containing '{query}' in {estado.upper()}!")
            
            # Create metrics section
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Laws Found", len(test))
            with col2:
                st.metric("Municipalities Covered", test['município'].nunique())
            with col3:
                year_range = f"{test['ano'].min()} - {test['ano'].max()}" if not test.empty else "N/A"
                st.metric("Year Range", year_range)
            
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
                            file_name=f"laws_{estado}_{query}_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                        )
                    
                    st.markdown("### Data Summary")
                    if not test.empty:
                        st.write(f"- **Law Types**: {test['tipo'].nunique()} different types")
                        st.write(f"- **Most Common Type**: {test['tipo'].value_counts().idxmax()}")
                        st.write(f"- **Most Active Year**: {test['ano'].value_counts().idxmax()}")
                    else:
                        st.write("No data available for summary.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Critical error: {str(e)}")
        st.error("Please refresh the page and try again with different parameters.")
