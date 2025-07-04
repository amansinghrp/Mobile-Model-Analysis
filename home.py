import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
import os

# File paths
RAW_DATA_PATH = "Mobiles_Dataset.csv"
CLEANED_DATA_PATH = "Cleaned_Mobiles_Dataset2.csv"

# Define preprocessing function
def preprocess_data():
    df = pd.read_csv(RAW_DATA_PATH)

    # Fix inconsistent brand naming
    df['Product Name'] = df['Product Name'].str.replace('I kall', 'IKall', case=False)
    df['Brand'] = df['Product Name'].str.split().str[0].str.lower()

    # Drop unnecessary columns
    df.drop(['Description', 'Link'], axis=1, inplace=True)

    # Clean 'Actual price' column
    df['Actual price'] = (
        df['Actual price']
        .str.replace('₹', '', regex=False)
        .str.replace(',', '', regex=False)
    )
    df['Actual price'] = pd.to_numeric(df['Actual price'], errors='coerce')

    # Clean 'Discount price' column
    df['Discount price'] = (
        df['Discount price']
        .str.replace('₹', '', regex=False)
        .str.replace(',', '', regex=False)
    )
    df['Discount price'] = pd.to_numeric(df['Discount price'], errors='coerce')
    df.dropna(subset=['Actual price', 'Discount price'], inplace=True)
    # Convert to integer
    df['Actual price'] = df['Actual price'].astype(int)
    df['Discount price'] = df['Discount price'].astype(int)
    
    #fix ram and storage columns
    def extract_ram(ram_str):
        if pd.isna(ram_str):
            return None
        match = re.search(r'(\d+)', ram_str)
        return int(match.group(1)) if match else None
    def extract_storage(storage_str):
        if pd.isna(storage_str):
            return None
        match = re.search(r'(\d+)', storage_str)
        return int(match.group(1)) if match else None

    df['RAM (GB)'] = df['RAM (GB)'].apply(extract_ram).astype("Int64")
    df['Storage (GB)'] = df['Storage (GB)'].apply(extract_storage).astype("Int64")

    # Clean Ratings and Reviews
    df['Rating'] = df['Rating'].str.replace(' Ratings', '').str.replace(',', '').astype(int)
    df['Reviews'] = df['Reviews'].str.replace(' Reviews', '').str.replace(',', '').astype(int)

    # Extract primary camera
    def extract_primary_camera(mp_str):
        if pd.isna(mp_str): return None
        match = re.search(r'(\d+)', mp_str)
        return int(match.group(1)) if match else None
    df['Primary Camera (MP)'] = df['Camera'].apply(extract_primary_camera)
    df.drop('Camera', axis=1, inplace=True)

    # Segment classification
    def classify_segment(price):
        if price < 10000:
            return 'Budget'
        elif price <= 30000:
            return 'Mid-Range'
        else:
            return 'Flagship'
    df['Segment'] = df['Actual price'].apply(classify_segment)

    # Standard RAM sizes (GB)
    valid_ram = [2, 4, 6, 8, 12, 16, 18, 24] 


    def round_to_nearest_valid(value, valid_options):
        if pd.isna(value):
            return value
        # Find the closest valid value
        closest = min(valid_options, key=lambda x: abs(x - value))
        return closest

    # Round RAM values
    df['RAM (GB)'] = df['RAM (GB)'].apply(lambda x: round_to_nearest_valid(x, valid_ram))

    #Hanlding Missing values
    # First, let's define a function to get the mode for a series, handling cases where mode might be empty
    def get_mode(series):
        mode_values = series.mode()
        if not mode_values.empty:
            return mode_values.iloc[0]  # Return the first mode if multiple exist
        return None

    # For RAM (GB)
    df['RAM (GB)'] = df.groupby(['Brand', 'Segment'])['RAM (GB)'].transform(
        lambda x: x.fillna(get_mode(x)) if get_mode(x) is not None else x.fillna(df['RAM (GB)'].mode()[0])
    )

    # For Storage (GB)
    df['Storage (GB)'] = df.groupby(['Brand', 'Segment'])['Storage (GB)'].transform(
        lambda x: x.fillna(get_mode(x)) if get_mode(x) is not None else x.fillna(df['Storage (GB)'].mode()[0])
    )

    # For Primary Camera (MP)
    df['Primary Camera (MP)'] = df.groupby(['Brand', 'Segment'])['Primary Camera (MP)'].transform(
        lambda x: x.fillna(get_mode(x)) if get_mode(x) is not None else x.fillna(df['Primary Camera (MP)'].mode()[0])
    )

    # Remove extreme values
    df = df[df['RAM (GB)'] < 24]

    # Save cleaned file
    df.to_csv(CLEANED_DATA_PATH, index=False)
    return df

# Streamlit app structure
st.set_page_config(page_title="Mobile Market Analysis", layout="wide")
tabs = st.tabs(["Overview", "Preprocessing", "EDA", "Brand Explorer", "Model Explorer"])

# Overview Tab
with tabs[0]:
    st.title("📊 Mobile Market Analysis (India - Flipkart)")
    st.markdown("""
    This Streamlit app performs an exploratory data analysis on a dataset of mobile phones sold in India, scraped from Flipkart. 
    It includes data preprocessing, price and feature analysis, and brand-specific insights.
    """)
    if os.path.exists(RAW_DATA_PATH):
        raw_df = pd.read_csv(RAW_DATA_PATH)
        st.subheader("Raw Dataset Sample")
        st.dataframe(raw_df, height=300, use_container_width=True)
    else:
        st.warning("Raw dataset not found.")

# Preprocessing Tab
with tabs[1]:
    st.title("🧹 Data Preprocessing")

    if st.button("Run Preprocessing"):
        cleaned_df = preprocess_data()
        st.success("Preprocessing complete. Cleaned data saved.")
        st.dataframe(cleaned_df, height=300, use_container_width=True)
    elif os.path.exists(CLEANED_DATA_PATH):
        cleaned_df = pd.read_csv(CLEANED_DATA_PATH)
        st.info("Loaded previously cleaned data.")
        st.dataframe(cleaned_df, height=300, use_container_width=True)
    else:
        st.warning("Please upload raw data or run preprocessing first.")

# EDA Tab Placeholder
with tabs[2]:
    st.title("📈 Exploratory Data Analysis")
    if os.path.exists(CLEANED_DATA_PATH):
        df = pd.read_csv(CLEANED_DATA_PATH)

        analysis_type = st.selectbox("Select the type of EDA you'd like to explore:",
                                     ["Brand-Wise", "Model-Wise" , "Feature-Wise", "Distribution & Correlation"])

        if analysis_type == "Brand-Wise":
            st.subheader("Top Brands by Number of Models")
            top_brands = df['Brand'].value_counts().head(10).reset_index()
            top_brands.columns = ['Brand', 'Number of Models']

            fig1 = px.bar(
                top_brands,
                x='Brand',
                y='Number of Models',
                title='Top 10 Brands by Number of Models',
                color='Number of Models',
                color_continuous_scale='reds'
            )

            fig1.update_layout(
                xaxis_title='Brand',
                yaxis_title='Number of Models',
                xaxis_tickangle=-45,
                hovermode='x unified'
            )
            st.plotly_chart(fig1, use_container_width=True)

            #Average Discount price by brand
            st.subheader("Average Dsicount Price per Brand")
            # 1. Data Preparation
            avg_discount_price = (
                df.groupby('Brand')['Discount price']
                .mean()
                .sort_values(ascending=False)
                .head(10)
                .reset_index()
            )
            avg_discount_price.columns = ['Brand', 'Avg_Discount_Price']

            # 2. Create Horizontal Bar Chart
            fig2 = px.bar(
                avg_discount_price,
                y='Brand',
                x='Avg_Discount_Price',
                orientation='h',
                title='<b>Top 10 Brands by Average Discount Price</b>',
                labels={'Avg_Discount_Price': 'Average Discount Price (₹)'},
                color='Avg_Discount_Price',
                color_continuous_scale=['#4CB5F5', '#BED3E4'],  # Blue gradient
                text='Avg_Discount_Price',
                height=600
            )

            # 3. Formatting with Pure White Background
            fig2.update_layout(
                plot_bgcolor='white',  # Pure white plot area
                paper_bgcolor='white',  # Pure white surrounding area
                xaxis=dict(
                    showgrid=True,
                    gridcolor='#f0f0f0',  # Very light gray gridlines
                    title_font=dict(size=14),
                    color='#2B2828'
                ),
                yaxis=dict(
                    title_font=dict(size=14),
                    tickfont=dict(size=12),
                    color='#2B2828'
                ),
                xaxis_range=[0, avg_discount_price['Avg_Discount_Price'].max() * 1.15],  # 15% extra space
                margin=dict(l=150, r=50, t=80, b=50),
                title_font=dict(size=18, color='#333333'),
                hoverlabel=dict(
                    bgcolor='white',
                    font_size=12,
                    font_family="Arial",
                    bordercolor='#ddd'
                )
            )

            # 4. Bar Styling
            fig2.update_traces(
                texttemplate='₹%{x:,.0f}',  # No decimals for cleaner look
                textposition='outside',
                marker_line_color="#3B3737",
                marker_line_width=1,
                width=0.7,
                textfont=dict(color='#333333', size=11)
            )

            # 5. The Red Dotted Line Explained (Industry Benchmark)
            overall_avg = df['Discount price'].mean()
            fig2.add_vline(
                x=overall_avg,
                line_dash="dot",
                line_color="#FF0000",  # Bright red
                line_width=2,
                annotation_text=f'Industry Average: ₹{overall_avg:,.0f}',
                annotation_position="top right",
                annotation_font=dict(color="#FF0000"),
                annotation_bgcolor="rgba(255,255,255,0.8)"
            )
            st.plotly_chart(fig2, use_container_width=True)

            
            
            st.subheader("Discount % per Brand")
            # 1. Calculate Discount Percentage
            df['Discount %'] = ((df['Actual price'] - df['Discount price']) / df['Actual price']) * 100

            # 2. Prepare Data
            avg_discount_pct = (
                df.groupby('Brand')['Discount %']
                .mean()
                .sort_values(ascending=False)
                .head(10)
                .reset_index()
            )
            avg_discount_pct.columns = ['Brand', 'Avg_Discount_Pct']

            # 3. Create Interactive Horizontal Bar Chart
            fig3 = px.bar(
                avg_discount_pct,
                y='Brand',
                x='Avg_Discount_Pct',
                orientation='h',
                title='<b>Top 10 Brands by Average Discount Percentage</b>',
                labels={'Avg_Discount_Pct': 'Discount %'},
                color='Avg_Discount_Pct',
                color_continuous_scale=['#2ecc71', '#27ae60'],  # Green gradient
                text='Avg_Discount_Pct',
                height=600
            )

            # 4. Professional Formatting
            fig3.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis=dict(
                    title='Average Discount %',
                    showgrid=True,
                    gridcolor='#f0f0f0',
                    range=[0, avg_discount_pct['Avg_Discount_Pct'].max() * 1.1],  # 10% buffer
                    ticksuffix='%'  # Add % to x-axis values
                ),
                yaxis=dict(
                    title='Brand',
                    autorange='reversed'  # Highest discount on top
                ),
                margin=dict(l=150, r=50, t=80, b=50),
                title_font=dict(size=18, color='#333333'),
                hoverlabel=dict(
                    bgcolor='white',
                    font_size=12,
                    bordercolor='#ddd'
                ),
                coloraxis_showscale=False
            )

            # 5. Smart Text Handling
            fig3.update_traces(
                texttemplate='%{x:.1f}%',  # 1 decimal place with %
                textposition=[
                    'inside' if (pct > avg_discount_pct['Avg_Discount_Pct'].quantile(0.75)) 
                    else 'outside' 
                    for pct in avg_discount_pct['Avg_Discount_Pct']
                ],
                textfont_color=[
                    'white' if (pct > avg_discount_pct['Avg_Discount_Pct'].quantile(0.75)) 
                    else '#333333' 
                    for pct in avg_discount_pct['Avg_Discount_Pct']
                ],
                marker_line_color='#666666',
                marker_line_width=1,
                width=0.7
            )

            # 6. Add Industry Benchmark
            industry_avg = df['Discount %'].mean()
            fig3.add_vline(
                x=industry_avg,
                line_dash="dot",
                line_color="#e74c3c",
                line_width=2,
                annotation_text=f'Industry Avg: {industry_avg:.1f}%',
                annotation_position="top right",
                annotation_font=dict(color="#e74c3c"),
                annotation_bgcolor="rgba(255,255,255,0.8)"
            )
            st.plotly_chart(fig3, use_container_width=True)
            
            
            
            st.subheader("Brand-wise Distribution Across Price Segments")
            # Prepare data (same as before)
            all_brands = df['Brand'].value_counts().head(23).index
            df_all_brands = df[df['Brand'].isin(all_brands)]
            brand_segment = pd.crosstab(df_all_brands['Brand'], df_all_brands['Segment'])
            brand_segment = brand_segment.sort_values(by='Flagship', ascending=False)

            # Reset index for Plotly
            brand_segment = brand_segment.reset_index().melt(id_vars='Brand', var_name='Segment', value_name='Count')

            # Define a professional color scheme (adjust as needed)
            segment_colors = {
                "Flagship": "#4E79A7",  # Deep blue (Premium)
                "Mid Range": "#59A14F",  # Green (Balanced)
                "Budget": "#F28E2B",     # Orange (Affordable)
            }

            # Create interactive plot
            fig4_1 = px.bar(
                brand_segment,
                x="Brand",
                y="Count",
                color="Segment",
                title="<b>Brand-wise Distribution Across Price Segments</b>",
                color_discrete_map=segment_colors,  # Apply custom colors
                height=600,
                width=1000,
                template="plotly_white",  # Clean white background
            )

            # Customize layout
            fig4_1.update_layout(
                barmode="stack",
                xaxis_title="<b>Brand</b>",
                yaxis_title="<b>Number of Models</b>",
                plot_bgcolor="rgba(255,255,255,1)",  # Pure white background
                paper_bgcolor="rgba(255,255,255,1)",  # No gray borders
                title_font_size=20,
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=12,
                    font_family="Arial",
                ),
                legend_title_text="<b>Price Segment</b>",
                xaxis=dict(tickangle=45, showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="lightgray"),  # Light gridlines for readability
                margin=dict(l=50, r=50, b=100, t=100, pad=4),
                font=dict(family="Arial", size=12),  # Consistent font
            )

            # Customize bars & hover
            fig4_1.update_traces(
                hovertemplate="<b>%{x}</b><br>Segment: %{fullData.name}<br>Models: %{y}<extra></extra>",
                marker_line_width=0.3,
                marker_line_color="white",  # Subtle white borders for segment separation
                opacity=0.9,  # Slightly transparent for modern look
            )

            # Add interactivity
            fig4_1.update_layout(
                hovermode="x unified",  # Shows all segments on hover
                xaxis=dict(showspikes=True, spikemode="across"),  # Hover guides
            )
            st.plotly_chart(fig4_1, use_container_width=True)
            
        elif analysis_type == "Model-Wise":
            # 1. Prepare Data
            top_expensive = (
                df.sort_values(by='Actual price', ascending=False)
                .head(10)
                .reset_index(drop=True)
            )

            # 2. Create Interactive Horizontal Bar Chart
            fig5 = px.bar(
                top_expensive,
                y='Product Name',
                x='Actual price',
                orientation='h',
                title='<b>Top 10 Most Expensive Mobile Models</b>',
                labels={'Actual price': 'Price (₹)'},
                color='Actual price',
                color_continuous_scale=['#ff9999', '#ff4d4d', '#cc0000'],  # Red gradient
                text='Actual price',
                hover_data={'Brand': True, 'Discount price': True},
                height=700  # Extra height for model names
            )

            # 3. Professional Formatting
            fig5.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis=dict(
                    title='Price (₹)',
                    showgrid=True,
                    gridcolor='#f0f0f0',
                    tickprefix='₹',
                    range=[0, top_expensive['Actual price'].max() * 1.05]  # 5% buffer
                ),
                yaxis=dict(
                    title='Model',
                    autorange='reversed',  # Most expensive on top
                    tickfont=dict(size=12)
                ),
                margin=dict(l=200, r=50, t=80, b=50),  # Extra left margin
                title_font=dict(size=20, color='#333333'),
                hoverlabel=dict(
                    bgcolor='white',
                    font_size=12,
                    bordercolor='#ddd'
                )
            )

            # 4. Smart Text Handling
            fig5.update_traces(
                texttemplate='₹%{x:,.0f}',
                textposition='auto',
                marker_line_color='#660000',
                marker_line_width=1,
                width=0.8,
                textfont=dict(color='#333333', size=9)
            )

            # 5. Add Brand Information
            fig5.update_traces(
                customdata=top_expensive[['Brand', 'Discount price']],
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Brand: %{customdata[0]}<br>"
                    "Actual Price: ₹%{x:,.0f}<br>"
                    "Discount Price: ₹%{customdata[1]:,.0f}<extra></extra>"
                )
            )
            st.plotly_chart(fig5, use_container_width=True)
            
            

            st.subheader("Top Models By Ratings")
            # Get top 10 models by star rating
            top_rated = df.sort_values(by='Stars', ascending=False).drop_duplicates('Product Name').head(10)

            # Create interactive plot
            fig6 = px.bar(top_rated,
                        x='Stars',
                        y='Product Name',
                        orientation='h',
                        title='<b>Top 10 Highest Rated Mobile Models</b>',
                        color='Stars',
                        color_continuous_scale='Tealgrn',
                        text='Stars',
                        hover_data={'Stars': ':.1f', 'Product Name': True},
                        height=600)

            # Customize layout
            fig6.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                xaxis_title='<b>Star Rating</b>',
                yaxis_title=None,
                xaxis_range=[0, 5],
                plot_bgcolor='rgba(0,0,0,0)',
                title_font_size=20,
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=12,
                )
            )

            # Customize bars and text
            fig6.update_traces(
                texttemplate='%{text:.1f}',
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Rating: %{x:.1f} stars<extra></extra>',
                marker_line_color='rgb(8,48,107)',
                marker_line_width=0.5
            )
            st.plotly_chart(fig6, use_container_width=True)
            
            

            st.subheader("Most Reviewed Models")
            # Get top 10 reviewed models
            top_reviewed = df.sort_values(by='Reviews', ascending=False).drop_duplicates('Product Name').head(10)

            # Create interactive plot
            fig7 = px.bar(top_reviewed,
                        x='Reviews',
                        y='Product Name',
                        orientation='h',
                        title='<b>Top 10 Most Reviewed Mobile Models</b>',
                        color='Reviews',
                        color_continuous_scale='Blues',
                        text='Reviews',
                        hover_data={'Reviews': ':,.0f', 'Product Name': True},
                        height=600)

            # Customize layout
            fig7.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                xaxis_title='<b>Number of Reviews</b>',
                yaxis_title=None,
                plot_bgcolor='rgba(0,0,0,0)',
                title_font_size=20,
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=12,
                ),
                # Format x-axis to show thousands with comma separator
                xaxis=dict(
                    tickformat=',.0f'
                )
            )

            # Customize bars and text
            fig7.update_traces(
                texttemplate='%{text:,.0f}',
                textposition='auto',
                hovertemplate='<b>%{y}</b><br>Reviews: %{x:,.0f}<extra></extra>',
                marker_line_color='rgb(8,48,107)',
                marker_line_width=0.5
            )
            st.plotly_chart(fig7, use_container_width=True)

        elif analysis_type == "Feature-Wise":
            st.subheader("RAM vs Actual Price")
            fig8 = px.box(
                df,
                x='RAM (GB)',
                y='Actual price',
                title='Boxplot: Actual Price by RAM Size',
                labels={'RAM (GB)': 'RAM (GB)', 'Actual price': 'Actual Price (INR)'},
                color='RAM (GB)',  # Optional: color per RAM size
                template='plotly_white'
            )

            fig8.update_layout(title_font_size=20, title_x=0.5)
            st.plotly_chart(fig8, use_container_width=True)



            st.subheader("Storage vs Actual Price")
            fig9 = px.box(
                df,
                x='Storage (GB)',
                y='Actual price',
                title='Boxplot: Actual Price by Storage Size',
                labels={'Storage (GB)': 'Storage (GB)', 'Actual price': 'Actual Price (INR)'},
                color='Storage (GB)',
                template='plotly_white',
                points='all'
            )
            st.plotly_chart(fig9, use_container_width=True)

        elif analysis_type == "Distribution & Correlation":
            st.subheader("Price Distribution")
            fig10 = go.Figure()
            # Actual Price Histogram
            fig10.add_trace(go.Histogram(
                x=df['Actual price'],
                name='Actual Price',
                nbinsx=30,
                opacity=0.6,
                marker_color='steelblue'
            ))

            # Discount Price Histogram
            fig10.add_trace(go.Histogram(
                x=df['Discount price'],
                name='Discount Price',
                nbinsx=30,
                opacity=0.6,
                marker_color='salmon'
            ))

            # Layout Settings
            fig10.update_layout(
                title='Distribution of Actual and Discount Prices',
                xaxis_title='Price (INR)',
                yaxis_title='Number of Phones',
                barmode='overlay',
                template='plotly_white',
                legend=dict(x=0.7, y=0.95),
                xaxis=dict(
                    tickformat=',',
                    tickfont=dict(size=14),
                    title_font=dict(size=16)
                ),
                yaxis=dict(
                    tickformat=',',
                    tickfont=dict(size=14),
                    title_font=dict(size=16),
                    gridcolor='lightgrey'
                ),
                title_font=dict(size=20),
            )
            st.plotly_chart(fig10, use_container_width=True)



            st.subheader("Correlation Heatmap")
            # Select numeric columns
            numeric_cols = df[['Actual price', 'Discount price', 'RAM (GB)', 'Storage (GB)', 
                            'Primary Camera (MP)', 'Rating', 'Reviews']]

            # Compute correlation matrix
            correlation_matrix = numeric_cols.corr().round(2)

            # Convert to long-form DataFrame for heatmap
            corr_long = correlation_matrix.reset_index().melt(id_vars='index')
            corr_long.columns = ['Feature 1', 'Feature 2', 'Correlation']

            # Plot using Plotly
            fig11 = px.imshow(
                correlation_matrix,
                text_auto=True,
                color_continuous_scale='RdBu_r',
                zmin=-1,  
                zmax=1,  
                title='Correlation Heatmap of Numeric Features',
                aspect='auto',
                labels=dict(color='Correlation'),
            )

            # Update layout for better spacing
            fig11.update_layout(
                title_x=0.5,
                xaxis_title='Features',
                yaxis_title='Features',
                font=dict(size=14),
                margin=dict(l=40, r=40, t=60, b=40)
            )
            st.plotly_chart(fig11, use_container_width=True)
    else:
        st.warning("Please preprocess the data first to access EDA.")

# Brand Explorer Tab
with tabs[3]:
    st.title("🔍 Brand-Specific Explorer")
    if os.path.exists(CLEANED_DATA_PATH):
        df = pd.read_csv(CLEANED_DATA_PATH)
        selected_brand = st.selectbox("Select a brand to explore:", sorted(df['Brand'].unique()))
        brand_df = df[df['Brand'] == selected_brand]

        st.subheader(f"📦 All Models from '{selected_brand}'")
        st.dataframe(brand_df, height=400, use_container_width=True)

        st.subheader("📊 Model Count by Segment")
        seg_count = brand_df['Segment'].value_counts().reset_index()
        seg_count.columns = ['Segment', 'Count']
        segment_order = ['Budget', 'Mid-Range', 'Flagship']
        fig1 = px.bar(seg_count, x='Segment', y='Count', category_orders={'Segment': segment_order},
                      title=f"Model Distribution by Segment - {selected_brand}")
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("💰 Price Distribution")
        fig2 = px.histogram(brand_df, x='Actual price', nbins=20, title=f"Price Distribution - {selected_brand}")
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("🔢 RAM vs Price")
        fig3 = px.scatter(brand_df, x='RAM (GB)', y='Actual price', size='Rating', color='Segment', title=f"RAM vs Price - {selected_brand}")
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("📷 Camera vs Price")
        fig4 = px.scatter(brand_df, x='Primary Camera (MP)', y='Actual price', color='Segment', title=f"Camera vs Price - {selected_brand}")
        st.plotly_chart(fig4, use_container_width=True)
        
        st.subheader("💾 Storage vs Price")
        fig5 = px.scatter(brand_df, x='Storage (GB)', y='Actual price', color='Segment', title=f"Storage vs Price - {selected_brand}")
        st.plotly_chart(fig5, use_container_width=True)
    else:
        st.warning("Please preprocess the data first to access Brand Explorer.")
with tabs[4]:
    st.title("📱 Model Explorer")

    if os.path.exists(CLEANED_DATA_PATH):
        df = pd.read_csv(CLEANED_DATA_PATH)

        # Session state to preserve selected models across brand changes
        if "selected_models" not in st.session_state:
            st.session_state.selected_models = []

        st.subheader("Select Brands and Models to Compare")
        selected_brand = st.selectbox("Select a Brand", sorted(df['Brand'].unique()))

        available_models = df[df['Brand'] == selected_brand]['Product Name'].unique()
        selected_models = st.multiselect("Select Models from This Brand",
                                         available_models,
                                         key=f"models_{selected_brand}")

        # Add to persistent session state
        if st.button("➕ Add Models to Compare"):
            for model in selected_models:
                if model not in st.session_state.selected_models:
                    st.session_state.selected_models.append(model)

        if st.button("❌ Clear All Selections"):
            st.session_state.selected_models.clear()

        if st.session_state.selected_models:
            st.markdown(f"### 🔍 Comparing {len(st.session_state.selected_models)} Selected Models")
            compare_df = df[df['Product Name'].isin(st.session_state.selected_models)]
            st.dataframe(compare_df.reset_index(drop=True), use_container_width=True)

            # Comparison Plots
            st.subheader("📊 Price Comparison")
            fig1 = px.bar(compare_df, x='Product Name', y='Actual price', color='Brand', title='Price Comparison')
            st.plotly_chart(fig1, use_container_width=True)

            st.subheader("🔢 RAM Comparison")
            fig2 = px.bar(compare_df, x='Product Name', y='RAM (GB)', color='Brand', title='RAM Comparison')
            st.plotly_chart(fig2, use_container_width=True)

            st.subheader("💾 Storage Comparison")
            fig3 = px.bar(compare_df, x='Product Name', y='Storage (GB)', color='Brand', title='Storage Comparison')
            st.plotly_chart(fig3, use_container_width=True)

            st.subheader("⭐ Ratings vs Reviews")
            fig4 = px.scatter(compare_df, x='Rating', y='Reviews', size='Actual price',
                              color='Brand', hover_name='Product Name',
                              title='Rating vs Reviews (Bubble = Price)')
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("No models selected yet. Choose a brand and models to begin.")
    else:
        st.warning("Please preprocess the data first.")