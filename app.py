import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import plotly.express as px
import plotly.graph_objects as go

# File paths
RAW_DATA_PATH = "Mobiles_Dataset.csv"
CLEANED_DATA_PATH = "Cleaned_Mobiles_Dataset.csv"

# Define preprocessing function
def preprocess_data():
    df = pd.read_csv(RAW_DATA_PATH)

    df['Product Name'] = df['Product Name'].str.replace('I kall', 'IKall', case=False)
    df['Brand'] = df['Product Name'].str.split().str[0].str.lower()
    df.drop(['Description', 'Link'], axis=1, inplace=True)

    for col in ['Actual price', 'Discount price']:
        df[col] = df[col].astype(str).str.replace("₹", "", regex=False).str.replace(",", "", regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['Actual price', 'Discount price'], inplace=True)
    df['Actual price'] = df['Actual price'].astype(int)
    df['Discount price'] = df['Discount price'].astype(int)

    def extract_number(text):
        if pd.isna(text): return None
        match = re.search(r'(\d+)', text)
        return int(match.group(1)) if match else None
    df['RAM (GB)'] = df['RAM (GB)'].apply(extract_number).astype('Int64')
    df['Storage (GB)'] = df['Storage (GB)'].apply(extract_number).astype('Int64')

    df['Rating'] = df['Rating'].str.replace(' Ratings', '').str.replace(',', '').astype(int)
    df['Reviews'] = df['Reviews'].str.replace(' Reviews', '').str.replace(',', '').astype(int)

    def extract_primary_camera(mp_str):
        if pd.isna(mp_str): return None
        match = re.search(r'(\d+)', mp_str)
        return int(match.group(1)) if match else None
    df['Primary Camera (MP)'] = df['Camera'].apply(extract_primary_camera)
    df.drop('Camera', axis=1, inplace=True)

    def classify_segment(price):
        if price < 10000:
            return 'Budget'
        elif price <= 30000:
            return 'Mid-Range'
        else:
            return 'Flagship'
    df['Segment'] = df['Actual price'].apply(classify_segment)

    valid_ram = [2, 4, 6, 8, 12, 16, 18, 24]
    def round_to_valid(value):
        if pd.isna(value): return value
        return min(valid_ram, key=lambda x: abs(x - value))
    df['RAM (GB)'] = df['RAM (GB)'].apply(round_to_valid)

    def get_mode(series):
        mode_vals = series.mode()
        return mode_vals.iloc[0] if not mode_vals.empty else None

    for col in ['RAM (GB)', 'Storage (GB)', 'Primary Camera (MP)']:
        df[col] = df.groupby(['Brand', 'Segment'])[col].transform(
            lambda x: x.fillna(get_mode(x)) if get_mode(x) is not None else x.fillna(df[col].mode()[0])
        )

    df = df[df['RAM (GB)'] < 24]

    df.to_csv(CLEANED_DATA_PATH, index=False)
    return df

# Streamlit App Setup
st.set_page_config(page_title="Mobile Market Analysis", layout="wide")

menu = st.sidebar.selectbox("📁 Menu", (
    "Overview", 
    "Preprocessing", 
    "Brand Explorer", 
    "Model Explorer", 
    "Feature Insight"
))


if menu == "Overview":
    st.title("📊 Mobile Market Analysis (India - Flipkart)")
    st.markdown("""
    This Streamlit app performs an exploratory data analysis on a dataset of mobile phones sold in India, scraped from Flipkart. 
    It includes data preprocessing, price and feature analysis, and brand-specific insights.
    """)
    if os.path.exists(RAW_DATA_PATH):
        raw_df = pd.read_csv(RAW_DATA_PATH)
        st.subheader("Raw Dataset Sample")
        st.dataframe(raw_df, height=600, use_container_width=True)
    else:
        st.warning("Raw dataset not found.")

elif menu == "Preprocessing":
    st.title("🧹 Data Preprocessing")
    st.write("This section applies the original preprocessing pipeline from your notebook.")

    if st.button("Run Preprocessing"):
        cleaned_df = preprocess_data()
        st.success("Preprocessing complete. Cleaned data saved.")
        st.dataframe(cleaned_df, height=600, use_container_width=True)
    elif os.path.exists(CLEANED_DATA_PATH):
        cleaned_df = pd.read_csv(CLEANED_DATA_PATH)
        st.info("Loaded previously cleaned data.")
        st.dataframe(cleaned_df, height=600, use_container_width=True)
    else:
        st.warning("Please upload raw data or run preprocessing first.")

elif menu == "Brand Explorer":
    st.title("🏷️ Brand Explorer")
    
    if os.path.exists(CLEANED_DATA_PATH):
        df = pd.read_csv(CLEANED_DATA_PATH)
        all_brands = sorted(df['Brand'].unique())
        selected_brand = st.selectbox("📍 Select a Brand:", all_brands)

        brand_df = df[df['Brand'] == selected_brand]

        # Basic Info Metrics
        st.markdown("### 📊 Brand Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Models Available", len(brand_df))
        col2.metric("Avg. Actual Price", f"₹{int(brand_df['Actual price'].mean()):,}")
        col3.metric("Avg. Rating", f"{brand_df['Stars'].mean():.2f} ⭐")

        # Data table
        with st.expander("🗃️ View All Models from this Brand"):
            st.dataframe(brand_df, height=400, use_container_width=True)

        # Segment Distribution
        with st.expander("📦 Segment Distribution"):
            seg_count = brand_df['Segment'].value_counts().reset_index()
            seg_count.columns = ['Segment', 'Count']
            segment_order = ['Budget', 'Mid-Range', 'Flagship']
            fig1 = px.bar(seg_count, x='Segment', y='Count', category_orders={'Segment': segment_order},
                          title=f"Segment Distribution - {selected_brand}",
                          color='Segment', color_discrete_map={"Budget": "green", "Mid-Range": "orange", "Flagship": "red"})
           
            st.plotly_chart(fig1, use_container_width=True)

        # Price Distribution
        with st.expander("💰 Price Distribution"):
            fig2 = px.histogram(brand_df, x='Actual price', nbins=20,
                                title=f"Price Distribution - {selected_brand}",
                                color_discrete_sequence=['teal'])
            fig2.update_layout(
                bargap=0.2  # Increase spacing between bars (0 = no gap, 1 = full gap)
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Rating and Review Distribution
        with st.expander("⭐ Rating & Reviews Overview"):
            col1, col2 = st.columns(2)
            fig3a = px.box(brand_df, y='Stars', points="all", title="Rating Distribution", color_discrete_sequence=['goldenrod'])
            fig3b = px.box(brand_df, y='Reviews', points="all", title="Reviews Distribution", color_discrete_sequence=['indianred'])
            col1.plotly_chart(fig3a, use_container_width=True)
            col2.plotly_chart(fig3b, use_container_width=True)

        with st.expander("📐 RAM vs Price Distribution"):
            min_price = brand_df['Actual price'].min()
            max_price = brand_df['Actual price'].max()

            # Define 5k bins dynamically
            bin_edges = np.arange(int(min_price // 5000) * 5000, int(max_price // 5000 + 2) * 5000, 5000)
            bin_labels = [f"{int(x/1000)}k" for x in bin_edges[1:]]

            # Cut prices into bins
            brand_df['Price Bin'] = pd.cut(brand_df['Actual price'], bins=bin_edges, labels=bin_labels, include_lowest=True)

            # Group by price bin and find max RAM
            ram_df = brand_df.groupby('Price Bin').agg({'RAM (GB)': 'max'}).reset_index()

            # Function to collect all top models per bin
            def get_hover_models(group):
                max_ram = group['RAM (GB)'].max()
                return ", ".join(group[group['RAM (GB)'] == max_ram]['Product Name'].unique())

            # Hover info with all top models
            hover_info = brand_df.groupby('Price Bin').apply(get_hover_models).reset_index(name='All Top Models')

            # Pick any one model per bin to show as text (first one)
            sample_labels = brand_df.groupby(['Price Bin', 'RAM (GB)'])['Product Name'].first().reset_index()
            sample_labels = sample_labels.drop_duplicates(subset='Price Bin')
            ram_df = ram_df.merge(sample_labels[['Price Bin', 'Product Name']], on='Price Bin', how='left')
            ram_df = ram_df.merge(hover_info, on='Price Bin', how='left')

            # Plot
            fig_ram = px.bar(
                ram_df,
                x="Price Bin",
                y="RAM (GB)",
                text=None,
                hover_data={"All Top Models": True},
                title="💾 Best RAM Configuration by Price Bin",
                color="RAM (GB)",
                color_continuous_scale="Blues"
            )

            fig_ram.update_traces(
                textposition='outside',
                hovertemplate="<b>Price Bin: %{x}</b><br>RAM: %{y} GB<br><br><b>Top Models:</b><br>%{customdata[0]}<extra></extra>"
            )

            fig_ram.update_layout(
                xaxis_title="Price Segment",
                yaxis_title="Max RAM (GB)",
                title_x=0.3,
                font=dict(size=13),
                margin=dict(l=40, r=40, t=60, b=40)
            )

            st.plotly_chart(fig_ram, use_container_width=True)
        with st.expander("💽 Storage vs Price Distribution"):
            # Reuse same binning
            brand_df['Price Bin'] = pd.cut(brand_df['Actual price'], bins=bin_edges, labels=bin_labels, include_lowest=True)

            # Max Storage per bin
            storage_df = brand_df.groupby('Price Bin').agg({'Storage (GB)': 'max'}).reset_index()

            def get_hover_models_storage(group):
                max_storage = group['Storage (GB)'].max()
                return ", ".join(group[group['Storage (GB)'] == max_storage]['Product Name'].unique())

            hover_info_storage = brand_df.groupby('Price Bin').apply(get_hover_models_storage).reset_index(name='All Top Models')

            sample_labels_storage = brand_df.groupby(['Price Bin', 'Storage (GB)'])['Product Name'].first().reset_index()
            sample_labels_storage = sample_labels_storage.drop_duplicates(subset='Price Bin')

            storage_df = storage_df.merge(sample_labels_storage[['Price Bin', 'Product Name']], on='Price Bin', how='left')
            storage_df = storage_df.merge(hover_info_storage, on='Price Bin', how='left')

            # Plot
            fig_storage = px.bar(
                storage_df,
                x="Price Bin",
                y="Storage (GB)",
                text=None,
                hover_data={"All Top Models": True},
                title="💽 Best Storage Configuration by Price Bin",
                color="Storage (GB)",
                color_continuous_scale="Purples"
            )

            fig_storage.update_traces(
                textposition='outside',
                hovertemplate="<b>Price Bin: %{x}</b><br>Storage: %{y} GB<br><br><b>Top Models:</b><br>%{customdata[0]}<extra></extra>"
            )

            fig_storage.update_layout(
                xaxis_title="Price Segment",
                yaxis_title="Max Storage (GB)",
                title_x=0.3,
                font=dict(size=13),
                margin=dict(l=40, r=40, t=60, b=40)
            )

            st.plotly_chart(fig_storage, use_container_width=True)


        with st.expander("📸 Primary Camera vs Price Distribution"):
            brand_df['Price Bin'] = pd.cut(brand_df['Actual price'], bins=bin_edges, labels=bin_labels, include_lowest=True)

            # Max Camera MP per bin
            camera_df = brand_df.groupby('Price Bin').agg({'Primary Camera (MP)': 'max'}).reset_index()

            def get_hover_models_camera(group):
                max_cam = group['Primary Camera (MP)'].max()
                return ", ".join(group[group['Primary Camera (MP)'] == max_cam]['Product Name'].unique())

            hover_info_camera = brand_df.groupby('Price Bin').apply(get_hover_models_camera).reset_index(name='All Top Models')

            sample_labels_camera = brand_df.groupby(['Price Bin', 'Primary Camera (MP)'])['Product Name'].first().reset_index()
            sample_labels_camera = sample_labels_camera.drop_duplicates(subset='Price Bin')

            camera_df = camera_df.merge(sample_labels_camera[['Price Bin', 'Product Name']], on='Price Bin', how='left')
            camera_df = camera_df.merge(hover_info_camera, on='Price Bin', how='left')

            # Plot
            fig_camera = px.bar(
                camera_df,
                x="Price Bin",
                y="Primary Camera (MP)",
                text=None,
                hover_data={"All Top Models": True},
                title="📸 Best Primary Camera by Price Bin",
                color="Primary Camera (MP)",
                color_continuous_scale="Reds"
            )

            fig_camera.update_traces(
                textposition='outside',
                hovertemplate="<b>Price Bin: %{x}</b><br>Camera: %{y} MP<br><br><b>Top Models:</b><br>%{customdata[0]}<extra></extra>"
            )

            fig_camera.update_layout(
                xaxis_title="Price Segment",
                yaxis_title="Max Camera (MP)",
                title_x=0.3,
                font=dict(size=13),
                margin=dict(l=40, r=40, t=60, b=40)
            )

            st.plotly_chart(fig_camera, use_container_width=True)

        with st.expander("💡 Segment-wise Distribution for Selected Brand"):
            # Compute segment distribution for selected brand
            segment_dist = brand_df['Segment'].value_counts().reindex(['Budget', 'Mid-Range', 'Flagship'], fill_value=0).reset_index()
            segment_dist.columns = ['Segment', 'Count']

            # Define a professional color scheme
            segment_colors = {
                "Flagship": "#4E79A7",     # Deep blue
                "Mid-Range": "#59A14F",    # Green
                "Budget": "#F28E2B",       # Orange
            }

            fig_pie = px.pie(
                segment_dist,
                names='Segment',
                values='Count',
                title=f"<b>Segment Distribution for {selected_brand.capitalize()}</b>",
                color='Segment',
                color_discrete_map=segment_colors,
                hole=0.4  # donut-style
            )

            fig_pie.update_traces(
                textinfo='label+percent',
                textfont=dict(size=14, family='Arial', color='white'),
                pull=[0.05, 0.02, 0.05],  # slight "pop-out" effect
                marker=dict(line=dict(color='white', width=2)),  # border styling
                hovertemplate="<b>%{label}</b><br>Models: %{value}<br>Share: %{percent}<extra></extra>",
            )

            fig_pie.update_layout(
                title_font=dict(size=20, family="Arial", color="#EDE5E5"),
                font=dict(size=13, family="Arial", color="#333"),
                showlegend=True,
                legend_title_text="<b>Segment</b>",
                legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5),
                margin=dict(l=50, r=50, t=80, b=100),
                template="plotly_white"
            )

            st.plotly_chart(fig_pie, use_container_width=True)


    else:
        st.warning("Please preprocess the data first to access Brand Explorer.")


elif menu == "Model Explorer":
    st.title("🔍 Model Explorer")
    if os.path.exists(CLEANED_DATA_PATH):
        df = pd.read_csv(CLEANED_DATA_PATH)
        min_price = int(df['Actual price'].min())
        max_price = int(df['Actual price'].max())

        price_range = st.slider(
            "Select Price Range (₹)",
            min_value=min_price,
            max_value=max_price,
            value=(min_price, max_price),
            step=500
        )

        # Filter dataframe based on price
        df = df[(df['Actual price'] >= price_range[0]) & (df['Actual price'] <= price_range[1])]

        st.subheader("Select Brands and Models")
        brands = sorted(df['Brand'].unique())
        selected_brands = st.multiselect("Select brand(s):", brands)

        model_options = sorted(df[df['Brand'].isin(selected_brands)]['Product Name'].unique())
        selected_models = st.multiselect("Select model(s):", model_options)

        compare_df = df[df['Product Name'].isin(selected_models)]
        # Drop duplicate models by keeping highest rated one (or use some other logic)
        compare_df = compare_df.sort_values('Rating', ascending=False).drop_duplicates(subset='Product Name')
        if not compare_df.empty:
            st.subheader("📋 Model Details")
            st.dataframe(compare_df[['Product Name', 'Brand', 'Actual price', 'RAM (GB)', 'Storage (GB)', 'Primary Camera (MP)', 'Rating', 'Reviews']], use_container_width=True)
            st.subheader("📊 Model Comparison - Price")
            fig_price = px.bar(compare_df, x='Product Name', y='Actual price', color='Brand', title="Price Comparison")
            fig_price.update_layout(
                barmode='group',  # ✅ Prevent stacking
            )
            st.plotly_chart(fig_price, use_container_width=True)


            st.subheader("💸 Actual vs Discount Price for Selected Models")
            fig_price = go.Figure()
            bar_width = 0.35  # Make bars narrower
            # Actual Price Bar
            fig_price.add_trace(go.Bar(
                x=compare_df['Product Name'],
                y=compare_df['Actual price'],
                name='Actual Price',
                marker_color='lightblue',
                width=[bar_width] * len(compare_df)
            ))

            # Discount Price Bar
            fig_price.add_trace(go.Bar(
                x=compare_df['Product Name'],
                y=compare_df['Discount price'],
                name='Discount Price',
                marker_color='lightcoral',
                width=[bar_width] * len(compare_df)
            ))

            # Layout
            fig_price.update_layout(
                title='Comparison of Actual and Discount Prices',
                xaxis_title='Model',
                yaxis_title='Price (₹)',
                barmode='group',
                template='plotly_white',
                xaxis=dict(tickangle=30),
                title_font=dict(size=20),
                margin=dict(b=120),
            )

            st.plotly_chart(fig_price, use_container_width=True)



            st.subheader("📊 Model Comparison - RAM")
            fig_ram = px.bar(compare_df, x='Product Name', y='RAM (GB)', color='Brand', title="RAM Comparison")
            st.plotly_chart(fig_ram, use_container_width=True)

            st.subheader("📊 Model Comparison - Storage")
            fig_storage = px.bar(compare_df, x='Product Name', y='Storage (GB)', color='Brand', title="Storage Comparison")
            st.plotly_chart(fig_storage, use_container_width=True)

            st.subheader("📊 Model Comparison - Camera")
            fig_camera = px.bar(compare_df, x='Product Name', y='Primary Camera (MP)', color='Brand', title="Primary Camera Comparison")
            st.plotly_chart(fig_camera, use_container_width=True)

            st.subheader("🌟 Ratings Distribution of Selected Models")
            fig_rating_pie = px.pie(
                compare_df,
                names="Product Name",
                values="Rating",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel  # Colorful & soft
            )

            fig_rating_pie.update_traces(
                textinfo="label+percent",
                textposition="outside",
                marker=dict(line=dict(color="white", width=2)),
                pull=[0.05] * len(compare_df),
                hovertemplate="<b>%{label}</b><br>Rating: %{value} (%{percent})<extra></extra>"
            )

            fig_rating_pie.update_layout(
            title=dict(
                text="<b>Ratings Distribution Across Selected Models</b>",
                x=0.0,
                xanchor="left",
                font=dict(size=20)
            ),
            margin=dict(t=100, b=5),  # More bottom margin for the legend
            legend=dict(
                orientation="h",     # horizontal legend
                yanchor="bottom",
                y=-0.6,              # Push it further down
                xanchor="center",
                x=0.5,
                font=dict(size=12),
            )
        )

            st.plotly_chart(fig_rating_pie, use_container_width=True)


            
            st.subheader("🗣️ Reviews Distribution of Selected Models")
            fig_reviews_pie = px.pie(
                compare_df,
                names="Product Name",
                values="Reviews",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set2  # Distinct colors
            )

            fig_reviews_pie.update_traces(
                textinfo="label+percent",
                textposition="outside",
                marker=dict(line=dict(color="white", width=2)),
                pull=[0.05] * len(compare_df),
                hovertemplate="<b>%{label}</b><br>Reviews: %{value} (%{percent})<extra></extra>"
            )

            fig_reviews_pie.update_layout(
                title=dict(
                    text="<b>Reviews Distribution Across Selected Models</b>",
                    x=0.0,                   
                    xanchor="left",
                    font=dict(size=20)
                ),
                margin=dict(t=100, b=5), 
                legend=dict(
                    orientation="h",         # Horizontal layout
                    yanchor="bottom",
                    y=-0.6,                  # Push legend lower
                    xanchor="center",
                    x=0.5,
                    font=dict(size=12)
                )
            )
            st.plotly_chart(fig_reviews_pie, use_container_width=True)
            
            if 'Display Size (inch)' in compare_df.columns:
                st.subheader("📺 Display Size Comparison")
                fig_display = px.bar(
                    compare_df,
                    x='Product Name',
                    y='Display Size (inch)',
                    color='Brand',
                    title="Display Size per Model (in inches)",
                    template='plotly_dark'
                )
                fig_display.update_layout(xaxis_title="Model", yaxis_title="Display Size (inch)")
                st.plotly_chart(fig_display, use_container_width=True)
            else:
                st.info("⚠️ Display Size data not available in the dataset.")
            
            st.subheader("📊 Segment Distribution of Selected Models")
            # Count models per segment
            segment_counts = compare_df["Segment"].value_counts().reset_index()
            segment_counts.columns = ["Segment", "Count"]

            # Model names by segment (tooltip)
            model_names_by_segment = compare_df.groupby("Segment")["Product Name"].apply(
                lambda x: "<br>".join(x)
            ).to_dict()

            segment_counts["Model Names"] = segment_counts["Segment"].apply(
                lambda seg: model_names_by_segment.get(seg, "")
            )

            # Consistent color mapping
            segment_colors = {
                "Flagship": "#4E79A7",
                "Mid-Range": "#59A14F",
                "Budget": "#F28E2B"
            }

            # Add a "gap" between slices using pull
            pull_values = [0.07 if count > 0 else 0 for count in segment_counts["Count"]]

            # Build the pie chart
            fig_pie = px.pie(
                segment_counts,
                names="Segment",
                values="Count",
                color="Segment",
                hole=0.35,  # Donut-style
                color_discrete_map=segment_colors,
            )

            # Customize traces
            fig_pie.update_traces(
                textinfo="label+percent",
                pull=pull_values,  # Pull slices slightly
                customdata=segment_counts["Model Names"],
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "Models:<br>%{customdata}<br>"
                    "Count: %{value} (%{percent})<extra></extra>"
                ),
                marker=dict(line=dict(color="white", width=2)),  # Gap between slices
            )

            # Layout beautification
            fig_pie.update_layout(
                title="<b>Segment-wise Distribution of Selected Models</b>",
                title_font_size=22,
                font=dict(size=13, family="Arial"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=80, b=20),
                showlegend=True,
                legend_title="<b>Segment</b>",
            )

            st.plotly_chart(fig_pie, use_container_width=True)



        else:
            st.info("Please select at least one valid model.")
    else:
        st.warning("Please preprocess the data first.")

elif menu == "Feature Insight":
    st.title("📐 Feature Insights")
    if os.path.exists(CLEANED_DATA_PATH):
        df = pd.read_csv(CLEANED_DATA_PATH)

        st.subheader("Correlation Heatmap")
        numeric_cols = df[['Actual price', 'Discount price', 'RAM (GB)', 'Storage (GB)', 
                           'Primary Camera (MP)', 'Rating', 'Reviews']]
        correlation_matrix = numeric_cols.corr().round(2)

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

        fig11.update_layout(
            title=dict(
                text='<b>Correlation Heatmap of Numeric Features</b>',
                x=0.5,
                xanchor='center',
                font=dict(size=20, color='#333')
            ),
            xaxis_title='Features',
            yaxis_title='Features',
            font=dict(size=14),
            margin=dict(l=40, r=40, t=100, b=40),
        )
        st.plotly_chart(fig11, use_container_width=True)
    else:
        st.warning("Please preprocess the data first to access Feature Insights.")

