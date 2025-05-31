import faicons as fa
import plotly.express as px
from shinywidgets import render_plotly, render_widget, output_widget
from ipyleaflet import Map, Marker, AwesomeIcon
import ipywidgets as widgets
from shiny import reactive, render, ui, App
import pandas as pd
import plotly.express as px 
from datetime import date
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
import math
from pathlib import Path
from shared import app_dir, ocean_climate, confusion_matrix_dir

# Prepare data
ocean_climate['Date'] = pd.to_datetime(ocean_climate['Date'])
date_range = (ocean_climate['Date'].min(), ocean_climate['Date'].max())
locations = ocean_climate['Location'].unique()
ocean_climate['Marine Heatwave'] = ocean_climate['Marine Heatwave'].astype(bool)
ocean_climate2 = ocean_climate.copy()
ocean_climate2['Year'] = ocean_climate2['Date'].dt.year
ocean_climate2 = ocean_climate2.replace(np.nan,'Unknown')

# Define icons
ICONS = {
    "map-pin": fa.icon_svg("map-pin"),
    "temperature-high": fa.icon_svg("temperature-high"),
    "flask": fa.icon_svg("flask"),
    "fire": fa.icon_svg("fire"),
    "triangle-exclamation": fa.icon_svg("triangle-exclamation")
}

# Define UI
app_ui = ui.page_fluid(
    ui.panel_title("SHIFTING SEAS: OCEAN CLIMATE AND MARINE ECOSYSTEM DASHBOARD"),
    ui.include_css("styles.css"),
    ui.layout_column_wrap(
        1/2,
        ui.input_date_range(
            "date_range_slider",
            "Date Range",
            start=date_range[0].date(),
            end=date_range[1].date(),
            min=date_range[0].date(),
            max=date_range[1].date(),
            format="yyyy-mm-dd",
            separator=" to ",
        ),
        ui.input_select(
            "location_selector",
            "Location",
            choices=["All Locations"] + locations.tolist(),
            selected="All Locations",
        ),
        #ui.input_action_button("reset", "Reset filter"),
    ),
    ui.layout_columns(
        ui.value_box(
            title="No. of Locations",
            value=ui.output_text("num_locations_box"),
            showcase=ICONS["map-pin"]
        ),
        ui.value_box(
            title="Avg. Global SST Level",
            value=ui.output_text("avg_sst_box"),
            showcase=ICONS["temperature-high"]
        ),
        ui.value_box(
            title="Avg. Global pH Level",
            value=ui.output_text("avg_ph_box"),
            showcase=ICONS["flask"]
        ),
        col_widths=[4, 4, 4],
    ),
    ui.h4("Interactive Map: Bleaching Severity Visualization"),
    output_widget("map"),

    ui.h4("SST, pH, and Species Observed Analysis"),
    ui.layout_columns(
        ui.card(
            ui.card_header("SST Over Time"),
            ui.output_plot("sst_plot"),
            style="height: 600px;" 
        ),
        ui.card(
            ui.card_header("SST Distribution"),
            ui.output_plot("sst_distribution"),
            style="height: 600px;"
        ),
        col_widths=[6,6],
    ),

    ui.layout_columns(
        ui.card(
            ui.card_header("pH Level Over Time"),
            ui.output_plot("ph_plot"),
            style="height: 600px;"
        ),
        ui.card(
            ui.card_header("pH Level Distribution"),
            ui.output_plot("pH_distribution"),
            style="height: 600px;"
        ),
        col_widths=[6,6],
    ),

    ui.layout_columns(
        ui.card(
            ui.card_header("Number of Species Observed Over Time"),
            ui.output_plot("species_plot"),
            style="height: 600px;"
        ),
        ui.card(
            ui.card_header("Number of Species Distribution"),
            ui.output_plot("species_distribution"),
            style="height: 600px;"
        ),
        col_widths=[6,6],
    ),

    ui.h4("Bleaching Severity and Marine Heatwave Analysis"),
    ui.layout_columns(
        ui.card(
            ui.card_header("Bleaching Severity Distribution"),
            ui.output_plot("bleaching_severity_distribution"),
            style="height: 600px;"
        ),
        ui.card(
            ui.card_header("Marine Heatwave Distribution"),
            ui.output_plot("marine_heatwave_distribution"),
            style="height: 600px;"
        ),
        col_widths=[6, 6],
    ),

    ui.h4("The Impact of Ocean Acidification on Ecosystems"),
    ui.layout_columns(
        ui.card(
            ui.card_header("pH vs SST Level with Marine Heatwave"),
            ui.output_plot("ph_sst"),
            style="height: 600px;"
        ),
        ui.card(
            ui.card_header("pH vs Number of Species with Bleaching Severity"),
            ui.output_plot("ph_species"),
            style="height: 600px;"
        ),
        col_widths=[6, 6],
    ),
    ui.h4("Machine Learning-based Bleaching Severity Classification"),
    ui.layout_columns(
        ui.card(
            ui.card_header("Confusion Matrix"),
            ui.img(src= "confusion_matrix.png"),
            style="height: 700px;"
        ),
        ui.card(
            ui.card_header("Feature Importance"),
            ui.img(src= "feature_importance.png"),
            style="height: 700px;"
        ),
        col_widths=[6, 6],
    ),
)

# Calculate metrics function
def calculate_metrics(selected_location, date_range_val):
    # Ensure date_range_val contains valid dates
    start_date = pd.to_datetime(date_range_val[0]) if date_range_val[0] else date_range[0]
    end_date = pd.to_datetime(date_range_val[1]) if date_range_val[1] else date_range[1]
    
    filtered_data = ocean_climate[
        (ocean_climate['Date'] >= start_date) &
        (ocean_climate['Date'] <= end_date)
    ]
    if selected_location and selected_location != "All Locations":
        filtered_data = filtered_data[filtered_data['Location'] == selected_location]
    
    num_locations = filtered_data['Location'].nunique()
    avg_sst = filtered_data['SST (°C)'].mean() if not filtered_data.empty else 0
    avg_ph = filtered_data['pH Level'].mean() if not filtered_data.empty else 0
    return num_locations, avg_sst, avg_ph

# Severity icon colors
def severity_icon_colors(severity):
    color_map = {
        "None": "white",
        "Low": "green",
        "Medium": "orange",
        "High": "red",
    }
    return color_map.get(severity, "blue")

# Server logic
def server(input, output, session):
    @output
    @render.text
    def num_locations_box():
        selected_location = input.location_selector()
        date_range_val = input.date_range_slider()
        num, _, _ = calculate_metrics(selected_location, date_range_val)
        return f"{num}"

    @output
    @render.text
    def avg_sst_box():
        selected_location = input.location_selector()
        date_range_val = input.date_range_slider()
        _, avg_sst, _ = calculate_metrics(selected_location, date_range_val)
        return f"{avg_sst:.2f} °C"

    @output
    @render.text
    def avg_ph_box():
        selected_location = input.location_selector()
        date_range_val = input.date_range_slider()
        _, _, avg_ph = calculate_metrics(selected_location, date_range_val)
        return f"{avg_ph:.2f}"

    
    @output
    @render_widget
    def map():
        date_range_val = input.date_range_slider()
        start_date = pd.to_datetime(date_range_val[0]) if date_range_val[0] else date_range[0]
        end_date = pd.to_datetime(date_range_val[1]) if date_range_val[1] else date_range[1]
        location = input.location_selector()
        
        # Filter data based on date range
        df_filtered = ocean_climate2[
            (ocean_climate2['Date'] >= start_date) &
            (ocean_climate2['Date'] <= end_date)
        ]
        
        # Filter by location if not "All Locations"
        if location != "All Locations":
            df_filtered = df_filtered[df_filtered['Location'] == location]
        
        # Aggregate data by location for cleaner visualization
        df_agg = df_filtered.groupby(['Location', 'Latitude', 'Longitude']).agg({
            'SST (°C)': 'mean',
            'pH Level': 'mean', 
            'Species Observed': 'mean',
            'Marine Heatwave': lambda x: (x == True).sum(),  # Count of heatwave events
            'Year': lambda x: f"{x.min()}-{x.max()}" if x.min() != x.max() else str(x.min())
        }).reset_index()
        
        # Rename the Marine Heatwave column for clarity
        df_agg['Heatwave Events'] = df_agg['Marine Heatwave']
        df_agg = df_agg.drop('Marine Heatwave', axis=1)
        
        # Round values for cleaner display
        df_agg['SST (°C)'] = df_agg['SST (°C)'].round(2)
        df_agg['pH Level'] = df_agg['pH Level'].round(3)
        df_agg['Species Observed'] = df_agg['Species Observed'].round(0)
        
        # Create the scatter_geo plot
        fig = px.scatter_geo(
            df_agg,
            lat='Latitude',
            lon='Longitude',
            # Use pH Level for color to better show ocean health
            color='SST (°C)',
            color_continuous_scale='Inferno', 
            # Use SST for size to show temperature impact
            size='SST (°C)',
            size_max=20,
            projection='natural earth',
            #title='Ocean Climate Observations by Location',
            hover_data={
                'Location': True,
                'SST (°C)': ':.2f',
                'pH Level': ':.3f', 
                'Species Observed': ':.0f',
                'Heatwave Events': True,
                'Year': True,
                'Latitude': ':.2f',
                'Longitude': ':.2f'
            },
            hover_name='Location',
            labels={
                'pH Level': 'pH Level (Ocean Acidity)',
                'SST (°C)': 'Sea Surface Temp (°C)',
                'Species Observed': 'Avg Species Count',
                'Heatwave Events': 'Marine Heatwave Events'
            }
        )     
        
        return fig

    @output
    @render.plot 
    def sst_plot():
        date_range_val = input.date_range_slider()
        start_date = pd.to_datetime(date_range_val[0]) if date_range_val[0] else date_range[0]
        end_date = pd.to_datetime(date_range_val[1]) if date_range_val[1] else date_range[1]
        location = input.location_selector()
        
        # Filter data by date range
        df = ocean_climate[
            (ocean_climate['Date'] >= start_date) &
            (ocean_climate['Date'] <= end_date)
        ]
        
        # Filter by location if not "All Locations"
        if location != "All Locations":
            df = df[df['Location'] == location]
            
        # Add year column
        df = df.copy()
        df['Year'] = df['Date'].dt.year
        
        # Plot
        plt.figure(figsize=(8, 6))

        if not df.empty:
            sns.lineplot(
                data=df,
                x='Year',
                y='SST (°C)',
                hue='Location',
                marker=True,
                dashes=False,
                errorbar=('ci', 99) 
            )

        plt.xlabel('Year', fontsize=12, fontweight='bold')
        plt.ylabel('Sea Surface Temperature (°C)', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        # Horizontal legend
        handles, labels = plt.gca().get_legend_handles_labels()
        n_labels = len(labels)
        plt.legend(
            handles=handles,
            labels=labels,
            title='Location',
            loc='upper center',
            bbox_to_anchor=(0.5, 1.2),  # adjust position as needed
            ncol=math.ceil(n_labels / 2),  # split across 2 rows
            frameon=False
        )

        plt.tight_layout()

        return plt.gcf()
    
    @output
    @render.plot
    def sst_distribution():
        date_range_val = input.date_range_slider()
        start_date = pd.to_datetime(date_range_val[0]) if date_range_val[0] else date_range[0]
        end_date = pd.to_datetime(date_range_val[1]) if date_range_val[1] else date_range[1]
        location = input.location_selector()
        
        df = ocean_climate[
            (ocean_climate['Date'] >= start_date) &
            (ocean_climate['Date'] <= end_date)
        ]
        if location != "All Locations":
            df = df[df['Location'] == location]

        plt.figure(figsize=(8, 6))
        
        if not df.empty:
            sns.histplot(df['SST (°C)'], bins=30, kde=True, color='purple')
            #plt.title('Distribution of Sea Surface Temperature (°C)')
            plt.xlabel('Sea Surface Temperature (°C)',fontsize=12, fontweight='bold')
            plt.ylabel('Frequency',fontsize=12, fontweight='bold')
        else:
            plt.text(0.5, 0.5, 'No data available', horizontalalignment='center', 
                     verticalalignment='center', transform=plt.gca().transAxes)
        
        plt.grid(True)
        plt.tight_layout()
        
        return plt.gcf()

    
    @output
    @render.plot 
    def ph_plot():
        date_range_val = input.date_range_slider()
        start_date = pd.to_datetime(date_range_val[0]) if date_range_val[0] else date_range[0]
        end_date = pd.to_datetime(date_range_val[1]) if date_range_val[1] else date_range[1]
        location = input.location_selector()
        
        df = ocean_climate[
            (ocean_climate['Date'] >= start_date) &
            (ocean_climate['Date'] <= end_date)
        ]
        if location != "All Locations":
            df = df[df['Location'] == location]

        df = df.copy()
        df['Year'] = df['Date'].dt.year

        plt.figure(figsize=(8, 6))
        
        if not df.empty:
            sns.lineplot(
                data=df,
                x='Year',
                y='pH Level',
                hue='Location',
                marker=True,
                dashes=False,
                errorbar=('ci', 99) 
            )

        plt.xlabel('Year', fontsize=12, fontweight='bold')
        plt.ylabel('pH Level', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        # Horizontal legend
        handles, labels = plt.gca().get_legend_handles_labels()
        n_labels = len(labels)
        plt.legend(
            handles=handles,
            labels=labels,
            title='Location',
            loc='upper center',
            bbox_to_anchor=(0.5, 1.2),  # adjust position as needed
            ncol=math.ceil(n_labels / 2),  # split across 2 rows
            frameon=False
        )

        plt.tight_layout()

        return plt.gcf()
    
    @output
    @render.plot
    def pH_distribution():
        date_range_val = input.date_range_slider()
        start_date = pd.to_datetime(date_range_val[0]) if date_range_val[0] else date_range[0]
        end_date = pd.to_datetime(date_range_val[1]) if date_range_val[1] else date_range[1]
        location = input.location_selector()
        
        df = ocean_climate[
            (ocean_climate['Date'] >= start_date) &
            (ocean_climate['Date'] <= end_date)
        ]
        if location != "All Locations":
            df = df[df['Location'] == location]

        plt.figure(figsize=(8, 6))
        
        if not df.empty:
            sns.histplot(df['pH Level'], bins=30, kde=True, color='steelblue')
            #plt.title('Distribution of pH Level')
            plt.xlabel('pH Level',fontsize=12, fontweight='bold')
            plt.ylabel('Frequency',fontsize=12, fontweight='bold')
        else:
            plt.text(0.5, 0.5, 'No data available', horizontalalignment='center', 
                     verticalalignment='center', transform=plt.gca().transAxes)
        
        plt.grid(True)
        plt.tight_layout()
        
        return plt.gcf()
    
    @output
    @render.plot 
    def species_plot():
        date_range_val = input.date_range_slider()
        start_date = pd.to_datetime(date_range_val[0]) if date_range_val[0] else date_range[0]
        end_date = pd.to_datetime(date_range_val[1]) if date_range_val[1] else date_range[1]
        location = input.location_selector()
        
        df = ocean_climate[
            (ocean_climate['Date'] >= start_date) &
            (ocean_climate['Date'] <= end_date)
        ]
        if location != "All Locations":
            df = df[df['Location'] == location]
        
        df = df.copy()
        df['Year'] = df['Date'].dt.year
        plt.figure(figsize=(8, 6))
        if not df.empty:
            sns.lineplot(
                data=df,
                x='Year',
                y='Species Observed',
                hue='Location',
                marker=True,
                dashes=False,
                errorbar=('ci', 99) 
            )

        plt.xlabel('Year', fontsize=12, fontweight='bold')
        plt.ylabel('Species Observed', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        # Horizontal legend
        handles, labels = plt.gca().get_legend_handles_labels()
        n_labels = len(labels)
        plt.legend(
            handles=handles,
            labels=labels,
            title='Location',
            loc='upper center',
            bbox_to_anchor=(0.5, 1.2),  # adjust position as needed
            ncol=math.ceil(n_labels / 2),  # split across 2 rows
            frameon=False
        )

        plt.tight_layout()

        return plt.gcf()
    
    @output
    @render.plot
    def species_distribution():
        date_range_val = input.date_range_slider()
        start_date = pd.to_datetime(date_range_val[0]) if date_range_val[0] else date_range[0]
        end_date = pd.to_datetime(date_range_val[1]) if date_range_val[1] else date_range[1]
        location = input.location_selector()

        df = ocean_climate[
            (ocean_climate['Date'] >= start_date) &
            (ocean_climate['Date'] <= end_date)
        ]
        if location != "All Locations":
            df = df[df['Location'] == location]

        plt.figure(figsize=(8, 6))

        if not df.empty:
            sns.histplot(df['Species Observed'], bins=30, kde=True, color='orange')
            plt.xlabel('Species Observed', fontsize=12, fontweight='bold')
            plt.ylabel('Frequency', fontsize=12, fontweight='bold')
        else:
            plt.text(0.5, 0.5, 'No data available', horizontalalignment='center', 
                     verticalalignment='center', transform=plt.gca().transAxes)

        plt.grid(True)
        plt.tight_layout()

        return plt.gcf()
    
    @output
    @render.plot
    def bleaching_severity_distribution():
        date_range_val = input.date_range_slider()
        start_date = pd.to_datetime(date_range_val[0]) if date_range_val[0] else date_range[0]
        end_date = pd.to_datetime(date_range_val[1]) if date_range_val[1] else date_range[1]
        location = input.location_selector()

        df = ocean_climate[
            (ocean_climate['Date'] >= start_date) &
            (ocean_climate['Date'] <= end_date)
        ]
        if location != "All Locations":
            df = df[df['Location'] == location]
        
        plt.figure(figsize=(8, 6))

        if not df.empty:
            sns.countplot(data=df, x='Bleaching Severity',hue='Bleaching Severity')
            plt.xlabel('Bleaching Severity', fontsize=12, fontweight='bold')
            plt.ylabel('Count', fontsize=12, fontweight='bold')
        else:
            plt.text(0.5, 0.5, 'No data available', horizontalalignment='center', 
                     verticalalignment='center', transform=plt.gca().transAxes)

        plt.grid(False)
        plt.tight_layout()

        return plt.gcf()

    @output
    @render.plot
    def marine_heatwave_distribution():
        date_range_val = input.date_range_slider()
        start_date = pd.to_datetime(date_range_val[0]) if date_range_val[0] else date_range[0]
        end_date = pd.to_datetime(date_range_val[1]) if date_range_val[1] else date_range[1]
        location = input.location_selector()

        df = ocean_climate[
            (ocean_climate['Date'] >= start_date) &
            (ocean_climate['Date'] <= end_date)
        ]
        if location != "All Locations":
            df = df[df['Location'] == location]

        plt.figure(figsize=(8, 6))

        if not df.empty:
            heatwave_counts = df[df['Marine Heatwave'] == True]['Location'].value_counts()
            sns.barplot(x=heatwave_counts.values, y=heatwave_counts.index, palette='Reds')
            #sns.countplot(data=df, x='Marine Heatwave', hue='Location')
            plt.xlabel('Marine Heatwave', fontsize=12, fontweight='bold')
            plt.ylabel('Count', fontsize=12, fontweight='bold')
        else:
            plt.text(0.5, 0.5, 'No data available', horizontalalignment='center', 
                     verticalalignment='center', transform=plt.gca().transAxes)

        plt.grid(False)
        plt.tight_layout()

        return plt.gcf()

    @output
    @render.plot
    def ph_sst():
        date_range_val = input.date_range_slider()
        start_date = pd.to_datetime(date_range_val[0]) if date_range_val[0] else date_range[0]
        end_date = pd.to_datetime(date_range_val[1]) if date_range_val[1] else date_range[1]
        location = input.location_selector()

        df = ocean_climate[
            (ocean_climate['Date'] >= start_date) &
            (ocean_climate['Date'] <= end_date)
        ]
        if location != "All Locations":
            df = df[df['Location'] == location]

        plt.figure(figsize=(8, 6))

        if not df.empty:
            sns.scatterplot(data=df, x='SST (°C)', y='pH Level', hue='Marine Heatwave')
            #plt.title('SST vs pH Level')
        else:
            plt.text(0.5, 0.5, 'No data available', horizontalalignment='center', 
                     verticalalignment='center', transform=plt.gca().transAxes)

        plt.tight_layout()
        return plt.gcf()
    
    @output
    @render.plot
    def ph_species():
        date_range_val = input.date_range_slider()
        start_date = pd.to_datetime(date_range_val[0]) if date_range_val[0] else date_range[0]
        end_date = pd.to_datetime(date_range_val[1]) if date_range_val[1] else date_range[1]
        location = input.location_selector()

        df = ocean_climate[
            (ocean_climate['Date'] >= start_date) &
            (ocean_climate['Date'] <= end_date)
        ]
        if location != "All Locations":
            df = df[df['Location'] == location]

        # Filter out rows with invalid or missing Species Observed
        df = df.copy()
        df = df[df['Species Observed'].notna()]

        plt.figure(figsize=(8, 6))

        if not df.empty:
            sns.scatterplot(data=df, x='pH Level', y='Species Observed', hue='Bleaching Severity')
            #plt.title('pH Level vs Species Observed')
        else:
            plt.text(0.5, 0.5, 'No data available', horizontalalignment='center', 
                     verticalalignment='center', transform=plt.gca().transAxes)

        plt.tight_layout()
        return plt.gcf() 

    @reactive.effect
    @reactive.event(input.reset)
    def reset_filters():
        ui.update_date_range(
            "date_range_slider",
            start=date_range[0],
            end=date_range[1],
        )
        ui.update_select("location_selector", selected="All Locations")

# Include CSS
ui.include_css(app_dir / "styles.css")

# Create app
www_dir = Path(__file__).parent / "www"
print(f"Serving static assets from: {www_dir.resolve()}")
app = App(app_ui, server,static_assets=www_dir)