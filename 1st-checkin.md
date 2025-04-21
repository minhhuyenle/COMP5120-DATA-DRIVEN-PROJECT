# Shifting Seas: Ocean Climate & Marine Life Dataset

## Dataset Overview

The [**Shifting Seas: Ocean Climate & Marine Life Dataset**](https://www.kaggle.com/datasets/atharvasoundankar/shifting-seas-ocean-climate-and-marine-life-dataset/data) from Kaggle is a multi-year global dataset of sea surface temperature, pH levels, and corals. It provides a synthetic yet realistic view of marine environments from **2015 to 2023**, covering critical zones like the **Great Barrier Reef** and **Maldives**. The dataset includes sea surface temperature (SST), pH levels, coral bleaching severity, species observations, and marine heatwave indicators, simulating the impacts of climate change.

### Variables Included:
- **Date**: Date of observation  
- **Location**: Marine location name (e.g., Great Barrier Reef, Maldives)  
- **Latitude**: Latitude of the observation site  
- **Longitude**: Longitude of the observation site  
- **SST (°C)**: Sea Surface Temperature in degrees Celsius  
- **pH Level**: Acidity level of seawater (lower means more acidic, indicating acidification)  
- **Bleaching Severity**: Categorical variable — *None, Low, Medium, High*  
- **Species Observed**: Count of marine species observed during the sampling period  
- **Marine Heatwave**: Boolean flag (True/False), indicating whether SST > 30°C  

---

## Question

- How have rising **SST** and **ocean acidification** affected **coral bleaching** and **marine biodiversity** across time and regions? Can we detect **climate-driven marine heatwaves** using these trends?

---

## Importance

As climate change accelerates, the world's oceans are experiencing significant transformations. This question is vital because the health of coral reefs and surrounding marine life is a direct indicator of climate stress. Coral bleaching events often precede ecological collapse, and biodiversity loss can disrupt the marine food web, impacting global fisheries and coastal communities. Thdashboard now integrates visualizations and machine learning to provide predictive insights and identify environmental patterns, enhancing its utility for conservation and policy.

Some visualization could be use such as Line Chart (to show temporal changes in SST, pH, and species observed); Stacked Bar Chart (to visualize annual trends in bleaching severity and major contributing factors); Scatter Plot (to show environmental clustering results (e.g., regions with similar SST/pH profiles and bleaching risk); Heatmaps or Map Overlays (to visualize geographic distribution of marine heatwaves and bleaching severity).
 
---

## Visualization Challenges

Visualizing this dataset presents several challenges:

- **Multivariate Complexity**: Variables like SST, pH, bleaching severity, and species counts interact in non-linear ways  
- **Spatiotemporal Dynamics**: Patterns vary across both regions and time, requiring spatial and temporal analysis  
- **Categorical and Continuous Mix**: With both numeric (SST, pH) and categorical (bleaching severity) data, unified visual representation is non-trivial  
- **Event Detection**: Marine heatwaves, though Boolean, are critical events that must be highlighted in both time series and spatial maps

---
## Dashboard Wireframe
![Dashboard Wireframw](/Sketch.png)
