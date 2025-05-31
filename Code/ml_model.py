import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Loading data
df = pd.read_csv('realistic_ocean_climate_dataset.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
def assign_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'
df['Season'] = df['Month'].apply(assign_season)
df['Bleaching Severity'] = df['Bleaching Severity'].replace("None", np.nan)
df['Marine Heatwave'] = df['Marine Heatwave'].astype(int)
df_encoded = pd.get_dummies(df, columns=['Season', 'Location'], drop_first=True)
severity_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
df_encoded['Bleaching Severity'] = df_encoded['Bleaching Severity'].map(severity_mapping)
df_clean = df_encoded.dropna(subset=['Bleaching Severity']).copy()
df_clean['Bleaching Severity'] = df_clean['Bleaching Severity'].astype(int)

columns_to_scale = ['SST (°C)', 'pH Level', 'Species Observed', 'Latitude', 'Longitude', 'Year', 'Month']
scaler = MinMaxScaler()
df_clean[columns_to_scale] = scaler.fit_transform(df_clean[columns_to_scale])

# Original location and anomalies
original_locations = df.loc[df_clean.index, 'Location'].reset_index(drop=True)
df_clean = df_clean.reset_index(drop=True)
df_clean['Location'] = original_locations
location_avg_sst = df_clean.groupby('Location')['SST (°C)'].transform('mean')
df_clean['SST_Anomaly'] = df_clean['SST (°C)'] - location_avg_sst
location_avg_ph = df_clean.groupby('Location')['pH Level'].transform('mean')
df_clean['pH_Anomaly'] = df_clean['pH Level'] - location_avg_ph

# Property and target variable
X = df_clean.drop(columns=['Bleaching Severity', 'Location', 'Date'])
y = df_clean['Bleaching Severity']

#2. Class balance with Upsample
df_balancing = X.copy()
df_balancing['Bleaching Severity'] = y
class_0 = df_balancing[df_balancing['Bleaching Severity'] == 0]
class_1 = df_balancing[df_balancing['Bleaching Severity'] == 1]
class_2 = df_balancing[df_balancing['Bleaching Severity'] == 2]
max_size = max(len(class_0), len(class_1), len(class_2))
class_0_upsampled = resample(class_0, replace=True, n_samples=max_size, random_state=42)
class_1_upsampled = resample(class_1, replace=True, n_samples=max_size, random_state=42)
class_2_upsampled = resample(class_2, replace=True, n_samples=max_size, random_state=42)
df_upsampled = pd.concat([class_0_upsampled, class_1_upsampled, class_2_upsampled])
X_balanced = df_upsampled.drop(columns=['Bleaching Severity'])
y_balanced = df_upsampled['Bleaching Severity']

# 3. Modeling
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_

y_pred = best_rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))

def save_confusion_matrix(best_rf, X_test, y_test, y_pred, filename='confusion_matrix.png', dpi=300):
    """
    Create and save confusion matrix as a separate file
    """
    
    # Create figure for confusion matrix only
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                ax=ax, cbar_kws={'label': 'Number of Samples'},
                square=True, linewidths=1, annot_kws={'size': 14})
    
    # Styling
    # ax.set_title('Confusion Matrix\nCoral Bleaching Severity Prediction', 
    #              fontsize=20, fontweight='bold', pad=30)
    ax.set_xlabel('Predicted Bleaching Severity', fontsize=16, fontweight='bold')
    ax.set_ylabel('Actual Bleaching Severity', fontsize=16, fontweight='bold')
    
    # Custom class labels
    class_labels = ['Low (0)', 'Medium (1)', 'High (2)']
    ax.set_xticklabels(class_labels, rotation=0, fontsize=14)
    ax.set_yticklabels(class_labels, rotation=0, fontsize=14)
    
    # Add accuracy text box
    accuracy = accuracy_score(y_test, y_pred)
    textstr = f'Overall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    ax.text(0.5, -0.15, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', horizontalalignment='center',
            bbox=props, fontweight='bold')
    
    # Add sample counts for each class
    class_counts = np.bincount(y_test)
    info_text = f'Test Set: Low={class_counts[0]}, Medium={class_counts[1]}, High={class_counts[2]} samples'
    # ax.text(0.5, -0.25, info_text, transform=ax.transAxes, fontsize=12,
    #         verticalalignment='top', horizontalalignment='center',
    #         style='italic')
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(filename, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"✅ Confusion matrix saved as: {filename}")
    
    plt.show()
    return fig

def save_feature_importance(best_rf, X_test, filename='feature_importance.png', dpi=300, top_n=None):
    """
    Create and save feature importance plot as a separate file
    """
    
    # Get feature importance
    importances = best_rf.feature_importances_
    feature_names = X_test.columns.tolist()
    
    # Create DataFrame and sort
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=True)
    
    # Select top N features if specified
    if top_n:
        importance_df = importance_df.tail(top_n)
        title_suffix = f" (Top {top_n})"
    else:
        title_suffix = ""
    
    # Create figure for feature importance only
    fig, ax = plt.subplots(figsize=(12, max(8, len(importance_df) * 0.4)))
    
    # Create color gradient
    colors = plt.cm.viridis(np.linspace(0.2, 1, len(importance_df)))
    
    # Create horizontal bar plot
    bars = ax.barh(range(len(importance_df)), importance_df['importance'], 
                   color=colors, edgecolor='black', alpha=0.8, linewidth=0.8)
    
    # Styling
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df['feature'], fontsize=11)
    ax.set_xlabel('Feature Importance Score', fontsize=16, fontweight='bold')
    #ax.set_title(f'Feature Importance{title_suffix}\nCoral Bleaching Severity Prediction', 
                 #fontsize=18, fontweight='bold', pad=30)
    
    # Add value labels on bars
    for i, (bar, importance) in enumerate(zip(bars, importance_df['importance'])):
        width = bar.get_width()
        ax.text(width + max(importance_df['importance']) * 0.01, 
                bar.get_y() + bar.get_height()/2, 
                f'{importance:.4f}', ha='left', va='center', 
                fontsize=10, fontweight='bold')
    
    # Add grid and styling
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_xlim(0, max(importance_df['importance']) * 1.15)
    
    # Add total features info
    total_features = len(X_test.columns)
    info_text = f'Total Features: {total_features} | Model: Random Forest'
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(filename, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"✅ Feature importance plot saved as: {filename}")
    
    plt.show()
    return fig, importance_df

def save_both_plots(best_rf, X_test, y_test, y_pred, 
                   cm_filename='confusion_matrix.png', 
                   fi_filename='feature_importance.png',
                   dpi=300, top_features=None):
    """
    Save both plots with custom filenames and settings
    """
    
    print("🎨 Generating and saving visualizations...")
    print("-" * 50)
    
    # Save confusion matrix
    fig1 = save_confusion_matrix(best_rf, X_test, y_test, y_pred, 
                                filename=cm_filename, dpi=dpi)
    
    # Save feature importance
    fig2, importance_df = save_feature_importance(best_rf, X_test, 
                                                 filename=fi_filename, 
                                                 dpi=dpi, top_n=top_features)
    
    print("-" * 50)
    print("✨ All visualizations saved successfully!")
    
    return fig1, fig2, importance_df

# ============ USAGE EXAMPLES ============

# Option 1: Save with default settings
# save_both_plots(best_rf, X_test, y_test, y_pred)

# Option 2: Save with custom filenames and high DPI
# save_both_plots(best_rf, X_test, y_test, y_pred, 
#                cm_filename='coral_bleaching_confusion_matrix.png',
#                fi_filename='coral_bleaching_feature_importance.png',
#                dpi=600)

# Option 3: Save feature importance with only top 15 features
# save_both_plots(best_rf, X_test, y_test, y_pred,
#                cm_filename='confusion_matrix_high_res.png',
#                fi_filename='top15_feature_importance.png',
#                dpi=400, top_features=15)

# Option 4: Save them separately with different settings
# save_confusion_matrix(best_rf, X_test, y_test, y_pred, 
#                      filename='confusion_matrix_publication.png', dpi=600)
# 
# save_feature_importance(best_rf, X_test, 
#                        filename='all_features_importance.png', dpi=600)

# Quick save with default names:
save_both_plots(best_rf, X_test, y_test, y_pred)