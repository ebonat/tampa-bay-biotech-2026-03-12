import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
from typing import Any
import joblib
import os

def save_pickle(obj: Any, file_path: str, protocol: int = pickle.HIGHEST_PROTOCOL) -> None:
    """
    Serialize (pickle) a Python object to a file.

    Parameters:
        obj (Any): Python object to serialize
        file_path (str): Path to save the pickle file
        protocol (int): Pickle protocol version (default: highest available)
    """
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f, protocol=protocol)
        print(f"Object successfully saved to {file_path}")
    except Exception as e:
        print(f"Error saving pickle file: {e}")


def load_pickle(file_path: str) -> Any:
    """
    Deserialize (unpickle) a Python object from a file.

    Parameters:
        file_path (str): Path to the pickle file

    Returns:
        Any: Loaded Python object
    """
    try:
        with open(file_path, 'rb') as f:
            obj = pickle.load(f)
        print(f"Object successfully loaded from {file_path}")
        return obj
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return None


# ============================================================================
# FEATURE ENGINEERING FUNCTION
# ============================================================================
def engineer_features(df):
    """Add engineered features for better LumA/LumB separation"""
    df = df.copy()
    
    # 1. Proliferation score (weighted by importance)
    proliferation_genes = ['MKI67_expr', 'CCNB1_expr', 'PTTG1_expr', 'UBE2C_expr', 
                           'CEP55_expr', 'UBE2T_expr', 'CDC20_expr', 'CCNE2_expr',
                           'KIF4A_expr', 'TYMS_expr', 'MELK_expr', 'NDC80_expr']
    available = [g for g in proliferation_genes if g in df.columns]
    
    df['proliferation_score'] = df[available].mean(axis=1)
    
    # Weighted version (MKI67 is most important)
    if all(f in df.columns for f in ['MKI67_expr', 'CCNB1_expr', 'PTTG1_expr', 'UBE2C_expr']):
        df['proliferation_score_weighted'] = (
            df['MKI67_expr'] * 2.5 +      # Highest weight
            df['CCNB1_expr'] * 1.5 +
            df['PTTG1_expr'] * 1.5 +
            df['UBE2C_expr'] * 1.5 +
            df['CEP55_expr'] * 1.3
        ) / 8.3
    
    # 2. Hormone receptor score
    if 'ESR1_expr' in df.columns and 'PGR_expr' in df.columns:
        df['hormone_receptor_score'] = (df['ESR1_expr'] + df['PGR_expr']) / 2
        df['ER_PR_product'] = df['ESR1_expr'] * df['PGR_expr']
    
    # 3. KEY RATIO FEATURES (Critical for LumA/LumB)
    if 'proliferation_score' in df.columns and 'hormone_receptor_score' in df.columns:
        df['prolif_to_hormone_ratio'] = df['proliferation_score'] / (df['hormone_receptor_score'].abs() + 1)
    
    if 'MKI67_expr' in df.columns:
        if 'ESR1_expr' in df.columns:
            df['MKI67_to_ESR1_ratio'] = df['MKI67_expr'] / (df['ESR1_expr'].abs() + 1)
        if 'PGR_expr' in df.columns:
            df['MKI67_to_PGR_ratio'] = df['MKI67_expr'] / (df['PGR_expr'].abs() + 1)
    
    # 4. Cell cycle score
    cell_cycle_genes = ['CCNB1_expr', 'CCNE1_expr', 'CCNE2_expr']
    available_cc = [g for g in cell_cycle_genes if g in df.columns]
    if len(available_cc) > 0:
        df['cell_cycle_score'] = df[available_cc].mean(axis=1)
    
    # 5. Interaction features
    if 'ESR1_expr' in df.columns and 'MKI67_expr' in df.columns:
        df['ESR1_x_MKI67'] = df['ESR1_expr'] * df['MKI67_expr']
    if 'PGR_expr' in df.columns and 'MKI67_expr' in df.columns:
        df['PGR_x_MKI67'] = df['PGR_expr'] * df['MKI67_expr']
    
    # 6. Non-linear features
    if 'MKI67_expr' in df.columns:
        df['MKI67_squared'] = df['MKI67_expr'] ** 2
        df['MKI67_abs'] = df['MKI67_expr'].abs()
    
    if 'proliferation_score' in df.columns:
        df['proliferation_squared'] = df['proliferation_score'] ** 2
    
    return df

# Path to this script's directory
dir = os.path.dirname(os.path.abspath(__file__))

csv_file_path = r"G:\Visual WWW\Python\1000_python_workspace_new\multi-omics_dataset_projects\csv\yes_outliers\metabric_data_preprocess_yes_outliers_synthetic_30000_minus_5_classes.csv"

df_metabric = pd.read_csv(csv_file_path)
df_metabric.drop(columns=["Sample_ID"], inplace=True)
print(df_metabric.shape)

print("="*70)
print("TRAINING WITH FEATURE ENGINEERING")
print("="*70)
print(f"\nOriginal shape: {df_metabric.shape}")


# Apply feature engineering
df_metabric = engineer_features(df_metabric)
print(f"After feature engineering: {df_metabric.shape}")
print(df_metabric.shape)
# exit()

# Show new features
new_features = [col for col in df_metabric.columns if 'score' in col or 'ratio' in col or 'squared' in col or '_x_' in col]
print(f"\nNew engineered features ({len(new_features)}):")
for feat in new_features:
    print(f"  - {feat}")

y_original = df_metabric['PAM50'].value_counts()
print(f"\nClass distribution:")
print(y_original)

class_names = y_original.index.tolist()

X_metabric = df_metabric.drop(columns=['PAM50', 'PAM50_Label'])
y_metabric = df_metabric['PAM50']

# Split data
X_metabric_train, X_metabric_temp, y_metabric_train, y_metabric_temp = train_test_split(
    X_metabric, y_metabric, test_size=0.2, random_state=50, stratify=y_metabric)
X_metabric_val, X_metabric_test, y_metabric_val, y_metabric_test = train_test_split(
    X_metabric_temp, y_metabric_temp, test_size=0.5, random_state=50, stratify=y_metabric_temp)

X_metabric_train = X_metabric_train.astype(float)
X_metabric_val = X_metabric_val.astype(float)
X_metabric_test = X_metabric_test.astype(float)

print('\nmetabric before smote:', Counter(y_metabric_train))

# Apply SMOTE
smote = SMOTE(random_state=50)
X_metabric_train, y_metabric_train = smote.fit_resample(X_metabric_train, y_metabric_train)
print('metabric after smote:', Counter(y_metabric_train))

X_train, y_train = X_metabric_train, y_metabric_train
X_val, y_val = X_metabric_val, y_metabric_val
X_test, y_test = X_metabric_test, y_metabric_test

# Train Random Forest with engineered features
print("\n" + "="*70)
print("TRAINING RANDOM FOREST")
print("="*70)

rf = RandomForestClassifier(
    class_weight='balanced', 
    random_state=50, 
    n_jobs=-1, 
    max_depth=15,          # Increased from 12
    min_samples_leaf=10,   # Decreased from 20
    min_samples_split=30,  # Decreased from 50
    max_features='sqrt'
)

rf.fit(X_train, y_train)
print(X_train.shape)

# PKL THIS RF MODEL HERE...
save_pickle(rf, "random_forest_model.pkl")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 25 Most Important Features:")
print(feature_importance.head(25).to_string(index=False))

# Check if engineered features are important
engineered_in_top20 = feature_importance.head(20)[
    feature_importance.head(20)['Feature'].str.contains('score|ratio|squared|_x_|_abs')
]
print(f"\nEngineered features in top 20: {len(engineered_in_top20)}")
if len(engineered_in_top20) > 0:
    print(engineered_in_top20[['Feature', 'Importance']].to_string(index=False))

# Predictions
y_train_pred = rf.predict(X_train)
y_val_pred = rf.predict(X_val) 
y_test_pred = rf.predict(X_test)

# Evaluation
print("\n" + "="*70)
print("MODEL PERFORMANCE")
print("="*70)

train_acc = accuracy_score(y_train, y_train_pred)
print(f"\nTrain accuracy: {train_acc * 100:.2f}%")

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=50)
cv_scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
print(f"10-Fold CV: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

val_acc = accuracy_score(y_val, y_val_pred)
print(f"validation accuracy: {val_acc * 100:.2f}%")

print("\nvalidation confusion matrix:")
print(confusion_matrix(y_test, y_test_pred))

print("\nvalidation classification report:")
print(classification_report(y_test, y_test_pred, target_names=class_names))

test_acc = accuracy_score(y_test, y_test_pred)
print(f"test accuracy: {test_acc * 100:.2f}%")

print("\ntest confusion matrix:")
print(confusion_matrix(y_test, y_test_pred))

print("\ntest classification report:")
print(classification_report(y_test, y_test_pred, target_names=class_names))

# ============================================================================
# PRODUCTION DATA WITH CONFIDENCE THRESHOLDS
# ============================================================================
print("\n" + "="*70)
print("PRODUCTION DATA TESTING (WITH ENGINEERED FEATURES)")
print("="*70)


df_production = pd.read_csv(csv_file_path)
# df_metabric = df_metabric[(df_metabric["Sample_ID"] != "NC")]

# Apply same feature engineering
df_production = engineer_features(df_production)

y_classes = df_production["PAM50"].to_list()
print("\nTrue labels:", y_classes)

X_production = df_production.drop(columns=['PAM50', 'PAM50_Label'])
X_production = X_production.astype(float)
print(X_production.shape)

# Make predictions
y_prediction = rf.predict(X_production)
prediction_probs = rf.predict_proba(X_production)

print("Predicted labels:", list(y_prediction))

# Get model's class order
model_classes = rf.classes_

print("\n" + "="*70)
print("DETAILED PREDICTION ANALYSIS")
print("="*70)

for i in range(len(y_classes)):
    true_label = y_classes[i]
    pred_label = y_prediction[i]
    probs = prediction_probs[i]
    
    class_prob_dict = {model_classes[j]: probs[j] for j in range(len(model_classes))}
    sorted_probs = sorted(class_prob_dict.items(), key=lambda x: x[1], reverse=True)
    
    match_status = "✓ CORRECT" if true_label == pred_label else "✗ WRONG"
    
    print(f"\nSample {i+1}: {match_status}")
    print(f"  True:      {true_label}")
    print(f"  Predicted: {pred_label}")
    print(f"  Probabilities:")
    for class_name, prob in sorted_probs:
        marker = " <--- PREDICTED" if class_name == pred_label else ""
        bar = "█" * int(prob * 40)
        print(f"    {class_name:10s}: {prob:.4f} {bar}{marker}")
    
    # Confidence analysis
    top_2 = sorted_probs[:2]
    confidence_gap = top_2[0][1] - top_2[1][1]
    
    if confidence_gap < 0.20:
        print(f"  ⚠ LOW CONFIDENCE! Gap: {confidence_gap:.4f}")
        print(f"    Consider: {top_2[1][0]} (prob: {top_2[1][1]:.4f})")
    
    # LumA/LumB specific check
    if 'LumA' in class_prob_dict and 'LumB' in class_prob_dict:
        luma_prob = class_prob_dict['LumA']
        lumb_prob = class_prob_dict['LumB']
        diff = abs(luma_prob - lumb_prob)
        
        if diff < 0.20:
            print(f"  ⚠ UNCERTAIN LumA/LumB (diff: {diff:.4f})")
            
            # Show key feature values for this sample
            if i == 1:  # Sample 2
                print(f"\n  Key features for Sample 2:")
                key_feats = ['MKI67_expr', 'proliferation_score_weighted', 'prolif_to_hormone_ratio']
                for feat in key_feats:
                    if feat in X_production.columns:
                        val = X_production.iloc[i][feat]
                        print(f"    {feat}: {val:.3f}")

# Save model
joblib.dump(rf, os.path.join(dir, 'rf_model_engineered.pkl'))
print("\n✓ Model saved: rf_model_engineered.pkl")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print("""
Sample 2 Analysis:
- Has very LOW MKI67 (-1.1), which is characteristic of LumA
- Low proliferation markers overall  
- This sample is biologically ambiguous - it could be:
  1. A mislabeled sample (should be LumA)
  2. A LumB with unusually low proliferation
  3. A borderline case requiring additional clinical context

The model is actually making a reasonable prediction based on the
molecular features. Consider:
- Getting additional markers (Ki-67 IHC, grade, etc.)
- Using confidence thresholds to flag uncertain cases
- Expert review for borderline samples
""")
