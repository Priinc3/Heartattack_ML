# Second attempt: create the improved notebook file. This will write /mnt/data/heart_disease_improved_pipeline.ipynb
import nbformat as nbf
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
from pathlib import Path

notebook_path = "heart_disease_Chatgpt.ipynb"
csv_candidates = ["final.csv", "heart_2020_cleaned.csv"]

nb = new_notebook()
cells = []

cells.append(new_markdown_cell("# Heart Disease Prediction — Improved pipeline\n\nThis notebook loads the dataset, performs EDA, cleaning, feature engineering, handles class imbalance, trains multiple models with hyperparameter tuning, evaluates them, and saves the best pipelines and a results summary.\n\nRun cells sequentially. Created by ChatGPT per your request."))

cells.append(new_code_cell(
"from pathlib import Path\ncsv_candidates = %r\ncsv_path = None\nfor p in csv_candidates:\n    if Path(p).exists():\n        csv_path = p\n        break\nif csv_path is None:\n    raise FileNotFoundError(f'None of the expected files found: {csv_candidates}')\nprint('Using dataset:', csv_path)\n\nimport pandas as pd\ndf = pd.read_csv(csv_path)\nprint('Loaded shape:', df.shape)\ndf.head()" % csv_candidates))

cells.append(new_code_cell(
"# EDA\nimport numpy as np\nprint(df.dtypes.value_counts())\nprint('\\nSummary stats:')\ndisplay(df.describe(include='all'))\nprint('\\nMissing values:')\nprint(df.isnull().sum())\n\npossible_targets = [c for c in df.columns if 'heart' in c.lower() or 'target' in c.lower() or 'disease' in c.lower()]\nprint('Possible targets:', possible_targets)\nif 'heart_disease' in df.columns:\n    target_col = 'heart_disease'\nelif 'HeartDisease' in df.columns:\n    target_col = 'HeartDisease'\nelif possible_targets:\n    target_col = possible_targets[0]\nelse:\n    raise ValueError('Could not detect target column')\nprint('Target ->', target_col)\nprint(df[target_col].value_counts())\n\nnum_df = df.select_dtypes(include=['number'])\nif target_col in num_df.columns:\n    corr = num_df.corr()\n    print('\\nTop correlations with target:')\n    print(corr[target_col].abs().sort_values(ascending=False).head(10))\nelse:\n    print('Target not numeric; skipping numeric-target correlations')"
))

cells.append(new_code_cell(
"# Class distribution plot\nimport matplotlib.pyplot as plt\ncounts = df[target_col].value_counts()\nplt.figure(figsize=(5,4))\nplt.bar(counts.index.astype(str), counts.values)\nplt.title('Target class distribution')\nplt.xlabel('Class')\nplt.ylabel('Count')\nplt.show()"
))

cells.append(new_code_cell(
"# Cleaning & split\n# Drop duplicates\nbefore = df.shape[0]\ndf = df.drop_duplicates()\nafter = df.shape[0]\nprint('Dropped', before-after, 'duplicates')\n\n# Impute missing\nfor c in df.columns:\n    if df[c].isnull().sum()>0:\n        if df[c].dtype.kind in 'biufc':\n            df[c] = df[c].fillna(df[c].median())\n        else:\n            df[c] = df[c].fillna(df[c].mode()[0])\nprint('Total missing after imputation:', df.isnull().sum().sum())\n\nX = df.drop(columns=[target_col]).copy()\ny = df[target_col].copy()\nfrom sklearn.model_selection import train_test_split\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)\nprint('Train/Test shapes:', X_train.shape, X_test.shape)"
))

cells.append(new_code_cell(
"# Preprocessor\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.compose import ColumnTransformer\nfrom sklearn.preprocessing import OneHotEncoder, StandardScaler\nnum_cols = X.select_dtypes(include=['number']).columns.tolist()\ncat_cols = X.select_dtypes(exclude=['number']).columns.tolist()\nprint('num_cols', num_cols)\nprint('cat_cols', cat_cols)\n\nnumeric_transformer = Pipeline([('scaler', StandardScaler())])\ncat_transformer = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))])\npreprocessor = ColumnTransformer([('num', numeric_transformer, num_cols), ('cat', cat_transformer, cat_cols)])\npreprocessor.fit(X_train)\nprint('Preprocessor fitted. Transformed feature count:', preprocessor.transform(X_train).shape[1])"
))

cells.append(new_code_cell(
"# SMOTE resampling (if imblearn available)\nfrom collections import Counter\nprint('Before:', Counter(y_train))\ntry:\n    from imblearn.over_sampling import SMOTE\n    X_train_trans = preprocessor.transform(X_train)\n    sm = SMOTE(random_state=42)\n    X_res, y_res = sm.fit_resample(X_train_trans, y_train)\n    print('After SMOTE:', Counter(y_res))\n    smote_used = True\nexcept Exception as e:\n    print('SMOTE not available or failed:', e)\n    X_res = preprocessor.transform(X_train)\n    y_res = y_train.values\n    smote_used = False\n\nprint('X_res shape:', X_res.shape)\n"
))

cells.append(new_code_cell(
"# Baseline CV for multiple models\nfrom sklearn.model_selection import cross_val_score, StratifiedKFold\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier\nfrom sklearn.svm import SVC\nfrom sklearn.neighbors import KNeighborsClassifier\nimport numpy as np\n\nmodels = {\n    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),\n    'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42),\n    'GradientBoosting': GradientBoostingClassifier(n_estimators=200, random_state=42),\n    'HistGradientBoosting': HistGradientBoostingClassifier(random_state=42),\n    'SVC': SVC(probability=True, random_state=42),\n    'KNeighbors': KNeighborsClassifier()\n}\ncv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\nresults = []\nfor name, model in models.items():\n    try:\n        if smote_used:\n            scores = cross_val_score(model, X_res, y_res, cv=cv, scoring='f1')\n        else:\n            pipe = Pipeline([('preprocessor', preprocessor), ('clf', model)])\n            scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='f1')\n        print(name, 'F1:', np.round(scores,4), 'mean=', np.round(scores.mean(),4))\n        results.append({'model': name, 'f1_mean': float(scores.mean())})\n    except Exception as e:\n        print('Failed', name, e)\n\nimport pandas as pd\npd.DataFrame(results).sort_values('f1_mean', ascending=False)\n"
))

cells.append(new_code_cell(
"# Hyperparameter tuning with RandomizedSearchCV for RandomForest\nfrom sklearn.model_selection import RandomizedSearchCV\nfrom sklearn.ensemble import RandomForestClassifier\n\nrf = RandomForestClassifier(random_state=42)\nrf_param_dist = {\n    'n_estimators': [100,200,400],\n    'max_depth': [None, 5, 10, 20],\n    'min_samples_split': [2,5,10],\n    'min_samples_leaf': [1,2,4],\n    'class_weight': [None, 'balanced']\n}\n\ntry:\n    if smote_used:\n        rf_search = RandomizedSearchCV(rf, rf_param_dist, n_iter=20, scoring='f1', cv=cv, random_state=42, n_jobs=-1)\n        rf_search.fit(X_res, y_res)\n    else:\n        rf_pipe = Pipeline([('preprocessor', preprocessor), ('rf', rf)])\n        rf_param_pipe = {f'rf__{k}': v for k,v in rf_param_dist.items()}\n        rf_search = RandomizedSearchCV(rf_pipe, rf_param_pipe, n_iter=20, scoring='f1', cv=cv, random_state=42, n_jobs=-1)\n        rf_search.fit(X_train, y_train)\n    print('Best RF score:', rf_search.best_score_)\n    print('Best RF params:', rf_search.best_params_)\n    best_rf = rf_search.best_estimator_\nexcept Exception as e:\n    print('RF tuning failed:', e)\n    best_rf = None\n"
))

cells.append(new_code_cell(
"# Final evaluation of best RF on test set\nfrom sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report\n\nfinal_results = []\nif best_rf is not None:\n    try:\n        if smote_used:\n            X_test_trans = preprocessor.transform(X_test)\n            y_pred = best_rf.predict(X_test_trans)\n            y_proba = best_rf.predict_proba(X_test_trans)[:,1] if hasattr(best_rf, 'predict_proba') else None\n        else:\n            y_pred = best_rf.predict(X_test)\n            y_proba = best_rf.predict_proba(X_test)[:,1] if hasattr(best_rf, 'predict_proba') else None\n        acc = accuracy_score(y_test, y_pred)\n        prec = precision_score(y_test, y_pred, zero_division=0)\n        rec = recall_score(y_test, y_pred, zero_division=0)\n        f1 = f1_score(y_test, y_pred, zero_division=0)\n        roc = roc_auc_score(y_test, y_proba) if y_proba is not None else None\n        print('RF test accuracy:', acc)\n        print('RF test precision:', prec)\n        print('RF test recall:', rec)\n        print('RF test f1:', f1)\n        print('RF test roc:', roc)\n        print('Confusion matrix:')\n        print(confusion_matrix(y_test, y_pred))\n        print('Classification report:')\n        print(classification_report(y_test, y_pred, zero_division=0))\n        final_results.append({'model': 'RandomForest', 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'roc_auc': roc})\n    except Exception as e:\n        print('Final evaluation failed', e)\nelse:\n    print('No tuned RF available to evaluate')\n\nimport pandas as pd\npd.DataFrame(final_results)\n"
))

cells.append(new_code_cell(
"# Save best model and results if available\nimport joblib\nmodels_dir = Path('/mnt/data/improved_models')\nmodels_dir.mkdir(parents=True, exist_ok=True)\nif best_rf is not None:\n    try:\n        joblib.dump(best_rf, models_dir / 'best_random_forest.joblib')\n        print('Saved best_random_forest.joblib')\n    except Exception as e:\n        print('Saving failed', e)\n\n# Save results CSV\ntry:\n    pd.DataFrame(final_results).to_csv('/mnt/data/improved_model_results_summary.csv', index=False)\n    print('Saved improved_model_results_summary.csv')\nexcept Exception as e:\n    print('Saving results failed', e)\n"
))

nb['cells'] = cells

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

notebook_path
