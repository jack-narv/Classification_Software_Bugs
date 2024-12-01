import re
import requests
from github import Github
from radon.complexity import cc_visit
from radon.metrics import mi_visit
from radon.raw import analyze
from flake8.api import legacy as flake8
import pydeps
import pandas as pd
import numpy as np
import ast
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder 
from prefect import task, flow
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier  
from sklearn.metrics import f1_score, precision_score, recall_score  
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV

# Configuración de autenticación y repositorio de GitHub
TOKEN_ANTIGUO = 'xxxxxxxxxxxxxxxxxxxxxx'
TOKEN = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
REPO_NAME = 'python/cpython'
REPOS = ['python/cpython', 'pandas-dev/pandas', 'django/django', 'scikit-learn/scikit-learn']


g = Github(TOKEN)  # TOKEN es tu token de GitHub
rate_limit = g.get_rate_limit()
print("Rate Limit: ",rate_limit)

# Función para clasificar el tipo de commit
def analyze_code(file_content, commit_message):
    bug_type = None  # Inicializamos sin tipo de error
    
    try:
        # Parsear el contenido del archivo para análisis de AST
        tree = ast.parse(file_content)

        # Si el archivo no arroja errores de sintaxis, seguimos con el análisis
        maintainability_index = mi_visit(file_content, True)
        complexity_blocks = cc_visit(file_content)
        complexity = sum([block.complexity for block in complexity_blocks])
        num_lines = analyze(file_content).loc
        num_functions = len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
        num_classes = len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])
        function_lengths = [len(list(ast.iter_child_nodes(node))) for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        avg_function_length = sum(function_lengths) / num_functions if num_functions > 0 else 0
        num_parameters_per_function = [len(node.args.args) for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        avg_num_parameters = sum(num_parameters_per_function) / num_functions if num_functions > 0 else 0
        nesting_levels = max([len(list(ast.iter_child_nodes(node))) for node in ast.walk(tree) if isinstance(node, (ast.If, ast.For, ast.While))], default=0)
        num_comments = sum([1 for line in file_content.split('\n') if line.strip().startswith('#')])
        duplicated_code_warnings = len(re.findall(r'(\bdef\b.+\bdef\b)', file_content))
        dependencies = set(re.findall(r'import (\w+)', file_content))
        num_imports = len(dependencies)
        cyclic_dependencies = 0  # Placeholder para análisis de dependencias cíclicas

        # Uso de herramientas de análisis estático (ejemplo con pylint)
        pylint_stdout, pylint_stderr = lint.py_run(file_content, return_std=True)
        pylint_output = pylint_stdout.getvalue()

        # Si pylint detecta problemas, puedes extraer tipos de errores de aquí.
        if "syntax-error" in pylint_output:
            bug_type = "SyntaxError"
        elif "undefined-variable" in pylint_output:
            bug_type = "NameError"
        elif "unused-import" in pylint_output:
            bug_type = "ImportError"
        elif "attribute-defined-outside-init" in pylint_output:
            bug_type = "AttributeError"
        
    except SyntaxError:
        bug_type = 'SyntaxError'
    except IndentationError:
        bug_type = 'IndentationError'
    except TabError:
        bug_type = 'TabError'
    except TypeError:
        bug_type = 'TypeError'
    except AttributeError:
        bug_type = 'AttributeError'
    except NameError:
        bug_type = 'NameError'
    except ValueError:
        bug_type = 'ValueError'
    except KeyError:
        bug_type = 'KeyError'
    except IndexError:
        bug_type = 'IndexError'
    except UnboundLocalError:
        bug_type = 'UnboundLocalError'
    except RuntimeError:
        bug_type = 'RuntimeError'
    except OverflowError:
        bug_type = 'OverflowError'
    except ZeroDivisionError:
        bug_type = 'ZeroDivisionError'
    except RecursionError:
        bug_type = 'RecursionError'
    except ArithmeticError:
        bug_type = 'ArithmeticError'
    except FloatingPointError:
        bug_type = 'FloatingPointError'
    except MemoryError:
        bug_type = 'MemoryError'
    except ReferenceError:
        bug_type = 'ReferenceError'
    except ImportError:
        bug_type = 'ImportError'
    except Exception as e:
        print(f"Unexpected error analyzing file: {e}")
        bug_type = 'UnknownError'

    # Mejorar clasificación basada en el mensaje del commit
    if re.search(r'syntax', commit_message, re.IGNORECASE):
        bug_type = 'SyntaxError'
    elif re.search(r'type', commit_message, re.IGNORECASE):
        bug_type = 'TypeError'
    elif re.search(r'attribute', commit_message, re.IGNORECASE):
        bug_type = 'AttributeError'
    elif re.search(r'name', commit_message, re.IGNORECASE):
        bug_type = 'NameError'
    elif re.search(r'index', commit_message, re.IGNORECASE):
        bug_type = 'IndexError'
    elif re.search(r'key', commit_message, re.IGNORECASE):
        bug_type = 'KeyError'
    elif re.search(r'import', commit_message, re.IGNORECASE):
        bug_type = 'ImportError'
    
    # Retornar las métricas junto con el tipo de bug identificado (si lo hay)
    return {
        'maintainability_index': maintainability_index if 'maintainability_index' in locals() else None,
        'complexity': complexity if 'complexity' in locals() else None,
        'complexity_per_function': complexity_blocks if 'complexity_blocks' in locals() else None,
        'num_lines': num_lines if 'num_lines' in locals() else None,
        'num_functions': num_functions if 'num_functions' in locals() else None,
        'num_classes': num_classes if 'num_classes' in locals() else None,
        'avg_function_length': avg_function_length if 'avg_function_length' in locals() else None,
        'avg_num_parameters': avg_num_parameters if 'avg_num_parameters' in locals() else None,
        'nesting_levels': nesting_levels if 'nesting_levels' in locals() else None,
        'num_comments': num_comments if 'num_comments' in locals() else None,
        'duplicated_code_warnings': duplicated_code_warnings if 'duplicated_code_warnings' in locals() else None,
        'num_imports': num_imports if 'num_imports' in locals() else None,
        'cyclic_dependencies': cyclic_dependencies if 'cyclic_dependencies' in locals() else None,
        'bug_type': bug_type
    }

@task
def fetch_balanced_commits_from_multiple_repos():
    all_commits = []
    bug_type_counts = {
        "SyntaxError": 0,
        "IndentationError": 0,
        "TabError": 0,
        "TypeError": 0,
        "AttributeError": 0,
        "NameError": 0,
        "ValueError": 0,
        "KeyError": 0,
        "IndexError": 0,
        "UnboundLocalError": 0,
        "RuntimeError": 0,
        "OverflowError": 0,
        "ZeroDivisionError": 0,
        "RecursionError": 0,
        "ArithmeticError": 0,
        "FloatingPointError": 0,
        "MemoryError": 0,
        "ReferenceError": 0,
        "ImportError": 0
    }
    
    # Definir la cantidad máxima de commits por tipo de bug
    max_commits_per_bug_type = 300  # Ajusta según las necesidades de tu proyecto
    
    for repo_name in REPOS:
        g = Github(TOKEN)
        repo = g.get_repo(repo_name)
        commits = repo.get_commits()
        
        for commit in commits:
            # Obtener el contenido del commit
            commit_message = commit.commit.message
            bug_type = classify_commit(commit_message)

            # Asegurarse de que solo añadimos el commit si necesitamos más ejemplos de este tipo de bug
            if bug_type and bug_type_counts.get(bug_type, 0) < max_commits_per_bug_type:
                selected_files = [file for file in commit.files if file.filename.endswith('.py')]
                for file in selected_files:
                    file_content = requests.get(file.raw_url).text
                    # Llamar a la función analyze_code para verificar el tipo de bug en el código
                    metrics = analyze_code(file_content, commit_message)
                    bug_detected = metrics['bug_type']
                    
                    if bug_detected and bug_type_counts[bug_detected] < max_commits_per_bug_type:
                        # Incrementar el contador para este tipo de bug
                        bug_type_counts[bug_detected] += 1
                        
                        # Agregar el commit a la lista de commits seleccionados
                        all_commits.append(commit)
                        
                        # Detener cuando hayamos alcanzado el máximo número de commits para todos los tipos de bugs
                        if all(bug_type_counts[bug] >= max_commits_per_bug_type for bug in bug_type_counts):
                            break

    return all_commits


# Función para clasificar el tipo de commit
def classify_commit(message):
    if re.search(r'\b(fix)\b', message, re.IGNORECASE):
        return 0
    elif re.search(r'\b(bug)\b', message, re.IGNORECASE):
        return 1
    else:
        return None


#Tarea para analizar outliers
@task
def process_outliers(df):
    Q1 = df['complexity'].quantile(0.25)
    Q3 = df['complexity'].quantile(0.75)
    IQR = Q3 - Q1
    
    # Definir límites
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filtrar outliers
    df_filtered = df[(df['complexity'] >= lower_bound) & (df['complexity'] <= upper_bound)]
    return df_filtered, lower_bound, upper_bound


# Tarea para aplicar logaritmo a la variable de complejidad (agregando 1 para evitar log(0))
@task
def variable_logarithm_complexity(df):
    df['complexity_log'] = np.log1p(df['complexity'])
    return df

# Imputar los outliers en 'complexity' con la mediana
@task 
def imput_outliers(df,lower_bound,upper_bound):
    complexity_median = df['complexity'].median()

    df['complexity'] = np.where((df['complexity'] < lower_bound) | (df['complexity'] > upper_bound),
                            complexity_median, df['complexity'])
    return df

@task
# Función para procesar commits y añadir todas las métricas
def process_commits(selected_commits):
    data = []
    errors = []

    for commit in selected_commits:
        sha = commit.sha
        commit_message = commit.commit.message  # Obtener el mensaje del commit
        is_bug = classify_commit(commit_message)  # Si es un bug o no
        if is_bug is None:
            continue

        files = commit.files
        for file in files:
            if file.filename.endswith('.py'):
                url = file.raw_url
                response = requests.get(url)
                file_content = response.text

                # Llamamos a la función analyze_code pasando el contenido del archivo y el mensaje del commit
                metrics = analyze_code(file_content, commit_message)

                if metrics:
                    # Guardamos el tipo de bug detectado y las métricas
                    data.append({
                        'commit_sha': sha,
                        'filename': file.filename,
                        'maintainability_index': metrics['maintainability_index'],
                        'complexity': metrics['complexity'],
                        'num_lines': metrics['num_lines'],
                        'num_functions': metrics['num_functions'],
                        'num_classes': metrics['num_classes'],
                        'avg_function_length': metrics['avg_function_length'],
                        'avg_num_parameters': metrics['avg_num_parameters'],
                        'nesting_levels': metrics['nesting_levels'],
                        'num_comments': metrics['num_comments'],
                        'duplicated_code_warnings': metrics['duplicated_code_warnings'],
                        'num_imports': metrics['num_imports'],
                        'cyclic_dependencies': metrics['cyclic_dependencies'],
                        'is_bug': is_bug,  # Si es un bug o no
                        'bug_type': metrics['bug_type']  # Tipo de bug identificado
                    })
                else:
                    # Si hubo errores durante el análisis del archivo
                    errors.append({
                        'commit_sha': sha,
                        'filename': file.filename,
                        'error': 'Error during analysis'
                    })

    # Convertimos los datos en un DataFrame
    df = pd.DataFrame(data)
    errors_df = pd.DataFrame(errors)

    # Guardamos los resultados en archivos CSV
    df.to_csv('bug_commit_dataset_with_types.csv', index=False)
    errors_df.to_csv('analysis_errors.csv', index=False)
    return df


# Función para hacer oversampling con SMOTE
def oversample_with_smote(X_train, y_train):
    smote = SMOTE(random_state=42, k_neighbors=2)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print("Oversampling realizado con SMOTE. Nuevo tamaño de datos:", X_resampled.shape)
    return X_resampled, y_resampled


# Función para hacer undersampling con RandomUnderSampler
def undersample(X_train, y_train):
    undersampler = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)
    print("Undersampling realizado. Nuevo tamaño de datos:", X_resampled.shape)
    return X_resampled, y_resampled


# Combina oversampling con SMOTE y undersampling
def balance_dataset(X_train, y_train):
    # Oversampling con SMOTE
    X_resampled, y_resampled = oversample_with_smote(X_train, y_train)
    
    # Undersampling
    X_resampled, y_resampled = undersample(X_resampled, y_resampled)
    
    return X_resampled, y_resampled


# Función para ajustar los hiperparámetros del modelo usando RandomizedSearchCV
def hyperparameter_tuning(X_train, y_train):
    # Definir el rango de hiperparámetros a explorar
    param_dist = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    # Inicializar el modelo
    clf = RandomForestClassifier(class_weight='balanced', random_state=42)

    # Inicializar RandomizedSearchCV
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist, 
                                       n_iter=50,  # Número de combinaciones a probar
                                       cv=3,  # Validación cruzada con 3 particiones
                                       verbose=2,  # Para obtener más detalles durante la ejecución
                                       random_state=42,
                                       n_jobs=-1,  # Utilizar todos los procesadores disponibles
                                       scoring='accuracy')

    # Ajustar el modelo a los datos de entrenamiento
    random_search.fit(X_train, y_train)

    # Devolver los mejores parámetros encontrados
    print(f"Mejores parámetros: {random_search.best_params_}")
    return random_search.best_estimator_


# Función para normalizar o estandarizar las características
def normalize_features(X_train, X_test):
    # Inicializar el escalador
    scaler = StandardScaler()

    # Ajustar el escalador y transformar los datos de entrenamiento
    X_train_scaled = scaler.fit_transform(X_train)

    # Transformar los datos de prueba
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled

@task
def train_model(df):
    imputer = SimpleImputer(strategy='mean')

    # Imputar valores nulos
    df[['complexity', 'num_lines', 'num_functions']] = imputer.fit_transform(df[['complexity', 'num_lines', 'num_functions']])
    
    # Verificar si hay valores nulos en bug_type y eliminarlos
    if df['bug_type'].isnull().any():
        print("Existen valores nulos en 'bug_type'. Eliminando filas...")
        df = df.dropna(subset=['bug_type'])
    
    # Codificar la variable bug_type
    label_encoder = LabelEncoder()
    df['bug_type'] = label_encoder.fit_transform(df['bug_type'])
    
    # Variables predictoras y objetivo
    X = df[['complexity', 'num_lines', 'num_functions', 'maintainability_index', 'num_classes', 
            'avg_function_length', 'nesting_levels', 'num_imports', 'num_comments']]
    y = df['bug_type']
    
    # División en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalización
    X_train_scaled, X_test_scaled = normalize_features(X_train, X_test)
    
    # Balancear el conjunto de datos
    X_resampled, y_resampled = balance_dataset(X_train_scaled, y_train)
    
    # Ajustar hiperparámetros (opcional)
    best_model = hyperparameter_tuning(X_resampled, y_resampled)

    # Predicciones
    y_pred = best_model.predict(X_test_scaled)
    
    # Evaluación
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    print("Accuracy:", accuracy)
    print("F1 Score:", f1)
    print("Precision:", precision)
    print("Recall:", recall)
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    
    # Guardar modelo entrenado
    joblib.dump(best_model, 'bug_type_xgboost_model.pkl')

    return accuracy, f1, precision, recall


@flow(log_prints=True)
def data_pipeline():
    commits = fetch_balanced_commits_from_multiple_repos()
    df_commits = process_commits(commits)
    df_commits = pd.read_csv('/teamspace/studios/this_studio/bug_commit_dataset_with_types.csv')
    df_outliers, lower_bound, upper_bound = process_outliers(df_commits)
    df_variable = variable_logarithm_complexity(df_outliers)
    df = imput_outliers(df_variable, lower_bound,upper_bound)
    accuracy, f1, precision, recall = train_model(df)

if __name__ == "__main__":
    # Ejecuta el flujo
    data_pipeline()