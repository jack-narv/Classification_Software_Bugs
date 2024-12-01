import streamlit as st
import joblib
import ast
from radon.complexity import cc_visit
from radon.raw import analyze

# Cargar el modelo entrenado
clf = joblib.load('bug_classifier_model.pkl')

# Función para calcular las métricas de un archivo
def analyze_code(file_content):
    try:
        # Validar el contenido del archivo
        ast.parse(file_content)
        
        complexity = sum([block.complexity for block in cc_visit(file_content)])
        num_lines = analyze(file_content).loc
        num_functions = len(cc_visit(file_content))
        return complexity, num_lines, num_functions
    except SyntaxError as e:
        st.error(f"Syntax error in file: {e}")
        return None, None, None
    except Exception as e:
        st.error(f"Error analyzing file: {e}")
        return None, None, None

# Función para predecir si un archivo tiene un bug
def predict_bug(file_content):
    complexity, num_lines, num_functions = analyze_code(file_content)
    
    if complexity is not None:
        features = [[complexity, num_lines, num_functions]]
        prediction = clf.predict(features)
        return prediction[0]
    else:
        return None

# Interfaz de Streamlit
st.title("Bug Prediction in Python Files")
st.write("Upload a Python file to predict if it contains a bug.")

uploaded_file = st.file_uploader("Choose a .py file", type="py")

if uploaded_file is not None:
    file_content = uploaded_file.read().decode("utf-8")
    st.code(file_content, language="python")
    
    is_bug = predict_bug(file_content)
    if is_bug is not None:
        if is_bug == 1:
            st.error("The file is predicted to contain a bug.")
        else:
            st.success("The file is predicted to not contain a bug.")
    else:
        st.error("Could not analyze the file.")
