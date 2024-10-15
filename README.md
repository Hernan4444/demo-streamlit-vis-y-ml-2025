# Demo Streamlit - Visualización y ML

El código de este directorio está pensado para enseñar las funcionalidades básicas de Streamlit de
acuerdo a lo mostrado en clases. En particular, usaremos un dataset de Airbnb.

## Estructura de archivos

1. `Airbnb_Locations`: _dataset_ para desarrollar el _dashboard_.
2. `app.py`: solución _dashboard_.
3. `constantes.py`: archivo .py con algunas constantes a utilizar en `app.py`
4. `requirements.txt`: librerías de Python necesarias para construir el dashboard.
5. `EntrenarPipeline.ipynb`: _notebook_ para entrenar modelo y guardarlo como un archivo.
6. `pipeline_model.pkl`: modelo entrenado para su posterior uso.
7. `README.md`: este archivo con el detalle de la demo.
8. `.gitignore`: archivo de `git` para indicar qué cosas no se deben subir a un repositorio de Github.

## Cómo ejecutar
1. Instalar librerías: `pip install -r requirements.txt`.
2. Ejecutar: `streamlit run app.py`
